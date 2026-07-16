package vulki

import (
	"errors"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"time"
	"unsafe"

	"github.com/srlehn/vulki/vk"
)

const submissionTestTimeout = 2 * time.Second

type submissionReservationResult struct {
	reservation *submissionReservation
	err         error
}

func TestSubmissionSchedulerAllowsCompatibleAccess(t *testing.T) {
	tests := []struct {
		name   string
		first  func(*Buffer, *Buffer) submissionResources
		second func(*Buffer, *Buffer) submissionResources
	}{
		{
			name: "disjoint writes",
			first: func(first, _ *Buffer) submissionResources {
				return submissionResources{{buffer: first, access: submissionWrite}}
			},
			second: func(_, second *Buffer) submissionResources {
				return submissionResources{{buffer: second, access: submissionWrite}}
			},
		},
		{
			name: "shared reads",
			first: func(first, _ *Buffer) submissionResources {
				return submissionResources{{buffer: first, access: submissionRead}}
			},
			second: func(first, _ *Buffer) submissionResources {
				return submissionResources{{buffer: first, access: submissionRead}}
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			device := &Device{}
			firstBuffer := &Buffer{}
			secondBuffer := &Buffer{}
			first, err := device.acquireSubmission(test.first(firstBuffer, secondBuffer))
			if err != nil {
				t.Fatalf("acquire first submission: %v", err)
			}
			defer device.releaseSubmission(first)

			result := make(chan submissionReservationResult, 1)
			go func() {
				reservation, err := device.acquireSubmission(test.second(firstBuffer, secondBuffer))
				result <- submissionReservationResult{reservation: reservation, err: err}
			}()
			second := receiveReservationResult(t, result, "compatible submission")
			if second.err != nil {
				t.Fatalf("acquire compatible submission: %v", second.err)
			}
			device.releaseSubmission(second.reservation)
		})
	}
}

func TestSubmissionSchedulerOrdersWriteHazards(t *testing.T) {
	tests := []struct {
		name   string
		first  submissionAccess
		second submissionAccess
	}{
		{name: "read then write", first: submissionRead, second: submissionWrite},
		{name: "write then read", first: submissionWrite, second: submissionRead},
		{name: "write then write", first: submissionWrite, second: submissionWrite},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			device := &Device{}
			buffer := &Buffer{}
			first, err := device.acquireSubmission(submissionResources{{buffer: buffer, access: test.first}})
			if err != nil {
				t.Fatalf("acquire first submission: %v", err)
			}
			defer device.releaseSubmission(first)

			result := make(chan submissionReservationResult, 1)
			go func() {
				reservation, err := device.acquireSubmission(submissionResources{{buffer: buffer, access: test.second}})
				result <- submissionReservationResult{reservation: reservation, err: err}
			}()
			waitForPendingSubmissions(t, device, 1)
			select {
			case acquired := <-result:
				device.releaseSubmission(acquired.reservation)
				t.Fatal("hazardous submission acquired before earlier completion")
			default:
			}

			device.releaseSubmission(first)
			second := receiveReservationResult(t, result, "ordered hazardous submission")
			if second.err != nil {
				t.Fatalf("acquire hazardous submission after release: %v", second.err)
			}
			device.releaseSubmission(second.reservation)
		})
	}
}

func TestSubmissionSchedulerDoesNotBypassWaitingWriter(t *testing.T) {
	device := &Device{}
	buffer := &Buffer{}
	activeRead, err := device.acquireSubmission(submissionResources{{buffer: buffer, access: submissionRead}})
	if err != nil {
		t.Fatalf("acquire active read: %v", err)
	}
	defer device.releaseSubmission(activeRead)

	writerResult := make(chan submissionReservationResult, 1)
	go func() {
		reservation, err := device.acquireSubmission(submissionResources{{buffer: buffer, access: submissionWrite}})
		writerResult <- submissionReservationResult{reservation: reservation, err: err}
	}()
	waitForPendingSubmissions(t, device, 1)

	readerResult := make(chan submissionReservationResult, 1)
	go func() {
		reservation, err := device.acquireSubmission(submissionResources{{buffer: buffer, access: submissionRead}})
		readerResult <- submissionReservationResult{reservation: reservation, err: err}
	}()
	waitForPendingSubmissions(t, device, 2)

	device.releaseSubmission(activeRead)
	writer := receiveReservationResult(t, writerResult, "waiting writer")
	if writer.err != nil {
		t.Fatalf("acquire waiting writer: %v", writer.err)
	}
	select {
	case reader := <-readerResult:
		device.releaseSubmission(reader.reservation)
		t.Fatal("later reader bypassed waiting writer")
	default:
	}
	device.releaseSubmission(writer.reservation)
	reader := receiveReservationResult(t, readerResult, "reader after writer")
	if reader.err != nil {
		t.Fatalf("acquire reader after writer: %v", reader.err)
	}
	device.releaseSubmission(reader.reservation)
}

func TestRecorderCompatibleSubmissionsWaitConcurrently(t *testing.T) {
	tests := []struct {
		name      string
		resources func(*Buffer, *Buffer) (submissionResources, submissionResources)
	}{
		{
			name: "disjoint writes",
			resources: func(first, second *Buffer) (submissionResources, submissionResources) {
				return submissionResources{{buffer: first, access: submissionWrite}},
					submissionResources{{buffer: second, access: submissionWrite}}
			},
		},
		{
			name: "shared reads",
			resources: func(first, _ *Buffer) (submissionResources, submissionResources) {
				return submissionResources{{buffer: first, access: submissionRead}},
					submissionResources{{buffer: first, access: submissionRead}}
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			device, _, _ := fakeBufferDevice("")
			firstBuffer := &Buffer{device: device}
			secondBuffer := &Buffer{device: device}
			firstResources, secondResources := test.resources(firstBuffer, secondBuffer)
			runCompatibleRecorderSubmissions(t, device, firstResources, secondResources)
		})
	}
}

func TestRecorderConcurrentSharedReadDirectVulkan(t *testing.T) {
	device, err := Open()
	if err != nil {
		t.Skipf("direct Vulkan device unavailable: %v", err)
	}
	t.Cleanup(func() {
		if err := device.Close(); err != nil {
			t.Errorf("Device.Close: %v", err)
		}
	})
	input, err := device.NewBuffer(64 * 4)
	if err != nil {
		t.Fatalf("new input buffer: %v", err)
	}
	t.Cleanup(func() { _ = input.Close() })
	firstOutput, err := device.NewBuffer(64 * 4)
	if err != nil {
		t.Fatalf("new first output buffer: %v", err)
	}
	t.Cleanup(func() { _ = firstOutput.Close() })
	secondOutput, err := device.NewBuffer(64 * 4)
	if err != nil {
		t.Fatalf("new second output buffer: %v", err)
	}
	t.Cleanup(func() { _ = secondOutput.Close() })
	kernel, err := device.NewKernel(KernelOptions{
		WGSL: doubleKernelWGSL,
		Bindings: []BindingLayout{
			{Binding: 0, Access: BufferReadOnly},
			{Binding: 1, Access: BufferReadWrite},
		},
	})
	if err != nil {
		t.Fatalf("NewKernel: %v", err)
	}
	t.Cleanup(func() { _ = kernel.Close() })
	firstBindings, err := kernel.NewBindings(
		BindBuffer(0, input),
		BindBuffer(1, firstOutput),
	)
	if err != nil {
		t.Fatalf("new first bindings: %v", err)
	}
	t.Cleanup(func() { _ = firstBindings.Close() })
	secondBindings, err := kernel.NewBindings(
		BindBuffer(0, input),
		BindBuffer(1, secondOutput),
	)
	if err != nil {
		t.Fatalf("new second bindings: %v", err)
	}
	t.Cleanup(func() { _ = secondBindings.Close() })
	values := make([]float32, 64)
	for index := range values {
		values[index] = float32(index + 1)
	}
	if err := input.Upload(encodeFloat32s(values)); err != nil {
		t.Fatalf("upload input: %v", err)
	}
	firstRecorder, err := device.NewRecorder()
	if err != nil {
		t.Fatalf("new first recorder: %v", err)
	}
	t.Cleanup(func() { _ = firstRecorder.Abort() })
	if err := firstRecorder.Dispatch(kernel, firstBindings, Workgroups{X: 1, Y: 1, Z: 1}); err != nil {
		t.Fatalf("record first dispatch: %v", err)
	}
	secondRecorder, err := device.NewRecorder()
	if err != nil {
		t.Fatalf("new second recorder: %v", err)
	}
	t.Cleanup(func() { _ = secondRecorder.Abort() })
	if err := secondRecorder.Dispatch(kernel, secondBindings, Workgroups{X: 1, Y: 1, Z: 1}); err != nil {
		t.Fatalf("record second dispatch: %v", err)
	}

	results := make(chan error, 2)
	go func() { results <- firstRecorder.SubmitAndWait() }()
	go func() { results <- secondRecorder.SubmitAndWait() }()
	for index := 0; index < 2; index++ {
		if err := receiveError(t, results, "direct Vulkan recorder result"); err != nil {
			t.Fatalf("SubmitAndWait: %v", err)
		}
	}
	for name, output := range map[string]*Buffer{
		"first":  firstOutput,
		"second": secondOutput,
	} {
		encoded := make([]byte, 64*4)
		if err := output.Download(encoded); err != nil {
			t.Fatalf("download %s output: %v", name, err)
		}
		for index, value := range decodeFloat32s(encoded) {
			if want := values[index] * 2; value != want {
				t.Fatalf("%s output[%d] = %v, want %v", name, index, value, want)
			}
		}
	}
}

func TestRecorderWriteHazardWaitsForCompletion(t *testing.T) {
	device, _, _ := fakeBufferDevice("")
	buffer := &Buffer{device: device}
	var nextFence atomic.Uint64
	submitted := make(chan vk.Fence, 2)
	waiting := make(chan vk.Fence, 2)
	firstWait := make(chan struct{})
	secondWait := make(chan struct{})
	releaseFirst := closeChannelOnce(firstWait)
	releaseSecond := closeChannelOnce(secondWait)
	t.Cleanup(releaseFirst)
	t.Cleanup(releaseSecond)
	configureRecorderSubmissionOps(device, &nextFence, submitted, waiting, func(fence vk.Fence) {
		if fence == 1 {
			<-firstWait
			return
		}
		<-secondWait
	})

	first := &Recorder{
		device:    device,
		state:     recorderRecording,
		resources: submissionResources{{buffer: buffer, access: submissionRead}},
	}
	second := &Recorder{
		device:    device,
		state:     recorderRecording,
		resources: submissionResources{{buffer: buffer, access: submissionWrite}},
	}
	results := make(chan error, 2)
	go func() { results <- first.SubmitAndWait() }()
	if fence := receiveFence(t, submitted, "first submission"); fence != 1 {
		t.Fatalf("first submitted fence = %d, want 1", fence)
	}
	if fence := receiveFence(t, waiting, "first fence wait"); fence != 1 {
		t.Fatalf("first waited fence = %d, want 1", fence)
	}

	go func() { results <- second.SubmitAndWait() }()
	waitForPendingSubmissions(t, device, 1)
	select {
	case fence := <-submitted:
		t.Fatalf("hazardous fence %d submitted before first completion", fence)
	default:
	}

	releaseFirst()
	if err := receiveError(t, results, "first submission result"); err != nil {
		t.Fatalf("first SubmitAndWait: %v", err)
	}
	if fence := receiveFence(t, submitted, "second submission"); fence != 2 {
		t.Fatalf("second submitted fence = %d, want 2", fence)
	}
	if fence := receiveFence(t, waiting, "second fence wait"); fence != 2 {
		t.Fatalf("second waited fence = %d, want 2", fence)
	}
	releaseSecond()
	if err := receiveError(t, results, "second submission result"); err != nil {
		t.Fatalf("second SubmitAndWait: %v", err)
	}
}

func TestDeviceCloseWaitsForConcurrentSubmissionWaits(t *testing.T) {
	device, _, _ := fakeBufferDevice("")
	firstBuffer := &Buffer{device: device}
	secondBuffer := &Buffer{device: device}
	var nextFence atomic.Uint64
	submitted := make(chan vk.Fence, 2)
	waiting := make(chan vk.Fence, 2)
	firstWait := make(chan struct{})
	secondWait := make(chan struct{})
	releaseFirst := closeChannelOnce(firstWait)
	releaseSecond := closeChannelOnce(secondWait)
	t.Cleanup(releaseFirst)
	t.Cleanup(releaseSecond)
	configureRecorderSubmissionOps(device, &nextFence, submitted, waiting, func(fence vk.Fence) {
		if fence == 1 {
			<-firstWait
			return
		}
		<-secondWait
	})
	waitIdle := make(chan struct{}, 1)
	device.state.deviceFns = &vk.DeviceFuncs{}
	device.state.hooks.deviceWaitIdle = func(*vk.DeviceFuncs, vk.Device) error {
		waitIdle <- struct{}{}
		return nil
	}
	device.state.hooks.destroyDevice = func(*vk.DeviceFuncs, vk.Device) {}

	results := make(chan error, 2)
	first := &Recorder{
		device: device, state: recorderRecording,
		resources: submissionResources{{buffer: firstBuffer, access: submissionWrite}},
	}
	second := &Recorder{
		device: device, state: recorderRecording,
		resources: submissionResources{{buffer: secondBuffer, access: submissionWrite}},
	}
	go func() { results <- first.SubmitAndWait() }()
	go func() { results <- second.SubmitAndWait() }()
	receiveFence(t, submitted, "first close-test submission")
	receiveFence(t, submitted, "second close-test submission")
	receiveFence(t, waiting, "first close-test wait")
	receiveFence(t, waiting, "second close-test wait")

	closed := make(chan error, 1)
	go func() { closed <- device.Close() }()
	waitForDeviceClosing(t, device)
	select {
	case <-waitIdle:
		t.Fatal("Device.Close waited for idle before active submissions completed")
	default:
	}
	releaseFirst()
	if err := receiveError(t, results, "first close-test result"); err != nil {
		t.Fatalf("first SubmitAndWait: %v", err)
	}
	select {
	case <-waitIdle:
		t.Fatal("Device.Close waited for idle while one submission was active")
	default:
	}
	releaseSecond()
	if err := receiveError(t, results, "second close-test result"); err != nil {
		t.Fatalf("second SubmitAndWait: %v", err)
	}
	select {
	case <-waitIdle:
	case <-time.After(submissionTestTimeout):
		t.Fatal("Device.Close did not wait for device idle")
	}
	if err := receiveError(t, closed, "Device.Close result"); err != nil {
		t.Fatalf("Device.Close: %v", err)
	}
}

func TestRecorderWaitFailureRejectsLaterSubmissions(t *testing.T) {
	waitErr := errors.New("injected uncertain completion")
	device, _, _ := fakeBufferDevice("")
	buffer := &Buffer{device: device}
	var nextFence atomic.Uint64
	queueSubmits := atomic.Int32{}
	device.state.ops.endCommandBuffer = func(*vk.DeviceFuncs, vk.CommandBuffer) error { return nil }
	device.state.ops.createFence = func(*vk.DeviceFuncs, vk.Device, *vk.FenceCreateInfo) (vk.Fence, error) {
		return vk.Fence(nextFence.Add(1)), nil
	}
	device.state.ops.destroyFence = func(*vk.DeviceFuncs, vk.Device, vk.Fence) {}
	device.state.ops.queueSubmit = func(*vk.DeviceFuncs, vk.Queue, []vk.SubmitInfo, vk.Fence) error {
		queueSubmits.Add(1)
		return nil
	}
	device.state.ops.waitForFences = func(*vk.DeviceFuncs, vk.Device, []vk.Fence, bool, uint64) error {
		return waitErr
	}
	device.state.deviceFns = &vk.DeviceFuncs{}
	device.state.hooks.deviceWaitIdle = func(*vk.DeviceFuncs, vk.Device) error { return nil }
	device.state.hooks.destroyDevice = func(*vk.DeviceFuncs, vk.Device) {}

	first := &Recorder{
		device: device, state: recorderRecording,
		resources: submissionResources{{buffer: buffer, access: submissionWrite}},
	}
	var err error
	first.childID, err = device.addChild(first)
	if err != nil {
		t.Fatalf("add first recorder: %v", err)
	}
	if err := first.SubmitAndWait(); !errors.Is(err, waitErr) {
		t.Fatalf("first SubmitAndWait error = %v, want wait failure", err)
	}

	second := &Recorder{
		device: device, state: recorderRecording,
		resources: submissionResources{{buffer: &Buffer{device: device}, access: submissionRead}},
	}
	second.childID, err = device.addChild(second)
	if err != nil {
		t.Fatalf("add second recorder: %v", err)
	}
	if err := second.SubmitAndWait(); !errors.Is(err, waitErr) {
		t.Fatalf("second SubmitAndWait error = %v, want earlier wait failure", err)
	}
	if got := queueSubmits.Load(); got != 1 {
		t.Fatalf("queue submissions after uncertain completion = %d, want 1", got)
	}
	if err := device.Close(); err != nil {
		t.Fatalf("Device.Close: %v", err)
	}
	device.submissionMu.Lock()
	active := len(device.activeSubmissions)
	device.submissionMu.Unlock()
	if active != 0 {
		t.Fatalf("active reservations after Device.Close = %d, want 0", active)
	}
}

func TestBufferWaitFailureRetainsTransientResourcesUntilDeviceClose(t *testing.T) {
	waitErr := errors.New("injected buffer wait failure")
	device, _, _ := fakeBufferDevice("")
	buffer, err := device.NewBuffer(4)
	if err != nil {
		t.Fatalf("NewBuffer: %v", err)
	}
	mapped := make([]byte, 4)
	device.state.ops.mapMemory = func(*vk.DeviceFuncs, vk.Device, vk.DeviceMemory, uint64, uint64) (unsafe.Pointer, error) {
		return unsafe.Pointer(&mapped[0]), nil
	}
	device.state.ops.unmapMemory = func(*vk.DeviceFuncs, vk.Device, vk.DeviceMemory) {}
	device.state.ops.bufferBarriers = func(*vk.DeviceFuncs, vk.CommandBuffer, vk.PipelineStageFlags, vk.PipelineStageFlags, []vk.BufferMemoryBarrier) {
	}
	device.state.ops.copyBuffer = func(*vk.DeviceFuncs, vk.CommandBuffer, vk.Buffer, vk.Buffer, []vk.BufferCopy) {}
	device.state.ops.createCommandPool = func(*vk.DeviceFuncs, vk.Device, *vk.CommandPoolCreateInfo) (vk.CommandPool, error) {
		return vk.CommandPool(20), nil
	}
	device.state.ops.allocateCommandBuffers = func(*vk.DeviceFuncs, vk.Device, *vk.CommandBufferAllocateInfo) ([]vk.CommandBuffer, error) {
		return []vk.CommandBuffer{21}, nil
	}
	device.state.ops.beginCommandBuffer = func(*vk.DeviceFuncs, vk.CommandBuffer, *vk.CommandBufferBeginInfo) error { return nil }
	device.state.ops.endCommandBuffer = func(*vk.DeviceFuncs, vk.CommandBuffer) error { return nil }
	device.state.ops.createFence = func(*vk.DeviceFuncs, vk.Device, *vk.FenceCreateInfo) (vk.Fence, error) {
		return vk.Fence(22), nil
	}
	destroyedPools := 0
	destroyedFences := 0
	device.state.ops.destroyCommandPool = func(*vk.DeviceFuncs, vk.Device, vk.CommandPool) { destroyedPools++ }
	device.state.ops.destroyFence = func(*vk.DeviceFuncs, vk.Device, vk.Fence) { destroyedFences++ }
	device.state.ops.queueSubmit = func(*vk.DeviceFuncs, vk.Queue, []vk.SubmitInfo, vk.Fence) error { return nil }
	device.state.ops.waitForFences = func(*vk.DeviceFuncs, vk.Device, []vk.Fence, bool, uint64) error {
		return waitErr
	}
	waitIdleCalls := 0
	device.state.deviceFns = &vk.DeviceFuncs{}
	device.state.hooks.deviceWaitIdle = func(*vk.DeviceFuncs, vk.Device) error {
		waitIdleCalls++
		return nil
	}
	device.state.hooks.destroyDevice = func(*vk.DeviceFuncs, vk.Device) {}

	if err := buffer.Upload([]byte{1, 2, 3, 4}); !errors.Is(err, waitErr) {
		t.Fatalf("Upload error = %v, want wait failure", err)
	}
	if destroyedPools != 0 || destroyedFences != 0 {
		t.Fatalf("transient resources destroyed early: pools=%d fences=%d", destroyedPools, destroyedFences)
	}
	if err := buffer.Close(); err == nil {
		t.Fatal("Buffer.Close succeeded while completion was unknown")
	}
	if err := device.Close(); err != nil {
		t.Fatalf("Device.Close: %v", err)
	}
	if waitIdleCalls != 1 || destroyedPools != 1 || destroyedFences != 1 {
		t.Fatalf("cleanup counts: waits=%d pools=%d fences=%d, want 1 each",
			waitIdleCalls, destroyedPools, destroyedFences)
	}
}

func runCompatibleRecorderSubmissions(
	t *testing.T,
	device *Device,
	firstResources, secondResources submissionResources,
) {
	t.Helper()
	var nextFence atomic.Uint64
	created := make(chan vk.Fence, 2)
	submitted := make(chan vk.Fence, 2)
	waiting := make(chan vk.Fence, 2)
	firstSubmit := make(chan struct{})
	firstWait := make(chan struct{})
	secondWait := make(chan struct{})
	releaseSubmit := closeChannelOnce(firstSubmit)
	releaseFirstWait := closeChannelOnce(firstWait)
	releaseSecondWait := closeChannelOnce(secondWait)
	t.Cleanup(releaseSubmit)
	t.Cleanup(releaseFirstWait)
	t.Cleanup(releaseSecondWait)
	var submitsInCall atomic.Int32
	var overlappingSubmitCalls atomic.Bool
	device.state.ops.endCommandBuffer = func(*vk.DeviceFuncs, vk.CommandBuffer) error { return nil }
	device.state.ops.createFence = func(*vk.DeviceFuncs, vk.Device, *vk.FenceCreateInfo) (vk.Fence, error) {
		fence := vk.Fence(nextFence.Add(1))
		created <- fence
		return fence, nil
	}
	device.state.ops.destroyFence = func(*vk.DeviceFuncs, vk.Device, vk.Fence) {}
	device.state.ops.queueSubmit = func(_ *vk.DeviceFuncs, _ vk.Queue, _ []vk.SubmitInfo, fence vk.Fence) error {
		if submitsInCall.Add(1) != 1 {
			overlappingSubmitCalls.Store(true)
		}
		submitted <- fence
		if fence == 1 {
			<-firstSubmit
		}
		submitsInCall.Add(-1)
		return nil
	}
	device.state.ops.waitForFences = func(_ *vk.DeviceFuncs, _ vk.Device, fences []vk.Fence, _ bool, _ uint64) error {
		fence := fences[0]
		waiting <- fence
		if fence == 1 {
			<-firstWait
		} else {
			<-secondWait
		}
		return nil
	}

	first := &Recorder{device: device, state: recorderRecording, resources: firstResources}
	second := &Recorder{device: device, state: recorderRecording, resources: secondResources}
	results := make(chan error, 2)
	go func() { results <- first.SubmitAndWait() }()
	if fence := receiveFence(t, created, "first fence creation"); fence != 1 {
		t.Fatalf("first created fence = %d, want 1", fence)
	}
	if fence := receiveFence(t, submitted, "first queue submission"); fence != 1 {
		t.Fatalf("first submitted fence = %d, want 1", fence)
	}

	go func() { results <- second.SubmitAndWait() }()
	if fence := receiveFence(t, created, "second fence creation"); fence != 2 {
		t.Fatalf("second created fence = %d, want 2", fence)
	}
	waitForActiveSubmissions(t, device, 2)
	releaseSubmit()
	if fence := receiveFence(t, submitted, "second queue submission"); fence != 2 {
		t.Fatalf("second submitted fence = %d, want 2", fence)
	}
	waited := map[vk.Fence]bool{
		receiveFence(t, waiting, "first concurrent wait"):  true,
		receiveFence(t, waiting, "second concurrent wait"): true,
	}
	if !waited[1] || !waited[2] {
		t.Fatalf("waited fences = %v, want 1 and 2", waited)
	}
	if overlappingSubmitCalls.Load() {
		t.Fatal("vkQueueSubmit calls overlapped")
	}
	releaseFirstWait()
	releaseSecondWait()
	for index := 0; index < 2; index++ {
		if err := receiveError(t, results, "compatible recorder result"); err != nil {
			t.Fatalf("SubmitAndWait: %v", err)
		}
	}
}

func configureRecorderSubmissionOps(
	device *Device,
	nextFence *atomic.Uint64,
	submitted, waiting chan<- vk.Fence,
	wait func(vk.Fence),
) {
	device.state.ops.endCommandBuffer = func(*vk.DeviceFuncs, vk.CommandBuffer) error { return nil }
	device.state.ops.createFence = func(*vk.DeviceFuncs, vk.Device, *vk.FenceCreateInfo) (vk.Fence, error) {
		return vk.Fence(nextFence.Add(1)), nil
	}
	device.state.ops.destroyFence = func(*vk.DeviceFuncs, vk.Device, vk.Fence) {}
	device.state.ops.queueSubmit = func(_ *vk.DeviceFuncs, _ vk.Queue, _ []vk.SubmitInfo, fence vk.Fence) error {
		submitted <- fence
		return nil
	}
	device.state.ops.waitForFences = func(_ *vk.DeviceFuncs, _ vk.Device, fences []vk.Fence, _ bool, _ uint64) error {
		fence := fences[0]
		waiting <- fence
		wait(fence)
		return nil
	}
}

func waitForPendingSubmissions(t *testing.T, device *Device, want int) {
	t.Helper()
	deadline := time.Now().Add(submissionTestTimeout)
	for {
		device.submissionMu.Lock()
		got := len(device.pendingSubmissions)
		device.submissionMu.Unlock()
		if got == want {
			return
		}
		if time.Now().After(deadline) {
			t.Fatalf("pending submissions = %d, want %d", got, want)
		}
		runtime.Gosched()
	}
}

func waitForActiveSubmissions(t *testing.T, device *Device, want int) {
	t.Helper()
	deadline := time.Now().Add(submissionTestTimeout)
	for {
		device.submissionMu.Lock()
		got := len(device.activeSubmissions)
		device.submissionMu.Unlock()
		if got == want {
			return
		}
		if time.Now().After(deadline) {
			t.Fatalf("active submissions = %d, want %d", got, want)
		}
		runtime.Gosched()
	}
}

func waitForDeviceClosing(t *testing.T, device *Device) {
	t.Helper()
	deadline := time.Now().Add(submissionTestTimeout)
	for {
		device.mu.Lock()
		closing := device.closing
		device.mu.Unlock()
		if closing {
			return
		}
		if time.Now().After(deadline) {
			t.Fatal("Device.Close did not enter closing state")
		}
		runtime.Gosched()
	}
}

func receiveReservationResult(
	t *testing.T,
	results <-chan submissionReservationResult,
	name string,
) submissionReservationResult {
	t.Helper()
	select {
	case result := <-results:
		return result
	case <-time.After(submissionTestTimeout):
		t.Fatalf("timed out waiting for %s", name)
		return submissionReservationResult{}
	}
}

func receiveFence(t *testing.T, fences <-chan vk.Fence, name string) vk.Fence {
	t.Helper()
	select {
	case fence := <-fences:
		return fence
	case <-time.After(submissionTestTimeout):
		t.Fatalf("timed out waiting for %s", name)
		return 0
	}
}

func receiveError(t *testing.T, results <-chan error, name string) error {
	t.Helper()
	select {
	case err := <-results:
		return err
	case <-time.After(submissionTestTimeout):
		t.Fatalf("timed out waiting for %s", name)
		return nil
	}
}

func closeChannelOnce(channel chan struct{}) func() {
	var once sync.Once
	return func() {
		once.Do(func() { close(channel) })
	}
}
