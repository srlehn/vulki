package vulki

import (
	"errors"
	"reflect"
	"testing"
	"time"

	"github.com/srlehn/vulki/vk"
)

type timestampWrite struct {
	stage vk.PipelineStageFlags
	query uint32
}

type timestampFakes struct {
	createdPools   int
	destroyedPools int
	resetFirst     uint32
	resetCount     uint32
	resets         int
	writes         []timestampWrite
	rawValues      []uint64
	queryErr       error
	reads          int
}

func configureTimestampFakes(device *Device, validBits uint32, period float32, fakes *timestampFakes) {
	device.state.timestampValidBits = validBits
	device.state.timestampPeriod = period
	device.state.ops.createQueryPool = func(*vk.DeviceFuncs, vk.Device, *vk.QueryPoolCreateInfo) (vk.QueryPool, error) {
		fakes.createdPools++
		return vk.QueryPool(91), nil
	}
	device.state.ops.destroyQueryPool = func(*vk.DeviceFuncs, vk.Device, vk.QueryPool) {
		fakes.destroyedPools++
	}
	device.state.ops.resetQueryPool = func(_ *vk.DeviceFuncs, _ vk.CommandBuffer, _ vk.QueryPool, first, count uint32) {
		fakes.resets++
		fakes.resetFirst, fakes.resetCount = first, count
	}
	device.state.ops.writeTimestamp = func(_ *vk.DeviceFuncs, _ vk.CommandBuffer, stage vk.PipelineStageFlags, _ vk.QueryPool, query uint32) {
		fakes.writes = append(fakes.writes, timestampWrite{stage: stage, query: query})
	}
	device.state.ops.queryPoolResults = func(_ *vk.DeviceFuncs, _ vk.Device, _ vk.QueryPool, first uint32, results []uint64, _ vk.QueryResultFlags) error {
		fakes.reads++
		if fakes.queryErr != nil {
			return fakes.queryErr
		}
		copy(results, fakes.rawValues[first:])
		return nil
	}
}

func configureNoopSubmitOps(device *Device) {
	device.state.ops.endCommandBuffer = func(*vk.DeviceFuncs, vk.CommandBuffer) error { return nil }
	device.state.ops.createFence = func(*vk.DeviceFuncs, vk.Device, *vk.FenceCreateInfo) (vk.Fence, error) {
		return vk.Fence(81), nil
	}
	device.state.ops.destroyFence = func(*vk.DeviceFuncs, vk.Device, vk.Fence) {}
	device.state.ops.queueSubmit = func(*vk.DeviceFuncs, vk.Queue, []vk.SubmitInfo, vk.Fence) error { return nil }
	device.state.ops.waitForFences = func(*vk.DeviceFuncs, vk.Device, []vk.Fence, bool, uint64) error { return nil }
}

func TestRecorderTimestampsMeasureSpansAndCleanUp(t *testing.T) {
	device, _, _ := fakeBufferDevice("")
	fakes := &timestampFakes{rawValues: []uint64{100, 150, 200, 260}}
	configureTimestampFakes(device, 64, 2, fakes)
	configureNoopSubmitOps(device)

	recorder := &Recorder{device: device, state: recorderRecording}
	if err := recorder.TimestampBegin("first"); err != nil {
		t.Fatalf("first TimestampBegin: %v", err)
	}
	if err := recorder.TimestampEnd("first"); err != nil {
		t.Fatalf("first TimestampEnd: %v", err)
	}
	if err := recorder.TimestampBegin("second"); err != nil {
		t.Fatalf("second TimestampBegin: %v", err)
	}
	if err := recorder.TimestampEnd("second"); err != nil {
		t.Fatalf("second TimestampEnd: %v", err)
	}
	if _, err := recorder.Timestamps(); err == nil {
		t.Fatal("Timestamps succeeded before completion was observed")
	}
	if err := recorder.SubmitAndWait(); err != nil {
		t.Fatalf("SubmitAndWait: %v", err)
	}
	spans, err := recorder.Timestamps()
	if err != nil {
		t.Fatalf("Timestamps: %v", err)
	}
	want := []TimestampSpan{
		{Label: "first", Duration: 100 * time.Nanosecond},
		{Label: "second", Duration: 120 * time.Nanosecond},
	}
	if !reflect.DeepEqual(spans, want) {
		t.Fatalf("spans = %v, want %v", spans, want)
	}
	wantWrites := []timestampWrite{
		{stage: vk.PipelineStageTopOfPipeBit, query: 0},
		{stage: vk.PipelineStageBottomOfPipeBit, query: 1},
		{stage: vk.PipelineStageTopOfPipeBit, query: 2},
		{stage: vk.PipelineStageBottomOfPipeBit, query: 3},
	}
	if !reflect.DeepEqual(fakes.writes, wantWrites) {
		t.Fatalf("timestamp writes = %v, want %v", fakes.writes, wantWrites)
	}
	if fakes.createdPools != 1 || fakes.destroyedPools != 1 {
		t.Fatalf("query pool lifecycle = %d created, %d destroyed, want 1 each",
			fakes.createdPools, fakes.destroyedPools)
	}
	if fakes.resets != 1 || fakes.resetFirst != 0 || fakes.resetCount != timestampQueryCount {
		t.Fatalf("query pool resets = %d at %d count %d, want one full reset",
			fakes.resets, fakes.resetFirst, fakes.resetCount)
	}
	if fakes.reads != 1 {
		t.Fatalf("query reads = %d, want 1", fakes.reads)
	}
}

func TestRecorderTimestampsUnsupportedDegradeAndEmpty(t *testing.T) {
	device, _, _ := fakeBufferDevice("")
	configureNoopSubmitOps(device)

	recorder := &Recorder{device: device, state: recorderRecording}
	if err := recorder.TimestampBegin("span"); err != nil {
		t.Fatalf("TimestampBegin: %v", err)
	}
	if err := recorder.TimestampEnd("span"); err != nil {
		t.Fatalf("TimestampEnd: %v", err)
	}
	if err := recorder.SubmitAndWait(); err != nil {
		t.Fatalf("SubmitAndWait: %v", err)
	}
	if _, err := recorder.Timestamps(); !errors.Is(err, ErrTimestampsUnsupported) {
		t.Fatalf("Timestamps error = %v, want ErrTimestampsUnsupported", err)
	}

	empty := &Recorder{device: device, state: recorderRecording}
	if err := empty.SubmitAndWait(); err != nil {
		t.Fatalf("empty SubmitAndWait: %v", err)
	}
	spans, err := empty.Timestamps()
	if err != nil {
		t.Fatalf("empty Timestamps: %v", err)
	}
	if len(spans) != 0 {
		t.Fatalf("empty spans = %v, want none", spans)
	}
}

func TestRecorderTimestampPairingValidation(t *testing.T) {
	device, _, _ := fakeBufferDevice("")
	configureNoopSubmitOps(device)

	recorder := &Recorder{device: device, state: recorderRecording}
	if err := recorder.TimestampEnd("missing"); err == nil {
		t.Fatal("TimestampEnd succeeded without an open span")
	}
	if err := recorder.TimestampBegin("outer"); err != nil {
		t.Fatalf("TimestampBegin: %v", err)
	}
	if err := recorder.TimestampBegin("nested"); err == nil {
		t.Fatal("nested TimestampBegin succeeded")
	}
	if err := recorder.TimestampEnd("wrong"); err == nil {
		t.Fatal("mismatched TimestampEnd succeeded")
	}
	if _, err := device.Submit(recorder); err == nil {
		t.Fatal("Submit succeeded with an open timestamp span")
	}
	if recorder.state != recorderRecording {
		t.Fatalf("recorder state after rejected submit = %d, want recording", recorder.state)
	}
	if err := recorder.TimestampEnd("outer"); err != nil {
		t.Fatalf("TimestampEnd: %v", err)
	}
	for index := 1; index < maxTimestampSpans; index++ {
		if err := recorder.TimestampBegin("span"); err != nil {
			t.Fatalf("TimestampBegin %d: %v", index, err)
		}
		if err := recorder.TimestampEnd("span"); err != nil {
			t.Fatalf("TimestampEnd %d: %v", index, err)
		}
	}
	if err := recorder.TimestampBegin("overflow"); err == nil {
		t.Fatalf("TimestampBegin succeeded beyond %d spans", maxTimestampSpans)
	}
	if err := recorder.SubmitAndWait(); err != nil {
		t.Fatalf("SubmitAndWait: %v", err)
	}
}

func TestRecorderTimestampMaskWraparound(t *testing.T) {
	device, _, _ := fakeBufferDevice("")
	fakes := &timestampFakes{rawValues: []uint64{0xFFFF_FFF0, 0x10}}
	configureTimestampFakes(device, 32, 1, fakes)
	configureNoopSubmitOps(device)

	recorder := &Recorder{device: device, state: recorderRecording}
	if err := recorder.TimestampBegin("wrap"); err != nil {
		t.Fatalf("TimestampBegin: %v", err)
	}
	if err := recorder.TimestampEnd("wrap"); err != nil {
		t.Fatalf("TimestampEnd: %v", err)
	}
	if err := recorder.SubmitAndWait(); err != nil {
		t.Fatalf("SubmitAndWait: %v", err)
	}
	spans, err := recorder.Timestamps()
	if err != nil {
		t.Fatalf("Timestamps: %v", err)
	}
	if len(spans) != 1 || spans[0].Duration != 32*time.Nanosecond {
		t.Fatalf("spans = %v, want one 32ns wrapped span", spans)
	}
}

func TestRecorderTimestampQueryFailureIsLatched(t *testing.T) {
	queryErr := errors.New("injected query failure")
	device, _, _ := fakeBufferDevice("")
	fakes := &timestampFakes{queryErr: queryErr}
	configureTimestampFakes(device, 64, 1, fakes)
	configureNoopSubmitOps(device)

	recorder := &Recorder{device: device, state: recorderRecording}
	if err := recorder.TimestampBegin("span"); err != nil {
		t.Fatalf("TimestampBegin: %v", err)
	}
	if err := recorder.TimestampEnd("span"); err != nil {
		t.Fatalf("TimestampEnd: %v", err)
	}
	if err := recorder.SubmitAndWait(); err != nil {
		t.Fatalf("SubmitAndWait: %v", err)
	}
	if _, err := recorder.Timestamps(); !errors.Is(err, queryErr) {
		t.Fatalf("Timestamps error = %v, want query failure", err)
	}
	if fakes.destroyedPools != 1 {
		t.Fatalf("destroyed pools = %d, want 1", fakes.destroyedPools)
	}
}

func TestRecorderTimestampsDirectVulkan(t *testing.T) {
	device, err := Open()
	if err != nil {
		t.Skipf("direct Vulkan device unavailable: %v", err)
	}
	t.Cleanup(func() {
		if err := device.Close(); err != nil {
			t.Errorf("Device.Close: %v", err)
		}
	})
	buffer, err := device.NewBuffer(4096)
	if err != nil {
		t.Fatalf("NewBuffer: %v", err)
	}
	t.Cleanup(func() { _ = buffer.Close() })
	recorder, err := device.NewRecorder()
	if err != nil {
		t.Fatalf("NewRecorder: %v", err)
	}
	t.Cleanup(func() { _ = recorder.Abort() })

	input := make([]byte, 4096)
	for index := range input {
		input[index] = byte(index * 13)
	}
	output := make([]byte, 4096)
	if err := recorder.TimestampBegin("round-trip"); err != nil {
		t.Fatalf("TimestampBegin: %v", err)
	}
	if err := recorder.Upload(buffer, 0, input); err != nil {
		t.Fatalf("Upload: %v", err)
	}
	if err := recorder.Download(buffer, 0, output); err != nil {
		t.Fatalf("Download: %v", err)
	}
	if err := recorder.TimestampEnd("round-trip"); err != nil {
		t.Fatalf("TimestampEnd: %v", err)
	}
	if err := recorder.SubmitAndWait(); err != nil {
		t.Fatalf("SubmitAndWait: %v", err)
	}
	if !reflect.DeepEqual(output, input) {
		t.Fatal("timestamped round trip differs")
	}
	spans, err := recorder.Timestamps()
	if errors.Is(err, ErrTimestampsUnsupported) {
		t.Skip("device timestamps unsupported; degrade verified")
	}
	if err != nil {
		t.Fatalf("Timestamps: %v", err)
	}
	if len(spans) != 1 || spans[0].Label != "round-trip" {
		t.Fatalf("spans = %v, want one round-trip span", spans)
	}
	if spans[0].Duration < 0 {
		t.Fatalf("span duration = %v, want non-negative", spans[0].Duration)
	}
	t.Logf("GPU round-trip span: %v", spans[0].Duration)
}
