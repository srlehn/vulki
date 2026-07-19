package vulki

import (
	"fmt"
	"slices"
	"time"

	"github.com/srlehn/vulki/vk"
)

const (
	maxTimestampSpans   = 16
	timestampQueryCount = 2 * maxTimestampSpans
)

// TimestampSpan reports one labeled GPU timing span measured by a Recorder.
type TimestampSpan struct {
	// Label names the span as passed to TimestampBegin.
	Label string
	// Duration is the device-side time between the span's begin and end
	// timestamps.
	Duration time.Duration
}

// TimestampBegin records the start of a labeled GPU timing span. Spans are
// flat and sequential: the previous span must be ended first, and at most 16
// spans may be recorded per submission. On a device without usable compute
// timestamps the span records nothing and Timestamps later reports
// ErrTimestampsUnsupported; recording never fails for that reason.
func (r *Recorder) TimestampBegin(label string) error {
	if r == nil {
		return fmt.Errorf("vulki: recorder is closed")
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if err := r.requireRecording(); err != nil {
		return err
	}
	if r.tsOpen {
		return fmt.Errorf("vulki: timestamp span %q is still open", r.tsLabels[len(r.tsLabels)-1])
	}
	if len(r.tsLabels) >= maxTimestampSpans {
		return fmt.Errorf("vulki: at most %d timestamp spans per submission", maxTimestampSpans)
	}
	state, err := r.device.beginOperation()
	if err != nil {
		return err
	}
	defer r.device.endOperation()
	if state.timestampsSupported() {
		if r.tsPool == 0 {
			info := vk.QueryPoolCreateInfo{
				SType:      vk.StructureTypeQueryPoolCreateInfo,
				QueryType:  vk.QueryTypeTimestamp,
				QueryCount: timestampQueryCount,
			}
			pool, err := state.ops.createQueryPool(state.deviceFns, state.device, &info)
			if err != nil {
				return fmt.Errorf("vulki: create timestamp query pool: %w", err)
			}
			r.tsPool = pool
			state.ops.resetQueryPool(state.deviceFns, r.command, pool, 0, timestampQueryCount)
		}
		query := uint32(2 * len(r.tsLabels))
		state.ops.writeTimestamp(state.deviceFns, r.command, vk.PipelineStageTopOfPipeBit, r.tsPool, query)
	}
	r.tsLabels = append(r.tsLabels, label)
	r.tsOpen = true
	return nil
}

// TimestampEnd records the end of the currently open GPU timing span. The
// label must match the span's TimestampBegin label.
func (r *Recorder) TimestampEnd(label string) error {
	if r == nil {
		return fmt.Errorf("vulki: recorder is closed")
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if err := r.requireRecording(); err != nil {
		return err
	}
	if !r.tsOpen {
		return fmt.Errorf("vulki: no open timestamp span to end")
	}
	current := r.tsLabels[len(r.tsLabels)-1]
	if current != label {
		return fmt.Errorf("vulki: timestamp span end %q does not match open span %q", label, current)
	}
	state, err := r.device.beginOperation()
	if err != nil {
		return err
	}
	defer r.device.endOperation()
	if state.timestampsSupported() && r.tsPool != 0 {
		query := uint32(2*(len(r.tsLabels)-1) + 1)
		state.ops.writeTimestamp(state.deviceFns, r.command, vk.PipelineStageBottomOfPipeBit, r.tsPool, query)
	}
	r.tsOpen = false
	return nil
}

// Timestamps returns the labeled GPU timing spans measured by this recorder,
// in recording order. It is valid once submission completion has been
// observed through SubmitAndWait, Submission.Wait, or Submission.Poll. On a
// device without usable compute timestamps it returns an error matching
// ErrTimestampsUnsupported when spans were recorded.
func (r *Recorder) Timestamps() ([]TimestampSpan, error) {
	if r == nil {
		return nil, fmt.Errorf("vulki: recorder is closed")
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if !r.tsResolved {
		return nil, fmt.Errorf("vulki: timestamps require observed submission completion")
	}
	if r.tsErr != nil {
		return nil, r.tsErr
	}
	return slices.Clone(r.tsResults), nil
}

// resolveTimestamps reads the recorded timestamp queries after known queue
// completion and latches per-span durations or the retrieval error. It must
// run before the recorder's native resources are destroyed.
func (r *Recorder) resolveTimestamps(state *deviceState) {
	if r.tsResolved {
		return
	}
	r.tsResolved = true
	if len(r.tsLabels) == 0 {
		return
	}
	if !state.timestampsSupported() || r.tsPool == 0 {
		r.tsErr = ErrTimestampsUnsupported
		return
	}
	raw := make([]uint64, 2*len(r.tsLabels))
	if err := state.ops.queryPoolResults(state.deviceFns, state.device, r.tsPool, 0, raw, 0); err != nil {
		r.tsErr = fmt.Errorf("vulki: read timestamp queries: %w", err)
		return
	}
	mask := ^uint64(0)
	if bits := state.timestampValidBits; bits < 64 {
		mask = 1<<bits - 1
	}
	period := float64(state.timestampPeriod)
	r.tsResults = make([]TimestampSpan, len(r.tsLabels))
	for index, label := range r.tsLabels {
		delta := (raw[2*index+1] - raw[2*index]) & mask
		r.tsResults[index] = TimestampSpan{
			Label:    label,
			Duration: time.Duration(float64(delta) * period),
		}
	}
}
