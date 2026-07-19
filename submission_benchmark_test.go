package vulki

import "testing"

// BenchmarkSubmissionModes measures per-submission fixed cost for many small
// recorded batches: blocking per-batch submission, asynchronous pipelined
// submission, and one fused queue submission carrying every batch.
func BenchmarkSubmissionModes(b *testing.B) {
	device, err := Open()
	if err != nil {
		b.Skipf("direct Vulkan device unavailable: %v", err)
	}
	defer device.Close()

	const batchCount = 8
	const payloadSize = 256
	payload := make([]byte, payloadSize)
	for index := range payload {
		payload[index] = byte(index)
	}
	buffers := make([]*Buffer, batchCount)
	for index := range buffers {
		buffer, err := device.NewBuffer(payloadSize)
		if err != nil {
			b.Fatalf("NewBuffer: %v", err)
		}
		defer buffer.Close()
		buffers[index] = buffer
	}
	record := func(b *testing.B, buffer *Buffer) *Recorder {
		b.Helper()
		recorder, err := device.NewRecorder()
		if err != nil {
			b.Fatalf("NewRecorder: %v", err)
		}
		if err := recorder.Upload(buffer, 0, payload); err != nil {
			b.Fatalf("Upload: %v", err)
		}
		return recorder
	}

	b.Run("BlockingPerBatch", func(b *testing.B) {
		for range b.N {
			for _, buffer := range buffers {
				if err := record(b, buffer).SubmitAndWait(); err != nil {
					b.Fatalf("SubmitAndWait: %v", err)
				}
			}
		}
	})
	b.Run("AsyncPipelined", func(b *testing.B) {
		submissions := make([]*Submission, 0, batchCount)
		for range b.N {
			submissions = submissions[:0]
			for _, buffer := range buffers {
				submission, err := record(b, buffer).Submit()
				if err != nil {
					b.Fatalf("Submit: %v", err)
				}
				submissions = append(submissions, submission)
			}
			for _, submission := range submissions {
				if err := submission.Wait(); err != nil {
					b.Fatalf("Wait: %v", err)
				}
			}
		}
	})
	b.Run("FusedSingleSubmit", func(b *testing.B) {
		recorders := make([]*Recorder, batchCount)
		for range b.N {
			for index, buffer := range buffers {
				recorders[index] = record(b, buffer)
			}
			submission, err := device.Submit(recorders...)
			if err != nil {
				b.Fatalf("Submit: %v", err)
			}
			if err := submission.Wait(); err != nil {
				b.Fatalf("Wait: %v", err)
			}
		}
	})
}
