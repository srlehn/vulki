package vulki

import (
	"bytes"
	"testing"
)

const benchmarkTransferSize = 4 * 1024 * 1024

func BenchmarkTransfer(b *testing.B) {
	device, err := Open()
	if err != nil {
		b.Skipf("direct Vulkan device unavailable: %v", err)
	}
	b.Cleanup(func() {
		if err := device.Close(); err != nil {
			b.Errorf("close device: %v", err)
		}
	})

	b.Run("Buffer/Upload", func(b *testing.B) {
		buffer := newBenchmarkTransferBuffer(b, device)
		source := benchmarkTransferPattern()
		if err := buffer.Upload(source); err != nil {
			b.Fatalf("warm upload: %v", err)
		}
		verifyBenchmarkTransferBuffer(b, buffer, source)

		b.SetBytes(benchmarkTransferSize)
		b.ReportAllocs()
		b.ResetTimer()
		for range b.N {
			if err := buffer.Upload(source); err != nil {
				b.Fatalf("upload: %v", err)
			}
		}
	})

	b.Run("Buffer/Download", func(b *testing.B) {
		buffer := newBenchmarkTransferBuffer(b, device)
		source := benchmarkTransferPattern()
		if err := buffer.Upload(source); err != nil {
			b.Fatalf("initialize buffer: %v", err)
		}
		destination := make([]byte, benchmarkTransferSize)
		if err := buffer.Download(destination); err != nil {
			b.Fatalf("warm download: %v", err)
		}
		if !bytes.Equal(destination, source) {
			b.Fatal("warm download returned incorrect data")
		}

		b.SetBytes(benchmarkTransferSize)
		b.ReportAllocs()
		b.ResetTimer()
		for range b.N {
			if err := buffer.Download(destination); err != nil {
				b.Fatalf("download: %v", err)
			}
		}
	})

	b.Run("Recorder/Upload", func(b *testing.B) {
		buffer := newBenchmarkTransferBuffer(b, device)
		source := benchmarkTransferPattern()
		runBenchmarkRecorderUpload(b, device, buffer, source)
		verifyBenchmarkTransferBuffer(b, buffer, source)

		b.SetBytes(benchmarkTransferSize)
		b.ReportAllocs()
		b.ResetTimer()
		for range b.N {
			runBenchmarkRecorderUpload(b, device, buffer, source)
		}
	})

	b.Run("Recorder/Download", func(b *testing.B) {
		buffer := newBenchmarkTransferBuffer(b, device)
		source := benchmarkTransferPattern()
		if err := buffer.Upload(source); err != nil {
			b.Fatalf("initialize buffer: %v", err)
		}
		destination := make([]byte, benchmarkTransferSize)
		runBenchmarkRecorderDownload(b, device, buffer, destination)
		if !bytes.Equal(destination, source) {
			b.Fatal("warm recorded download returned incorrect data")
		}

		b.SetBytes(benchmarkTransferSize)
		b.ReportAllocs()
		b.ResetTimer()
		for range b.N {
			runBenchmarkRecorderDownload(b, device, buffer, destination)
		}
	})
}

func newBenchmarkTransferBuffer(b *testing.B, device *Device) *Buffer {
	b.Helper()
	buffer, err := device.NewBuffer(benchmarkTransferSize)
	if err != nil {
		b.Fatalf("create transfer buffer: %v", err)
	}
	b.Cleanup(func() {
		if err := buffer.Close(); err != nil {
			b.Errorf("close transfer buffer: %v", err)
		}
	})
	return buffer
}

func benchmarkTransferPattern() []byte {
	data := make([]byte, benchmarkTransferSize)
	for index := range data {
		data[index] = byte(index*29 + index/251 + 17)
	}
	return data
}

func verifyBenchmarkTransferBuffer(b *testing.B, buffer *Buffer, want []byte) {
	b.Helper()
	got := make([]byte, len(want))
	if err := buffer.Download(got); err != nil {
		b.Fatalf("verify transfer: %v", err)
	}
	if !bytes.Equal(got, want) {
		b.Fatal("transfer verification returned incorrect data")
	}
}

func runBenchmarkRecorderUpload(b *testing.B, device *Device, buffer *Buffer, source []byte) {
	b.Helper()
	recorder, err := device.NewRecorder()
	if err != nil {
		b.Fatalf("create upload recorder: %v", err)
	}
	if err := recorder.Upload(buffer, 0, source); err != nil {
		_ = recorder.Abort()
		b.Fatalf("record upload: %v", err)
	}
	if err := recorder.SubmitAndWait(); err != nil {
		b.Fatalf("submit upload: %v", err)
	}
}

func runBenchmarkRecorderDownload(b *testing.B, device *Device, buffer *Buffer, destination []byte) {
	b.Helper()
	recorder, err := device.NewRecorder()
	if err != nil {
		b.Fatalf("create download recorder: %v", err)
	}
	if err := recorder.Download(buffer, 0, destination); err != nil {
		_ = recorder.Abort()
		b.Fatalf("record download: %v", err)
	}
	if err := recorder.SubmitAndWait(); err != nil {
		b.Fatalf("submit download: %v", err)
	}
}
