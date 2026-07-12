package imgproc

import (
	"bytes"
	_ "embed"
	"os/exec"
	"testing"

	"github.com/srlehn/vulki/shader"
)

//go:embed shaders/grayscale.wgsl
var standaloneGrayscaleWGSL string

//go:embed shaders/hann.wgsl
var standaloneHannWGSL string

func TestShadersPassSPIRVValidation(t *testing.T) {
	validator, err := exec.LookPath("spirv-val")
	if err != nil {
		t.Skip("spirv-val is not installed")
	}

	shaders := map[string]string{
		"bilinear_warp_gray": bilinearWarpGrayWGSL,
		"crosspower":         crosspowerWGSL,
		"fft_bitrev":         fftBitrevWGSL,
		"fft_butterfly":      fftButterflyWGSL,
		"fftshift":           fftshiftWGSL,
		"grayscale_pad":      grayscalePadWGSL,
		"grayscale":          standaloneGrayscaleWGSL,
		"hann":               standaloneHannWGSL,
		"highpass":           highpassWGSL,
		"logpolar":           logpolarWGSL,
		"magnitude":          magnitudeWGSL,
		"peak_finalize":      peakFinalizeWGSL,
		"peak_find":          peakFindWGSL,
		"peak_reduce":        peakReduceWGSL,
	}

	for name, source := range shaders {
		t.Run(name, func(t *testing.T) {
			spirv, err := shader.Compile(source)
			if err != nil {
				t.Fatalf("compile WGSL: %v", err)
			}
			cmd := exec.Command(validator, "--target-env", "vulkan1.1", "-")
			cmd.Stdin = bytes.NewReader(spirv)
			if output, err := cmd.CombinedOutput(); err != nil {
				t.Fatalf("spirv-val: %v\n%s", err, output)
			}
		})
	}
}
