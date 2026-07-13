// Package testutils provides helpers shared by Vulki tests and benchmarks.
package testutils

import (
	"fmt"
	"os"
	"path/filepath"
	"sync"
)

var moduleRoot = sync.OnceValues(findModuleRoot)

// GetTestDataFilePath returns the absolute path of a file in the repository's
// testdata directory.
func GetTestDataFilePath(testFile string) (string, error) {
	root, err := moduleRoot()
	if err != nil {
		return "", err
	}
	return filepath.Join(root, "testdata", filepath.FromSlash(testFile)), nil
}

func findModuleRoot() (string, error) {
	directory, err := os.Getwd()
	if err != nil {
		return "", fmt.Errorf("testutils: get working directory: %w", err)
	}
	start := directory
	for {
		if _, err := os.Stat(filepath.Join(directory, "go.mod")); err == nil {
			return directory, nil
		} else if !os.IsNotExist(err) {
			return "", fmt.Errorf("testutils: inspect go.mod in %q: %w", directory, err)
		}
		parent := filepath.Dir(directory)
		if parent == directory {
			return "", fmt.Errorf("testutils: find module root from %q", start)
		}
		directory = parent
	}
}
