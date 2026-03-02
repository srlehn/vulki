package imgproc

// nextPow2 returns the smallest power of 2 >= n.
func nextPow2(n int) int {
	p := 1
	for p < n {
		p <<= 1
	}
	return p
}

// log2i returns floor(log2(n)) for positive n.
func log2i(n int) int {
	r := 0
	v := n
	for v > 1 {
		v >>= 1
		r++
	}
	return r
}
