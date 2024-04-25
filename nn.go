// nn Neural Network algorithm
package nn

import (
	"math"
	"math/rand"
)

// Acc accumlate
func Acc[T float32 | float64](a, b []T) {
	for i := range a {
		a[i] += b[i]
	}
}

// RMSNorm Root Mean Square Norimalization
func RMSNorm[T float32 | float64](o, x, weight []T) {
	var sum T
	// sum square of x
	for _, v := range x {
		sum += v * v
	}
	sum /= T(len(x))
	sum += 1e-5
	sum = T(math.Sqrt(float64(sum)))
	// normalize and scale
	for i := range o {
		o[i] = weight[i] * x[i] / sum
	}
}

// SoftMax turns a vector of K real values into a vector of K real values that sum to 1
func SoftMax[T float32 | float64](x []T) {
	// find max for numerical stability
	m := x[0]
	for _, v := range x {
		if v > m {
			m = v
		}
	}

	// exp and sum
	var sum T
	for i := range x {
		x[i] = T(math.Exp(float64(x[i] - m)))
		sum += x[i]
	}
	// normalize
	for i := range x {
		x[i] /= sum
	}
}

// MatMul matrix product of two array
// W(d, n) @ x (n, ) -> out (d, )
func MatMul[T float32 | float64](out, x, w []T) {
	for i := range out {
		var sum T
		for j := range x {
			sum += w[i*len(x)+j] * x[j]
		}
		out[i] = sum
	}
}

// Sample index from probabilities, they must sum to 1
func Sample[T float32 | float64](probabilities []T) int {
	r := T(rand.Float64())
	var cdf T
	for i, p := range probabilities {
		cdf += p
		if r < cdf {
			return i
		}
	}
	return len(probabilities) - 1
}

func ArgMax[T float32 | float64](v []T) int {
	maxi, maxv := 0, v[0]
	for i, v := range v {
		if v > maxv {
			maxi, maxv = i, v
		}
	}
	return maxi
}
