// nn Neural Network algorithm
package nn

import (
	"math"
	"math/rand"
	"sort"
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

// SampleTopP ("top-p" sampling, or "nucleus sampling") samples from the smallest set of
// tokens that exceed probability topp. This way we never sample tokens that
// have very low probabilities and are less likely to go "off the rails".
// Notes on llama2.c: here not reusing probability index slice, since practically it is as fast to request new one.
func SampleTopP[T float32 | float64](probabilities []T, topp T) int {
	type PI struct {
		prob  T
		index int
	}
	pis := make([]PI, 0, len(probabilities))

	// quicksort indices in descending order of probabilities
	// values smaller than (1 - topp) / (n - 1) cannot be part of the result
	// so for efficiency we crop these out as candidates before sorting
	cutoff := (1.0 - topp) / T(len(probabilities)-1)
	for i, p := range probabilities {
		if p >= cutoff {
			pis = append(pis, PI{prob: p, index: i})
		}
	}
	sort.Slice(pis, func(i, j int) bool { return pis[i].prob > pis[j].prob })

	// truncate the list where cumulative probability exceeds topp
	cumulativeProb := T(0)
	lastIdx := len(pis) - 1 // in case of rounding errors consider all elements
	for i, pi := range pis {
		cumulativeProb += pi.prob
		if cumulativeProb > topp {
			lastIdx = i
			break // we've exceeded topp by including lastIdx
		}
	}

	// sample from the truncated list
	r := T(rand.Float32()) * cumulativeProb
	cdf := T(0)
	for i := 0; i <= lastIdx; i++ {
		cdf += pis[i].prob
		if r < cdf {
			return pis[i].index
		}
	}

	return pis[lastIdx].index // in case of rounding errors
}
