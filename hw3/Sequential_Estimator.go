package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"
)

func Univariate_gaussian(mean float64, variance float64) float64 {
	u := rand.Float64()
	v := rand.Float64()
	z := math.Sqrt(-2*math.Log(u)) * math.Cos(2*math.Pi*v)
	x := (z * math.Sqrt(variance)) + mean
	return x
}
func sum(vector []float64) float64 {
	value := 0.0
	for i := 0; i < len(vector); i++ {
		value = value + vector[i]
	}
	return value
}
func sample_var(vector []float64, s_mean float64) float64 {
	value := 0.0
	for i := 0; i < len(vector); i++ {
		value = value + math.Pow((vector[i]-s_mean), 2)
	}
	if len(vector) == 1 {
		return 0.0
	}
	value = value / float64(len(vector)-1)
	return value
}
func Sequential_Estimator(mean float64, variance float64) {
	var s_mean, s_variance float64
	var x_sample []float64
	for {
		x := Univariate_gaussian(mean, variance)
		x_sample = append(x_sample, x)
		log.Println("Add data point:", x)
		s_mean = sum(x_sample) / float64(len(x_sample))
		s_variance = sample_var(x_sample, s_mean)
		log.Printf("Mean = %-10f Variance = %-10f\n", s_mean, s_variance)
		if math.Abs(mean-s_mean) < 0.01 && math.Abs(variance-s_variance) < 0.01 {
			break
		}
	}
}
func main() {
	var mean, variance float64
	fmt.Printf("mean = ")
	fmt.Scanf("%f", &mean)
	fmt.Printf("var = ")
	fmt.Scanf("%f", &variance)
	log.Printf("Data point source function: N(%.1f, %.1f)\n", mean, variance)
	Sequential_Estimator(mean, variance)
}
