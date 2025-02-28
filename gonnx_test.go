package gonnx_test

import (
	"testing"

	"github.com/stretchr/testify/assert"

	. "github.com/GeorgKott/gonnx"
)

func TestIrisNNSmallModelRun(t *testing.T) {
	model, err := New("testdata/iris_nn_small.onnx")

	assert.NoError(t, err)
	assert.NotNil(t, model)

	input := &TensorInput{Data: []float32{5.1, 3.5, 1.4, 0.2}}

	output, err := model.Run(input) // 1.4371707 , 0.6069982 , 0.26410568

	assert.NoError(t, err)

	assert.NotEmpty(t, output)
}

func TestIrisNNModelRun(t *testing.T) {
	model, err := New("testdata/iris_nn.onnx")

	assert.NoError(t, err)
	assert.NotNil(t, model)

	input := &TensorInput{Data: []float32{5.1, 3.5, 1.4, 0.2}}

	output, err := model.Run(input) // 3.9639542,  1.5718641, -2.2148843

	assert.NoError(t, err)

	assert.NotEmpty(t, output)
}

func TestIrisNRNModelRun(t *testing.T) {
	model, err := New("testdata/iris_nrn.onnx")

	assert.NoError(t, err)
	assert.NotNil(t, model)

	input := &TensorInput{Data: []float32{5.1, 3.5, 1.4, 0.2}}

	output, err := model.Run(input) // 2.8726451 ,  0.41747826, -1.7224243

	assert.NoError(t, err)

	assert.NotEmpty(t, output)
}
