package gonnx_test

import (
	"testing"

	"github.com/stretchr/testify/assert"

	. "github.com/GeorgKott/gonnx"
)

func TestOnnxModelRun(t *testing.T) {
	model, err := New("testdata/iris_nn.onnx")

	assert.NoError(t, err)
	assert.NotNil(t, model)

	input := &TensorInput{Data: []float32{5.1, 3.5, 1.4, 0.2}}

	output, err := model.Run(input) // 2.8726451 ,  0.41747826, -1.7224243

	assert.NoError(t, err)

	assert.NotEmpty(t, output)
}
