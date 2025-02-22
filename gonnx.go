package gonnx

import (
	"fmt"
	"io"
	"os"

	"google.golang.org/protobuf/proto"

	dto "github.com/GeorgKott/gonnx/generated"
)

type OnnxModel struct {
	ModelProto *dto.ModelProto // test
}

func New(modelPath string) (*OnnxModel, error) {
	file, err := os.Open(modelPath)
	if err != nil {
		return nil, fmt.Errorf("error opening model file: %v", err)
	}

	defer func() {
		_ = file.Close()
	}()

	data, err := io.ReadAll(file)
	if err != nil {
		return nil, fmt.Errorf("error reading model file: %v", err)
	}

	model := &dto.ModelProto{}
	err = proto.Unmarshal(data, model)
	if err != nil {
		return nil, fmt.Errorf("error unmarshalling protobuf data: %v", err)
	}

	return &OnnxModel{
		ModelProto: model,
	}, nil
}

func (o *OnnxModel) Run(input []float32) ([]float32, error) {
	output := make([]float32, 0, len(input))

	return output, nil
}
