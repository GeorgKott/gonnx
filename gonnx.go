package gonnx

import (
	"fmt"
	"io"
	"os"

	"google.golang.org/protobuf/proto"

	dto "github.com/GeorgKott/gonnx/generated"
)

type OnnxModel struct {
	ModelProto *dto.ModelProto
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

func (o *OnnxModel) Run(input InputData) ([]float32, error) {
	data, err := input.PrepareData()
	if err != nil {
		return nil, fmt.Errorf("error preparing input data: %v", err)
	}

	result, ok := data.([]float32)
	if !ok {
		return nil, fmt.Errorf("unexpected data format")
	}

	return result, nil
}
