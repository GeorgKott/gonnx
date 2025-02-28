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
	if err := o.validateModel(input); err != nil {
		return nil, err
	}

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

func (o *OnnxModel) validateModel(input InputData) error {
	if o.ModelProto == nil {
		return fmt.Errorf("modelProto is nil")
	}
	if o.ModelProto.Graph == nil {
		return fmt.Errorf("graph is nil")
	}
	if len(o.ModelProto.Graph.Input) == 0 {
		return fmt.Errorf("model has no inputs")
	}
	inputInfo := o.ModelProto.Graph.Input[0]
	if inputInfo.Type == nil {
		return fmt.Errorf("input type is nil")
	}

	expectedInputType, err := GetProtobufInputType(inputInfo.Type)
	if err != nil {
		return fmt.Errorf("error determining expected input type: %v", err)
	}

	if input.GetInputType() != expectedInputType {
		return fmt.Errorf("expected input type: %s, but got: %s", expectedInputType, input.GetInputType())
	}

	if expectedInputType == TensorType {
		data, err := input.PrepareData()
		if err != nil {
			return fmt.Errorf("error preparing input data: %v", err)
		}

		tensorData, ok := data.([]float32)
		if !ok {
			return fmt.Errorf("unexpected data format, expected []float32")
		}

		expectedSize := 1
		tensorType, ok := inputInfo.Type.Value.(*dto.TypeProto_TensorType)
		if !ok {
			return fmt.Errorf("failed to extract tensor shape")
		}

		for _, dim := range tensorType.TensorType.Shape.Dim {
			if dim.GetDimValue() == 0 {
				return fmt.Errorf("dynamic dimensions are not supported")
			}

			expectedSize *= int(dim.GetDimValue())
		}

		if len(tensorData) != expectedSize {
			return fmt.Errorf("invalid input size: expected %d, got %d", expectedSize, len(tensorData))
		}
	}

	return nil
}
