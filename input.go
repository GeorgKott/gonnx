package gonnx

import (
	"fmt"

	dto "github.com/GeorgKott/gonnx/generated"
)

const (
	TensorType       = "Tensor"
	SequenceType     = "Sequence"
	MapType          = "Map"
	OptionalType     = "Optional"
	SparseTensorType = "SparseTensor"
)

type InputData interface {
	PrepareData() (interface{}, error)
	GetInputType() string
}

type TensorInput struct {
	Data []float32
}

func (t *TensorInput) PrepareData() (interface{}, error) {
	return t.Data, nil
}

func (t *TensorInput) GetInputType() string {
	return TensorType
}

func GetProtobufInputType(typeProto *dto.TypeProto) (string, error) {
	if typeProto == nil || typeProto.Value == nil {
		return "", fmt.Errorf("nil typeProto or Value")
	}

	switch v := typeProto.Value.(type) {
	case *dto.TypeProto_TensorType:
		return TensorType, nil
	case *dto.TypeProto_SequenceType:
		return SequenceType, nil
	case *dto.TypeProto_MapType:
		return MapType, nil
	case *dto.TypeProto_OptionalType:
		return OptionalType, nil
	case *dto.TypeProto_SparseTensorType:
		return SparseTensorType, nil
	default:
		return "", fmt.Errorf("unsupported protobuf type: %T", v)
	}
}
