package gonnx

type InputData interface {
	PrepareData() (interface{}, error)
}

type TensorInput struct {
	Data []float32
}

func (t *TensorInput) PrepareData() (interface{}, error) {
	return t.Data, nil
}
