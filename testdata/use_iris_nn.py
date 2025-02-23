import onnxruntime as ort
import numpy as np

onnx_model_path = "iris_nn.onnx"
session = ort.InferenceSession(onnx_model_path)

input_data = np.array([[5.1, 3.5, 1.4, 0.2]], dtype=np.float32)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

output = session.run([output_name], {input_name: input_data})

print("Prediction:", output)