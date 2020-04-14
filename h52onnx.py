from src.keras_utils			import load_model
import keras2onnx
import onnx


net_path = '/root/alpr-unconstrained-master/models/model-scracth.h5'
net = load_model(net_path)
onnx_model = keras2onnx.convert_keras(net, name=None, doc_string='', target_opset=None, channel_first_inputs=None)
temp_model_file = '/root/alpr-unconstrained-master/models/wpod0401.onnx'
onnx.save_model(onnx_model, temp_model_file)
