# Tensorrt onnx Implement

-   This repository provides a versatile framework for converting PyTorch model formats into TensorRT and ONNX, primarily servicing the YOLOv5 architecture. However, it can be easily applied to various neural network models. Simple configuration adjustments allow for rapid model transformation into different engines, achieving industrial-grade inference speeds.

## Installation

-   To quickly proceed with the model conversion, clone the repository:

```python=
git clone https://github.com/milk333445/Tensorrt-onnx-Implement.git
```

-   Ensure your environment includes PyTorch (for ONNX conversion and subsequent inference), TensorRT packages (for converting to TensorRT format), and torch2trt (for direct conversion of PyTorch to TensorRT format). Install these and verify their functionality. Here is the torch2trt repository:

```python=
git clone https://github.com/NVIDIA-AI-IOT/torch2trt.git
```

## Model Conversion Configurations

### PyTorch -> ONNX -> TensorRT Configurations

-   Main parameters are set in a YAML file, edit this file to configure the conversion:

```python=
./configs/example/convert.yaml
model_converter:
    model_path: "yolov5s.pt" # Path to your model
    image_path: "test.jpg" # Example image for the conversion process, optional
    onnx_output_path: "yolov5s.onnx" # Filename for the ONNX format output
    trt_output_path: "yolov5s.engine" # Filename for the TensorRT format output
    work_space: 2 # GB # Space setting for TensorRT optimization strategy
    image_size: # Inference size after conversion, consistent with the original model
        - 640
        - 640
    batch_size: 1 # Batch size format after conversion
    dynamic: False # Enable dynamic batch size
    simplify: False # Simplify ONNX model
    half: False # Convert to half precision
    opset_version: 12

```

-   Start the model conversion process:

```python=
from modelconvert import ModelConverter
modelconverter = ModelConverter('./configs/example/convert.yaml')
modelconverter.convert_torch_to_onnx() # torch->onnx
modelconverter.convert_onnx_to_tensorrt() # torch->onnx->tensorrt
```

### PyTorch -> TensorRT Configurations

```python=
from modelconvert import torch2trt
model_path = 'yolov5s.pt'
trt_path = 'yolov5s_trt.pth'
input_shape=(1, 3, 640, 640)
fp16_mode=False

torch2trt(model_path, trt_path, input_shape, fp16_mode)
```

## Model Inference Configurations

### High-Level API Inference Example

-   Configure the model for inference in a YAML file, allowing framework initialization with multiple models:

```python=
./configs/example/detection.yaml
normal:
    weight: "./weights/yolov5s.pt"
    size: 640
    conf_threshold: 0.5
    iou_threshold: 0.45
    fp16: false
    classes: [0]
onnx:
    weight: "./yolov5s.onnx"
    size: 640
    conf_threshold: 0.5
    iou_threshold: 0.45
    fp16: False
    classes:
tensorrt:
    weight: "./yolov5s.engine"
    size: 640
    conf_threshold: 0.5
    iou_threshold: 0.45
    fp16: False
    classes:
trt:
    weight: "./yolov5s_trt.pth"
    size: 640
    conf_threshold: 0.5
    iou_threshold: 0.45
    fp16: False
    classes:
```

-   Initiate model initialization:

```python=
from builder import DetectorBuilder
detector = DetectorBuilder('./configs/example/detection.yaml')
```

-   Begin inference:

```python=
img = cv2.imread('test.jpg')
# Normal PyTorch model inference
result = detector.normal.run(img)
# ONNX engine inference
result = detector.onnx.run(img)
# TensorRT engine inference
result = detector.tensorrt.run(img)
# torch2trt engine inference
result = detector.trt.run(img)
```

### Low-Level API Inference Example

#### Normal PyTorch

```python=
from inferencer import Detector
model_path = "./weights/yolov5s.pt"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
size = 640
conf_thres = 0.25
iou_thres = 0.45
detector = Detector(model_path, device, size, conf_thres, iou_thres)

img = cv2.imread('test.jpg')
result = detector.run(img)
```

#### ONNX Engine Inference

```python=
from inferencer import ONNXDetector
model_path = "./yolov5s.onnx"
device = 'cuda'
size = 640
conf_thres = 0.25
iou_thres = 0.45
detector = ONNXDetector(model_path, device, size, conf_thres, iou_thres)

img = cv2.imread('test.jpg')
result = detector.run(img)
```

#### TensorRT Engine Inference - Single Thread Version

```python=
from inferencer import TensorRTDetector
model_path = "./yolov5s.engine"
device = 'cuda'
size = 640
conf_thres = 0.25
iou_thres = 0.45
detector = TensorRTDetector(model_path, device, size, conf_thres, iou_thres)

img = cv2.imread('test.jpg')
result = detector.run(img)
```

#### TensorRT Engine Inference - Multi-Threaded Version

```python=
from inferencer import TensorRTDetector_Threading
model_path = "./yolov5s.engine"
device = 'cuda'
size = 640
conf_thres = 0.25
iou_thres = 0.45
detector = TensorRTDetector_Threading(model_path, device, size, conf_thres, iou_thres)

img = cv2.imread('test.jpg')
result = detector.run(img)
detector.destroy() # Manually release resources after use due to explicit CUDA context creation in the multi-threaded version
```

#### Torch2trt Engine Inference

```python=
from inferencer import Torch2trtDetector
model_path = "./yolov5s_trt.pth"
device = 'cuda'
size = 640
conf_thres = 0.25
iou_thres = 0.45
detector = Torch2trtDetector(model_path, device, size, conf_thres, iou_thres)

img = cv2.imread('test.jpg')
result = detector.run(img)
```
