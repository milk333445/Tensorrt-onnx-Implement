import yaml
import torch

from inferencer import Detector, ONNXDetector, TensorRTDetector, Torch2trtDetector

class DetectorBuilder():
    def __init__(self, file):
        assert file.endswith(('.yaml', '.yml'))
        with open(file, 'r') as f:
            setting = yaml.safe_load(f)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_names = []
        for key in setting.keys():
            weight_path = setting[key]['weight']
            if weight_path.endswith('.onnx'):
                print(f"Loading ONNX model: {weight_path}")
                detector = ONNXDetector(
                    model_path=weight_path,
                    device=device.type,  # 'cuda' or 'cpu'
                    size=setting[key]['size'],
                    conf_thres=setting[key]['conf_threshold'], 
                    iou_thres=setting[key]['iou_threshold'],
                    classes=setting[key]['classes'],
                    fp16=setting[key]['fp16']
                )
            elif weight_path.endswith(('.pt', '.pth')):
                if '_trt' in weight_path:
                    print(f"Loading TensorRT model: {weight_path}")
                    detector = Torch2trtDetector(
                        model_path=weight_path,
                        device=device.type,  # 'cuda' or 'cpu'
                        size=setting[key]['size'],
                        conf_thres=setting[key]['conf_threshold'], 
                        iou_thres=setting[key]['iou_threshold'],
                        classes=setting[key]['classes'],
                        fp16=setting[key]['fp16']
                    )
                else:
                    detector = Detector(
                        model_path=weight_path,
                        device=device,
                        size=setting[key]['size'],
                        conf_thres=setting[key]['conf_threshold'],
                        iou_thres=setting[key]['iou_threshold'],
                        classes=setting[key]['classes'],
                        fp16=setting[key]['fp16']
                        )
                    class_names = detector.class_names
                    setattr(self, key + '_cls', class_names)
            elif weight_path.endswith('.engine'):
                print(f"Loading TensorRT model: {weight_path}")
                detector = TensorRTDetector(
                    model_path=weight_path,
                    device=device.type,  # 'cuda' or 'cpu'
                    size=setting[key]['size'],
                    conf_thres=setting[key]['conf_threshold'], 
                    iou_thres=setting[key]['iou_threshold'],
                    classes=setting[key]['classes'],
                    fp16=setting[key]['fp16']
                )
            
            else:
                raise ValueError(f"Unsupported model format: {weight_path}")  
            setattr(self, key, detector)
            self.model_names.append(key)
            
    def run(self, images, roi_list=None):
        for name in self.model_names:
            results = getattr(self, name).run(images, roi_list)
            setattr(self, name + '_results', results)
            
    def destroy_detectors(self):
        for name in self.model_names:
            detector = getattr(self, name, None)
            if detector is not None:
                detector.destroy()
                delattr(self, name)