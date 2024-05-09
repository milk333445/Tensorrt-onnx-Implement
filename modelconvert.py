import torch
from PIL import Image
from torchvision import transforms
import os
import sys
import yaml
import time

from inferencer import load_model

def torch2trt(model_path, output_path, input_shape=(1, 3, 640, 640), fp16_mode=False):
    """
    Convert a PyTorch model to TensorRT.

    Args:
        model_path (str): Path to the PyTorch model file.
        output_path (str): Path where the TensorRT model will be saved.
        input_shape (tuple): Shape of the input tensor.
        fp16_mode (bool): Whether to enable FP16 mode for the TensorRT model.

    Returns:
        None
    """
    import tensorrt as trt
    from torch2trt import torch2trt
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    try:
        det_model = load_model(model_path, device=device)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    x = torch.ones(input_shape).to(device)
    
    print('Start conversion...')
    t = time.time()
    try:
        model_trt = torch2trt(det_model, [x], fp16_mode=fp16_mode, log_level=trt.Logger.VERBOSE)
        torch.save(model_trt.state_dict(), output_path)
        print(f"Conversion completed in {time.time() - t:.2f} sec.")
        print('Model successfully converted and saved.')
    except Exception as e:
        print(f"Failed to convert model: {e}")

def onnx2trt(image_path, onnx_file, trt_file, work_space=1, image_size=(640, 640), batch_size=1, dynamic=False, half=False):
    """
    Convert an ONNX model to a TensorRT engine, with image preprocessing and engine serialization.

    Parameters:
    - image_path: Path to the input image.
    - onnx_file: Path to the ONNX model file.
    - trt_file: Path to save the serialized TensorRT engine.
    - image_size: Tuple specifying the size of the image.
    - batch_size: Number of images per batch.
    - dynamic: Enable dynamic batching.
    - half: Enable half precision computation.
    """
    import tensorrt as trt
    # Load and process the image
    if image_path is None or not os.path.exists(image_path):
        print("Random image used for conversion.")
        image = torch.randn((batch_size, 3, *image_size))
        print("Random image shape:", image.shape)
    else:
        image = Image.open(image_path).convert('RGB')
        resize = transforms.Resize(image_size)
        to_tensor = transforms.ToTensor()
        image = to_tensor(resize(image)).unsqueeze(0)  # Add batch dimension
        image = image.repeat(batch_size, 1, 1, 1)
    
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    config.max_workspace_size = work_space << 30  # 1GB
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)

    # Parse ONNX model
    if not parser.parse_from_file(str(onnx_file)):
        print('ERROR: Failed to parse the ONNX file.')
        for error in range(parser.num_errors):
            print(parser.get_error(error))
        raise RuntimeError(f"Failed to load ONNX file: {onnx_file}")

    if dynamic:
        profile = builder.create_optimization_profile()
        for inp in network.get_inputs():
            profile.set_shape(inp.name, (1, *image.shape[1:]), (max(1, image.shape[0] // 2), *image.shape[1:]), image.shape)
        config.add_optimization_profile(profile)

    # Enable FP16 computation if supported and requested
    if builder.platform_has_fast_fp16 and half:
        config.set_flag(trt.BuilderFlag.FP16)
        
    # Build the engine
    with builder.build_engine(network, config) as engine:
        with open(trt_file, 'wb') as f:
            f.write(engine.serialize())

    print("Conversion completed successfully.")

def _torch2onnx(
    model, 
    im,
    output_path, 
    opset_version=12, 
    dynamic=False, # dynamic axes not supported on GPU
    simplify_model=False,
    ):
    """
    Converts a PyTorch model to ONNX format.

    Args:
        model (torch.nn.Module): The PyTorch model to convert.
        im (torch.Tensor): The input tensor to the model.
        output_path (str): The path to save the converted ONNX model.
        opset_version (int, optional): The ONNX opset version to use. Defaults to 12.
        dynamic (bool, optional): Whether to use dynamic axes. Defaults to False.
        simplify_model (bool, optional): Whether to simplify the ONNX model. Defaults to False.

    Returns:
        onnx.ModelProto: The converted ONNX model.
    """
    use_onnx = False
    if sys.version_info.major == 3 and sys.version_info.minor == 8:
        try:
            import onnx
            from onnxsim import simplify
            use_onnx = True
        except ImportError:
            print("Please install onnx and onnx-simplifier to use this function. make sure you have python 3.8.")

    model.eval()
    
    output_name = ['output0']
    if dynamic:
        dynamic_axes = {"images": {0: "batch", 2: "height", 3: "width"}}  # shape(1,3,640,640)
        dynamic_axes["output0"] = {0: "batch", 1: "anchors"}  # shape(1,25200,85)
    
    
    torch.onnx.export(
        model.cpu() if dynamic else model, # dynamic axes not supported on GPU
        im.cpu() if dynamic else im,
        output_path,
        verbose=False,
        opset_version=opset_version,
        input_names=['images'],
        output_names=output_name,
        dynamic_axes=dynamic_axes if dynamic else None,
    )
    
    # check 
    if use_onnx:
        model_onnx = onnx.load(output_path)
        onnx.checker.check_model(model_onnx)
        
        # metadata
        d = {"stride": int(max(model.stride)), "names": model.names}
        for k, v in d.items():
            meta = model_onnx.metadata_props.add()
            meta.key, meta.value = k, str(v)
        onnx.save(model_onnx, output_path)
        
        # simplify
        if simplify_model:
            try:
                # need to install onnxruntime-gpu
                model_onnx, check = simplify(model_onnx)
                assert check, "assert check failed"
                onnx.save(model_onnx, output_path)
            except Exception as e:
                print('simplify onnx failed:', e)
    else:
        model_onnx = None
    
    return model_onnx

def torch2onnx(
    model_path, 
    image_path, 
    output_path, 
    image_size=(640, 640), 
    batch_size=1, 
    dynamic=False, 
    simplify=False, 
    half=False, # Half is not supported by CPU, only CUDA.
    opset_version=12
    ):
    """
    Load YOLO model and image, perform dry runs, and export to ONNX with adjustable image size and batch size.

    Parameters:
    - model_path: Path to the YOLO model file (.pt).
    - image_path: Path to the image file.
    - output_path: Path to save the ONNX model file.
    - image_size: Tuple of two integers (width, height) for resizing the input image.
    - batch_size: Number of images per batch.
    - dynamic: Whether to enable dynamic axes for ONNX model.
    - simplify: Whether to simplify the ONNX model after export.
    - half: Whether to convert model and image to half precision (float16).
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(model_path, device=device)
    model.to(device)
    model.eval()

    # Load and preprocess the image
    if image_path is None or not os.path.exists(image_path):
        print("Random image used for export.")
        image = torch.randn((batch_size, 3, *image_size), device=device)
        print("Random image shape:", image.shape)
    else:
        image = Image.open(image_path).convert('RGB')
        resize = transforms.Resize(image_size)
        to_tensor = transforms.ToTensor()
        image = to_tensor(resize(image)).unsqueeze(0)  # Add batch dimension
        image = image.repeat(batch_size, 1, 1, 1).to(device)  # Repeat image for batch size

    print("Model device:", next(model.parameters()).device)
    print("Image device:", image.device)

    # Setup model for export
    for k, m in model.named_modules():
        m.inplace = False
        m.dynamic = dynamic
        m.export = True

    # Perform dry runs
    for _ in range(2):
        y = model(image)  # Dry runs

    # Convert model and image to half precision if required
    if half:
        model.half()
        image = image.half()

    # Export the model to ONNX
    model_onnx = _torch2onnx(model, image, output_path, opset_version=opset_version, dynamic=dynamic, simplify_model=simplify)
    return model_onnx

class ModelConverter:
    """
    Converts models from YOLO format to ONNX format and then from ONNX format to TensorRT format.
    
    Args:
        config_path (str): The path to the configuration file.
        
    Attributes:
        config (dict): The configuration settings for the model converter.
    """
    def __init__(self, config_path):
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            self.config = config['model_converter']
        except FileNotFoundError:
            sys.exit(f"Configuration file not found: {config_path}")
        except yaml.YAMLError as exc:
            sys.exit(f"Error parsing the configuration file: {exc}")
        
    def convert_torch_to_onnx(self):
        """
        Converts a YOLO model to ONNX format.
        
        Raises:
            Exception: If the conversion fails.
        """
        print('Converting model to ONNX')
        try:
            torch2onnx(
                model_path=self.config['model_path'],
                image_path=self.config['image_path'],
                output_path=self.config['onnx_output_path'],
                image_size=tuple(self.config['image_size']),
                batch_size=self.config['batch_size'],
                dynamic=self.config['dynamic'],
                simplify=self.config['simplify'],
                half=self.config['half'],
                opset_version=self.config['opset_version']
            )
        except Exception as e:
            sys.exit(f"Failed to convert YOLO to ONNX: {e}")
        
    def convert_onnx_to_tensorrt(self):
        """
        Converts an ONNX model to TensorRT format.
        
        Raises:
            Exception: If the conversion fails.
        """
        # 檢查是否有 onnx 檔案
        if not os.path.exists(self.config['onnx_output_path']):
            print(f"ONNX file not found: {self.config['onnx_output_path']}")
            print('try to convert model to ONNX')
            self.convert_torch_to_onnx()
            
        print('Converting ONNX to TensorRT')
        try:
            onnx2trt(
                image_path=self.config['image_path'],
                onnx_file=self.config['onnx_output_path'],
                work_space=self.config['work_space'],
                trt_file=self.config['trt_output_path'],
                image_size=tuple(self.config['image_size']),
                batch_size=self.config['batch_size'],
                dynamic=self.config['dynamic'],
                half=self.config['half']
            )
        except Exception as e:
            sys.exit(f"Failed to convert ONNX to TensorRT: {e}")