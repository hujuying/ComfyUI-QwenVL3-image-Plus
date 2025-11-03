# ComfyUI-QwenVL3-image-Plus
# 修改后支持两个提示词输入框并自动组合（Plus版本，避免冲突）

import torch
import time
import json
import platform
import psutil
import numpy as np
from PIL import Image
from enum import Enum
from pathlib import Path
from transformers import AutoModelForVision2Seq, AutoProcessor, AutoTokenizer
from huggingface_hub import snapshot_download
import folder_paths
import gc

NODE_DIR = Path(__file__).parent
CONFIG_PATH = NODE_DIR / "config.json"
TEXT_DELIMITER = "[SEP]"  # Batch text delimiter

def load_model_configs():
    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {CONFIG_PATH}")
        return {}
    except json.JSONDecodeError:
        print(f"Error: Failed to parse configuration file.")
        return {}

MODEL_CONFIGS = load_model_configs()
FIXED_MODEL_NAME = "Qwen3-VL-4B-Instruct-FP8"

def get_model_info(model_name: str) -> dict:
    return MODEL_CONFIGS.get(model_name, {})

def get_device_info() -> dict:
    gpu_info = {}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        total_mem = props.total_memory / 1024**3
        gpu_info = {"available": True, "total_memory": total_mem, "free_memory": total_mem - (torch.cuda.memory_allocated(0) / 1024**3)}
    else:
        gpu_info = {"available": False, "total_memory": 0, "free_memory": 0}

    sys_mem = psutil.virtual_memory()
    sys_mem_info = {"total": sys_mem.total / 1024**3, "available": sys_mem.available / 1024**3}

    device_info = {"gpu": gpu_info, "system_memory": sys_mem_info, "device_type": "cpu", "recommended_device": "cpu", "memory_sufficient": True, "warning_message": ""}

    if platform.system() == "Darwin" and platform.processor() == "arm":
        device_info.update({"device_type": "apple_silicon", "recommended_device": "mps"})
    elif gpu_info["available"]:
        device_info.update({"device_type": "nvidia_gpu", "recommended_device": "cuda"})
    return device_info

def check_flash_attention() -> bool:
    try:
        import flash_attn
        if torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability()
            return major >= 8
    except ImportError: return False
    return False

class ImageProcessor:
    def tensor_to_pil(self, image_tensor: torch.Tensor) -> Image.Image:
        image_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(image_np)

    def resize_if_needed(self, pil_image: Image.Image, max_size: int = 768) -> Image.Image:
        width, height = pil_image.size
        max_dimension = max(width, height)
        
        if max_dimension > max_size:
            scale = max_size / max_dimension
            new_width = int(width * scale)
            new_height = int(height * scale)
            resized_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
            print(f"  Image resized from {width}x{height} to {new_width}x{new_height} (max edge: {max_dimension} -> {max_size})")
            return resized_image
        else:
            print(f"  Image size {width}x{height} is within limit (max edge: {max_dimension} <= {max_size}), no resize needed")
            return pil_image

    def batch_to_pil_list(self, image_batch: torch.Tensor) -> list[Image.Image]:
        if image_batch.dim() == 3:
            return [self.tensor_to_pil(image_batch)]
        elif image_batch.dim() == 4:
            return [self.tensor_to_pil(img) for img in image_batch]
        else:
            raise ValueError("Input image tensor must be 3D or 4D")

class ModelHandler:
    def __init__(self, configs):
        self.configs = configs
        custom_path = configs.get("custom_model_path", "")
        if custom_path and custom_path.strip():
            self.models_dir = Path(custom_path.strip())
            print(f"Using custom model path: {self.models_dir}")
        else:
            self.models_dir = Path(folder_paths.models_dir) / "Qwen"
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def get_model_path(self, model_name):
        model_info = self.configs.get(model_name)
        if not model_info:
            raise ValueError(f"Model '{model_name}' not found in configuration.")

        repo_id = model_info['repo_id']
        model_folder_name = repo_id.split('/')[-1]
        model_path = self.models_dir / model_folder_name
        
        if not model_path.exists() or not any(model_path.iterdir()):
            print(f"Model not found locally. Downloading '{model_name}' to {model_path}...")
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(model_path),
                local_dir_use_symlinks=False,
                ignore_patterns=["*.md", ".git*"]
            )
            print(f"Model '{model_name}' downloaded successfully.")
        else:
            print(f"Model '{model_name}' found at: {model_path}")
        
        return str(model_path)

class QwenVL3ImagePlus:
    def __init__(self):
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.current_device = None
        self.device_info = get_device_info()
        self.model_handler = ModelHandler(MODEL_CONFIGS)
        self.image_processor = ImageProcessor()
        print(f"QwenVL3ImagePlus Node Initialized. ID: {id(self)}. Device: {self.device_info['device_type']}")

    def clear_model_resources(self):
        if self.model is not None:
            print(f"\n[{id(self)}] ========== Starting Model Unload ==========")
            vram_before = 0
            if torch.cuda.is_available():
                vram_before = torch.cuda.memory_allocated(0) / 1024**2
                vram_reserved_before = torch.cuda.memory_reserved(0) / 1024**2
                print(f"[{id(self)}] VRAM allocated before unload: {vram_before:.2f} MB")
                print(f"[{id(self)}] VRAM reserved before unload: {vram_reserved_before:.2f} MB")
            
            try:
                print(f"[{id(self)}] Moving model to CPU...")
                self.model.to('cpu')
                print(f"[{id(self)}] Model moved to CPU successfully.")
            except Exception as e:
                print(f"[{id(self)}] Warning: Could not move model to CPU: {e}")

            print(f"[{id(self)}] Deleting model references...")
            del self.model
            del self.processor
            del self.tokenizer
            self.model = None
            self.processor = None
            self.tokenizer = None
            self.current_device = None
            print(f"[{id(self)}] All references deleted.")
            
            print(f"[{id(self)}] Running garbage collection...")
            for i in range(3):
                collected = gc.collect()
                print(f"[{id(self)}]   GC pass {i+1}: collected {collected} objects")
            
            if torch.cuda.is_available():
                print(f"[{id(self)}] Clearing CUDA cache...")
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                vram_after = torch.cuda.memory_allocated(0) / 1024**2
                vram_reserved_after = torch.cuda.memory_reserved(0) / 1024**2
                vram_freed = vram_before - vram_after
                
                print(f"[{id(self)}] VRAM allocated after unload: {vram_after:.2f} MB")
                print(f"[{id(self)}] VRAM reserved after unload: {vram_reserved_after:.2f} MB")
                print(f"[{id(self)}] VRAM freed: {vram_freed:.2f} MB")
            
            print(f"[{id(self)}] ========== Model Unload Complete ==========\n")
        else:
            print(f"[{id(self)}] No model resources to release (model is None).")

    def load_model(self, device: str = "auto"):
        effective_device = self.device_info["recommended_device"] if device == "auto" else device
        
        if self.model is not None and self.current_device == effective_device:
            return

        self.clear_model_resources()

        model_info = get_model_info(FIXED_MODEL_NAME)
        if self.device_info["gpu"]["available"]:
            major, minor = torch.cuda.get_device_capability()
            cc = major + minor / 10
            if cc < 8.9:
                raise ValueError(
                    f"FP8 models require a GPU with Compute Capability 8.9 or higher (e.g., RTX 4090). "
                    f"Your GPU's capability is {cc}. This node only supports FP8 models."
                )

        model_path = self.model_handler.get_model_path(FIXED_MODEL_NAME)
        
        device_map = "auto"
        if effective_device == "cuda" and torch.cuda.is_available(): device_map = {"": 0}

        load_kwargs = {
            "device_map": device_map, 
            "dtype": torch.float16, 
            "attn_implementation": "flash_attention_2" if check_flash_attention() else "sdpa", 
            "use_safetensors": True
        }

        print(f"Loading model '{FIXED_MODEL_NAME}'...")
        self.model = AutoModelForVision2Seq.from_pretrained(model_path, **load_kwargs).eval()
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        self.current_device = effective_device
        print("Model loaded successfully.")

    @classmethod
    def INPUT_TYPES(cls):
        preset_prompts = MODEL_CONFIGS.get("预设提示词", ["Describe this image in detail."])
        return {
            "required": {
                "image": ("IMAGE",),
                "preset_prompt": (preset_prompts, {"default": preset_prompts[0]}),
                "custom_prompt": ("STRING", {"default": "", "multiline": True, "placeholder": "If provided, this will be combined with additional prompt"}),
                "additional_prompt": ("STRING", {"default": "", "multiline": True, "placeholder": "Additional prompt to combine with custom prompt"}),
                "max_tokens": ("INT", {"default": 1024, "min": 64, "max": 2048, "step": 16}),
                "temperature": ("FLOAT", {"default": 0.6, "min": 0.1, "max": 1.0, "step": 0.1}),
                "device": (["auto", "cuda", "cpu", "mps"], {"default": "auto"}),
                "keep_model_loaded": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 1, "min": 1, "max": 0xFFFFFFFFFFFFFFFF}),
            }
        }

    RETURN_TYPES = ("STRING", "QWEN_PIPE_PLUS",)
    RETURN_NAMES = ("text", "qwen_pipe_plus",)
    FUNCTION = "process"
    CATEGORY = "QwenvlPlus"

    @torch.no_grad()
    def process(self, image, preset_prompt, max_tokens, temperature, device, seed, 
                custom_prompt="", additional_prompt="", keep_model_loaded=True):
        start_time = time.time()
        
        # 处理提示词组合逻辑
        prompt_parts = []
        # 收集非空提示词部分
        if custom_prompt.strip():
            prompt_parts.append(custom_prompt.strip())
        if additional_prompt.strip():
            prompt_parts.append(additional_prompt.strip())
        # 如果有自定义提示词组合，则使用组合结果，否则使用预设提示词
        final_prompt = ", ".join(prompt_parts) if prompt_parts else preset_prompt
        
        try:
            torch.manual_seed(seed)
            self.load_model(device)
            print(f"Processing in instance {id(self)}. Model is on device: {self.model.device}")
            effective_device = self.current_device
            
            pil_images = self.image_processor.batch_to_pil_list(image)
            print(f"Processing a batch of {len(pil_images)} image(s) individually.")

            all_descriptions = []
            for i, pil_image in enumerate(pil_images):
                print(f"Processing image {i+1}/{len(pil_images)}...")
                
                pil_image = self.image_processor.resize_if_needed(pil_image, max_size=768)
                
                conversation = [{"role": "user", "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": final_prompt}
                ]}]

                text_prompt = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
                
                inputs = self.processor(text=text_prompt, images=[pil_image], return_tensors="pt")
                model_inputs = {k: v.to(effective_device) for k, v in inputs.items() if torch.is_tensor(v)}

                stop_tokens = [self.tokenizer.eos_token_id]
                if hasattr(self.tokenizer, 'eot_id'): stop_tokens.append(self.tokenizer.eot_id)

                gen_kwargs = {"max_new_tokens": max_tokens, "repetition_penalty": 1.2, "num_beams": 1, "eos_token_id": stop_tokens, "pad_token_id": self.tokenizer.pad_token_id}
                if 1 > 1:
                    gen_kwargs["do_sample"] = False
                else:
                    gen_kwargs.update({"do_sample": True, "temperature": temperature, "top_p": 0.9})

                outputs = self.model.generate(** model_inputs, **gen_kwargs)
                input_ids_len = model_inputs["input_ids"].shape[1]
                text = self.tokenizer.decode(outputs[0, input_ids_len:], skip_special_tokens=True)
                all_descriptions.append(text.strip())

            final_text = TEXT_DELIMITER.join(all_descriptions)
            print(f"Generation finished in {time.time() - start_time:.2f} seconds.")
            pipe = {"qwen_instance": self}
            print(f"Returning pipe from instance {id(self)}")
            return (final_text, pipe,)

        except (ValueError, RuntimeError) as e:
            error_message = f"ERROR: {str(e)}"
            print(error_message)
            pipe = {"qwen_instance": self}
            print(f"Returning pipe from instance {id(self)} after error")
            return (error_message, pipe,)
        finally:
            if not keep_model_loaded: self.clear_model_resources()

class QwenVLUnloadModelPlus:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "qwen_pipe_plus": ("QWEN_PIPE_PLUS",),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "unload"
    CATEGORY = "QwenvlPlus"

    def unload(self, qwen_pipe_plus):
        print(f"\n{'='*60}")
        print(f"[QwenVLUnloadModelPlus] Unload node triggered")
        print(f"[QwenVLUnloadModelPlus] Received pipe: {qwen_pipe_plus}")
        
        instance = qwen_pipe_plus.get("qwen_instance")
        print(f"[QwenVLUnloadModelPlus] Extracted instance: {instance}")
        print(f"[QwenVLUnloadModelPlus] Instance ID: {id(instance) if instance else 'None'}")

        if instance and hasattr(instance, 'clear_model_resources'):
            print(f"[QwenVLUnloadModelPlus] Calling clear_model_resources on instance {id(instance)}")
            instance.clear_model_resources()
            print(f"[QwenVLUnloadModelPlus] Unload completed successfully")
        else:
            print(f"[QwenVLUnloadModelPlus] ERROR: Could not find valid instance or clear_model_resources method")
        
        print(f"{'='*60}\n")
        return ()

NODE_CLASS_MAPPINGS = {
    "QwenVL3ImagePlus": QwenVL3ImagePlus,
    "QwenVLUnloadModelPlus": QwenVLUnloadModelPlus,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenVL3ImagePlus": "QwenVL3 Image Plus (FP8)",
    "QwenVLUnloadModelPlus": "QwenVL Unload Model Plus",
}