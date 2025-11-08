"""
Omni Model Manager
Handles loading and managing Qwen2.5-Omni model
"""

import torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from typing import Optional, Tuple
import sys
from pathlib import Path

# Add parent directory to path to import omni utilities
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to import process_mm_info if available
try:
    from qwen_omni_utils import process_mm_info
    HAS_OMNI_UTILS = True
except ImportError:
    HAS_OMNI_UTILS = False
    print("Warning: qwen_omni_utils not found. Install it for full multimodal support.")


class OmniModelManager:
    """Manages Qwen2.5-Omni model loading and generation"""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-Omni-3B",
        use_cpu_offload: bool = False,
        max_memory: Optional[dict] = None,
        use_flash_attention: bool = True
    ):
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.use_cpu_offload = use_cpu_offload
        self.max_memory = max_memory
        self.use_flash_attention = use_flash_attention
        self.talker_enabled = False
        
    def load_model(self):
        """Load Qwen2.5-Omni model with proper device handling"""
        print(f"Loading model: {self.model_name}")
        
        # Prepare model loading kwargs
        model_kwargs = {
            "torch_dtype": torch.float16,
            "device_map": "auto"
        }
        
        # Add flash attention if requested
        if self.use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"
            print("Using flash_attention_2 (talker will be disabled to avoid conflicts)")
        
        # Add CPU offload settings if requested
        if self.use_cpu_offload and self.max_memory:
            print(f"Using CPU/GPU split with memory config: {self.max_memory}")
            model_kwargs["max_memory"] = self.max_memory
            model_kwargs["offload_folder"] = "offload"
        else:
            print("Loading model on GPU (or CPU if no GPU available)...")
        
        # Load the model
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            self.model_name,
            **model_kwargs
        )
        
        # Handle talker: disable by default to avoid meta tensor errors
        talker_disabled = False
        if hasattr(self.model, 'disable_talker'):
            try:
                self.model.disable_talker()
                talker_disabled = True
                print("✓ Talker disabled (avoids meta tensor errors, no audio output)")
            except Exception as e:
                print(f"Warning: Could not disable talker: {e}")
        
        # Additional safety: Check if talker is in meta state and patch generate() if needed
        if hasattr(self.model, 'talker'):
            try:
                has_meta = any(p.is_meta for p in self.model.talker.parameters())
                if has_meta or not talker_disabled:
                    print("Warning: Talker may cause issues, patching generate() to handle it safely")
                    original_generate = self.model.generate
                    
                    def safe_generate(*args, return_audio=False, **kwargs):
                        if not return_audio:
                            talker_backup = getattr(self.model, 'talker', None)
                            try:
                                class DummyTalker:
                                    def generate(self, *args, **kwargs):
                                        return None
                                    def __getattr__(self, name):
                                        return lambda *args, **kwargs: None
                                
                                self.model.talker = DummyTalker()
                                result = original_generate(*args, return_audio=False, **kwargs)
                                return result
                            except Exception as e:
                                if hasattr(self.model, 'talker'):
                                    delattr(self.model, 'talker')
                                try:
                                    result = original_generate(*args, return_audio=False, **kwargs)
                                    return result
                                finally:
                                    if talker_backup is not None:
                                        self.model.talker = talker_backup
                            finally:
                                if talker_backup is not None and hasattr(self.model, 'talker'):
                                    self.model.talker = talker_backup
                        else:
                            return original_generate(*args, return_audio=True, **kwargs)
                    
                    self.model.generate = safe_generate
            except Exception as e:
                print(f"Warning: Could not patch talker: {e}")
        
        # Load processor
        self.processor = Qwen2_5OmniProcessor.from_pretrained(self.model_name)
        
        # Print device map
        print("\n=== Model Device Map ===")
        if hasattr(self.model, 'hf_device_map'):
            for name, device in list(self.model.hf_device_map.items())[:10]:
                print(f"{name}: {device}")
            if len(self.model.hf_device_map) > 10:
                print(f"... ({len(self.model.hf_device_map)} total modules)")
        else:
            print("Device map not available")
        
        print("✅ Model loaded successfully")
        
    def generate_response(
        self,
        text_prompt: str,
        audio_path: Optional[str] = None,
        image_path: Optional[str] = None,
        video_path: Optional[str] = None,
        max_new_tokens: int = 512,
        return_audio: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> Tuple[str, Optional[torch.Tensor]]:
        """
        Generate response from Qwen 2.5 Omni.
        
        Args:
            text_prompt: Text input/question
            audio_path: Optional path to audio file
            image_path: Optional path to image file
            video_path: Optional path to video file
            max_new_tokens: Maximum tokens to generate
            return_audio: Whether to return audio (requires talker to be enabled)
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
        
        Returns:
            If return_audio=True: (text_response, audio_tensor)
            If return_audio=False: (text_response, None)
        """
        if not self.model or not self.processor:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Prepare conversation with system prompt
        conversation = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
                ],
            },
            {"role": "user", "content": []}
        ]
        
        # Add multimodal inputs
        if audio_path:
            conversation[1]["content"].append({
                "type": "audio",
                "audio": audio_path
            })
        
        if image_path:
            conversation[1]["content"].append({
                "type": "image",
                "image": image_path
            })
        
        if video_path:
            conversation[1]["content"].append({
                "type": "video",
                "video": video_path
            })
        
        # Add text prompt
        conversation[1]["content"].append({
            "type": "text",
            "text": text_prompt
        })
        
        # Process inputs using process_mm_info if available
        if HAS_OMNI_UTILS:
            USE_AUDIO_IN_VIDEO = False
            text = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=False
            )
            audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
            
            inputs = self.processor(
                text=text,
                audio=audios,
                images=images,
                videos=videos,
                return_tensors="pt",
                padding=True,
                use_audio_in_video=USE_AUDIO_IN_VIDEO
            )
        else:
            # Fallback without process_mm_info
            text = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=False
            )
            inputs = self.processor(
                text=text,
                return_tensors="pt",
                padding=True
            )
        
        # Move inputs to model device
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                  for k, v in inputs.items()}
        
        # Generate response
        print(f"Generating response (return_audio={return_audio})...")
        with torch.no_grad():
            if return_audio:
                # Generate with audio output (requires talker to be enabled)
                text_ids, audio = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    return_audio=True
                )
                # Decode text
                response_text = self.processor.batch_decode(
                    text_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )[0]
                return response_text, audio
            else:
                # Generate text only
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p
                )
                # Decode and return
                response = self.processor.batch_decode(
                    output[:, inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                )[0]
                return response, None

