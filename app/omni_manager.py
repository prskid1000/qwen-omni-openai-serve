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
        model_name: str = "wolfofbackstreet/Qwen2.5-Omni-3B-4Bit",  # 4-bit quantized model
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
        self.context_length = None
        self.use_talker = False  # Track talker state (like USE_TALKER in omni_bnb.py)
        
    def load_model(self, use_talker: bool = False):
        """Load Qwen2.5-Omni model with proper device handling (using bnb 4-bit quantized model)
        
        Args:
            use_talker: If True, keep talker enabled. If False, disable talker (like USE_TALKER in omni_bnb.py)
        """
        print(f"Loading model: {self.model_name}")
        
        # Prepare model loading kwargs - exactly like omni_bnb.py
        model_kwargs = {
            "trust_remote_code": True,  # Required for quantized models
            "device_map": "auto",  # spreads across your GPU/CPU if needed
            "torch_dtype": torch.bfloat16  # or torch.float16 if your GPU prefers it
        }
        
        # Add flash attention if requested
        if self.use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"
            print("Using flash_attention_2")
        
        # Load the model (bnb 4-bit picked up from repo config) - exactly like omni_bnb.py
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            self.model_name,
            **model_kwargs
        ).to("cuda")
        
        self.model.eval()  # Set to evaluation mode for faster inference
        
        # Get device info (with device_map="auto", parameters may be on different devices)
        model_device = next(self.model.parameters()).device
        print(f"Model loaded with device_map='auto' (primary device: {model_device})")
        print("ℹ️  Parameters may be distributed across devices")
        
        # Handle talker exactly like omni_bnb.py
        self.use_talker = use_talker
        if not use_talker and hasattr(self.model, "disable_talker"):
            try:
                self.model.disable_talker()
                print("✓ Talker disabled")
            except Exception as e:
                print(f"Warning: Could not disable talker: {e}")
        else:
            self.talker_enabled = True
            print("✓ Talker enabled")
        
        # Load processor
        self.processor = Qwen2_5OmniProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        # Get context length from model config
        if hasattr(self.model, 'config'):
            config = self.model.config
            # Try different common attribute names for context length
            if hasattr(config, 'max_position_embeddings'):
                self.context_length = config.max_position_embeddings
            elif hasattr(config, 'max_seq_length'):
                self.context_length = config.max_seq_length
            elif hasattr(config, 'n_positions'):
                self.context_length = config.n_positions
            elif hasattr(config, 'context_length'):
                self.context_length = config.context_length
        
        # Print context length if available
        if self.context_length:
            print(f"📏 Context Length: {self.context_length:,} tokens")
        
        print("✅ Model loaded successfully")
    
    def reload_model_if_needed(self, return_audio: bool):
        """Reload model completely when toggling talker (exactly like omni_bnb.py pattern)"""
        use_talker = return_audio
        
        # If model not loaded or talker state changed, reload completely
        if self.model is None or self.use_talker != use_talker:
            if self.model is not None:
                print(f"🔄 Reloading model (talker: {self.use_talker} -> {use_talker})...")
                # Clear current model
                del self.model
                self.model = None
            if self.processor is not None:
                del self.processor
                self.processor = None
            
            # Reload with correct talker state
            self.load_model(use_talker=use_talker)
            return True  # Reloaded
        
        return False  # No reload needed
        
    def generate_response(
        self,
        text_prompt: str,
        audio_path: Optional[str] = None,
        image_path: Optional[str] = None,
        video_path: Optional[str] = None,
        max_new_tokens: int = 512,
        return_audio: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.9,
        use_audio_in_video: bool = True,
        do_sample: bool = False
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
            temperature: Sampling temperature (used if do_sample=True)
            top_p: Top-p sampling parameter (used if do_sample=True)
            use_audio_in_video: Whether to use audio in video processing
            do_sample: Whether to use sampling (False = greedy decoding, faster)
        
        Returns:
            If return_audio=True: (text_response, audio_tensor)
            If return_audio=False: (text_response, None)
        """
        if not self.model or not self.processor:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Prepare conversation with system prompt
        # Default system prompt encourages English responses
        conversation = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech. Please respond in English unless the user explicitly asks for another language."}
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
            text = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=False
            )
            audios, images, videos = process_mm_info(conversation, use_audio_in_video=use_audio_in_video)
            
            inputs = self.processor(
                text=text,
                audio=audios,
                images=images,
                videos=videos,
                return_tensors="pt",
                padding=True,
                use_audio_in_video=use_audio_in_video
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
        
        # Move all tensors to the same device/dtype as the model
        # Note: input_ids and other integer tensors must stay as Long/Int, not float16
        # Get the device from the model (handles device_map="auto" case)
        model_device = next(self.model.parameters()).device
        for k, v in list(inputs.items()):
            if isinstance(v, torch.Tensor):
                if v.dtype in (torch.long, torch.int, torch.int32, torch.int64):
                    # Integer tensors (like input_ids) should only move to device, keep integer dtype
                    inputs[k] = v.to(model_device)
                else:
                    # Float tensors can use model's dtype (like omni_bnb.py)
                    inputs[k] = v.to(model_device, dtype=self.model.dtype)
        
        # Generate response (exactly like omni_bnb.py)
        print(f"Generating response (return_audio={return_audio})...")
        
        # Get input length to extract only newly generated tokens
        input_length = inputs['input_ids'].shape[1]
        
        with torch.inference_mode():  # Faster inference, disables gradient computation
            if self.use_talker:
                # Generate with audio output (USE_TALKER=True, exactly like omni_bnb.py)
                text_ids, audio = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    use_audio_in_video=use_audio_in_video,
                    do_sample=do_sample,
                    temperature=temperature if do_sample else None,
                    top_p=top_p if do_sample else None
                )
                # Extract only newly generated tokens (skip input prompt)
                generated_ids = text_ids[:, input_length:]
                # Decode only the newly generated text
                response_text = self.processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )[0]
                return response_text.strip(), audio
            else:
                # Text-only generation (USE_TALKER=False, exactly like omni_bnb.py)
                text_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    use_audio_in_video=use_audio_in_video,
                    do_sample=do_sample,
                    temperature=temperature if do_sample else None,
                    top_p=top_p if do_sample else None,
                    return_audio=False  # Disable audio generation (exactly like omni_bnb.py)
                )
                # Extract only newly generated tokens (skip input prompt)
                generated_ids = text_ids[:, input_length:]
                # Decode only the newly generated text
                response = self.processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )[0]
                return response.strip(), None

