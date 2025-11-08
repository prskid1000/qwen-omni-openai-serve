import torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

# Note: flash_attention_2 requires the flash-attn package to be installed
# Install with: pip install flash-attn --no-build-isolation
# WARNING: flash_attention_2 conflicts with talker module - talker will be automatically disabled

# Try to import process_mm_info if available
try:
    from qwen_omni_utils import process_mm_info
    HAS_OMNI_UTILS = True
except ImportError:
    HAS_OMNI_UTILS = False
    print("Warning: qwen_omni_utils not found. Install it for full multimodal support.")


def load_model(model_name, use_cpu_offload=False, max_memory=None, use_flash_attention=False):
    """
    Load Qwen2.5-Omni model with proper device handling.
    
    Args:
        model_name: Model identifier (e.g., "Qwen/Qwen2.5-Omni-3B")
        use_cpu_offload: Whether to use CPU offloading (can cause meta tensor issues)
        max_memory: Dict specifying memory per device, e.g. {0: "10GB", "cpu": "30GB"}
        use_flash_attention: Whether to use flash_attention_2 (requires disabling talker)
    """
    print(f"Loading model: {model_name}")
    
    # Prepare model loading kwargs
    model_kwargs = {
        "torch_dtype": torch.float16,
        "device_map": "auto"
    }
    
    # Add flash attention if requested
    if use_flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"
        print("Using flash_attention_2 (talker will be disabled to avoid conflicts)")
    
    # Add CPU offload settings if requested
    if use_cpu_offload and max_memory:
        print(f"Using CPU/GPU split with memory config: {max_memory}")
        model_kwargs["max_memory"] = max_memory
        model_kwargs["offload_folder"] = "offload"
    else:
        print("Loading model on GPU (or CPU if no GPU available)...")
    
    # Load the model
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_name,
        **model_kwargs
    )
    
    # Handle talker: disable by default to avoid meta tensor errors
    # The talker module often ends up in meta state when using device_map="auto"
    # Disable it unless audio output is specifically needed
    talker_disabled = False
    if hasattr(model, 'disable_talker'):
        try:
            model.disable_talker()
            talker_disabled = True
            print("✓ Talker disabled (avoids meta tensor errors, no audio output)")
        except Exception as e:
            print(f"Warning: Could not disable talker: {e}")
    
    # Additional safety: Check if talker is in meta state and patch generate() if needed
    # This prevents the library from trying to use talker during generation
    if hasattr(model, 'talker'):
        try:
            # Check if talker is in meta state
            has_meta = any(p.is_meta for p in model.talker.parameters())
            if has_meta or not talker_disabled:
                print("Warning: Talker may cause issues, patching generate() to handle it safely")
                # Store original generate method
                original_generate = model.generate
                
                # Create a wrapper that safely handles talker calls
                def safe_generate(*args, return_audio=False, **kwargs):
                    # If return_audio is False, ensure talker is not called
                    if not return_audio:
                        # Temporarily replace talker with a dummy to prevent calls
                        talker_backup = getattr(model, 'talker', None)
                        try:
                            # Create a dummy talker that does nothing
                            class DummyTalker:
                                def generate(self, *args, **kwargs):
                                    # Return None to indicate no audio
                                    return None
                                def __getattr__(self, name):
                                    # Return dummy for any attribute access
                                    return lambda *args, **kwargs: None
                            
                            model.talker = DummyTalker()
                            # Call original generate
                            result = original_generate(*args, return_audio=False, **kwargs)
                            return result
                        except Exception as e:
                            # If something goes wrong, try without talker at all
                            if hasattr(model, 'talker'):
                                delattr(model, 'talker')
                            try:
                                result = original_generate(*args, return_audio=False, **kwargs)
                                return result
                            finally:
                                if talker_backup is not None:
                                    model.talker = talker_backup
                        finally:
                            # Restore talker if it was backed up
                            if talker_backup is not None and hasattr(model, 'talker'):
                                model.talker = talker_backup
                    else:
                        # If return_audio=True, use original (but this may fail if talker is in meta state)
                        return original_generate(*args, return_audio=True, **kwargs)
                
                model.generate = safe_generate
        except Exception as e:
            print(f"Warning: Could not patch talker: {e}")
    
    # If you need audio output, uncomment the following (may cause meta tensor errors):
    # if hasattr(model, 'talker'):
    #     try:
    #         # Try to load talker from meta state using to_empty()
    #         has_meta = any(p.is_meta for p in model.talker.parameters())
    #         if has_meta:
    #             # Use to_empty() to properly load from meta state
    #             model.talker = model.talker.to_empty(device="cpu")
    #             # Then load the actual weights
    #             # Note: This requires the talker weights to be available
    #         else:
    #             model.talker = model.talker.to("cpu")
    #         print("✓ Talker enabled and moved to CPU (audio output available)")
    #     except Exception as e:
    #         print(f"Warning: Could not enable talker: {e}")
    
    return model


# Initialize model and processor
model_name = "Qwen/Qwen2.5-Omni-3B"

# Option 1: Load on GPU only (talker disabled by default to avoid meta tensor errors)
model = load_model(model_name, use_cpu_offload=False, use_flash_attention=True)

# Option 1b: Load on GPU with flash attention (faster inference, talker disabled)
# model = load_model(model_name, use_cpu_offload=False, use_flash_attention=True)

# Option 2: Load with CPU/GPU split (uncomment to use, may cause meta tensor issues)
# model = load_model(
#     model_name,
#     use_cpu_offload=True,
#     max_memory={
#         0: "12GB",   # GPU 0
#         "cpu": "8GB"  # CPU
#     },
#     use_flash_attention=False  # Set to True for flash attention (talker will be disabled)
# )

processor = Qwen2_5OmniProcessor.from_pretrained(model_name)

# Print device map
print("\n=== Model Device Map ===")
if hasattr(model, 'hf_device_map'):
    for name, device in list(model.hf_device_map.items())[:10]:
        print(f"{name}: {device}")
    if len(model.hf_device_map) > 10:
        print(f"... ({len(model.hf_device_map)} total modules)")
else:
    print("Device map not available")


def generate_response(
    text_prompt,
    audio_path=None,
    image_path=None,
    video_path=None,
    max_new_tokens=512,
    return_audio=False
):
    """
    Generate response from Qwen 2.5 Omni.
    
    Args:
        text_prompt: Text input/question
        audio_path: Optional path to audio file
        image_path: Optional path to image file
        video_path: Optional path to video file
        max_new_tokens: Maximum tokens to generate
        return_audio: Whether to return audio (requires talker to be enabled)
    
    Returns:
        If return_audio=True: (text_response, audio_tensor)
        If return_audio=False: text_response
    """
    
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
        text = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False
        )
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
        
        inputs = processor(
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
        text = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False
        )
        inputs = processor(
            text=text,
            return_tensors="pt",
            padding=True
        )
    
    # Move inputs to model device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
              for k, v in inputs.items()}
    
    # Generate response
    print(f"Generating response (return_audio={return_audio})...")
    with torch.no_grad():
        if return_audio:
            # Generate with audio output (requires talker to be enabled)
            text_ids, audio = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                return_audio=True
            )
            # Decode text
            response_text = processor.batch_decode(
                text_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            return response_text, audio
        else:
            # Generate text only
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
            # Decode and return
            response = processor.batch_decode(
                output[:, inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )[0]
            return response


# Example usage
if __name__ == "__main__":
    # Text-only example
    print("\n=== Text-only example ===")
    response = generate_response(
        text_prompt="Explain quantum computing in simple terms.",
        return_audio=False
    )
    print(f"Response: {response}")
    
    # Audio example (uncomment and provide path)
    # print("\n=== Audio example ===")
    response = generate_response(
        text_prompt="Transcribe and summarize this audio.",
        audio_path="audio.wav",
        return_audio=False
    )
    print(f"Response: {response}")
    
    # Image example (uncomment and provide path)
    # print("\n=== Image example ===")
    response = generate_response(
        text_prompt="What do you see in this image?",
        image_path="image.png",
        return_audio=False
    )
    print(f"Response: {response}")
    
    # Multimodal example (uncomment and provide paths)
    # print("\n=== Multimodal example ===")
    response = generate_response(
        text_prompt="Compare what's in the image with what you hear in the audio.",
        audio_path="audio.wav",
        image_path="image.png",
        return_audio=False
    )
    print(f"Response: {response}")
    
    # Audio output example (uncomment to generate audio)
    # Note: Requires talker to be enabled (not disabled)
    # import soundfile as sf
    # print("\n=== Audio output example ===")
    # response_text, audio = generate_response(
    #     text_prompt="Say hello in a friendly way.",
    #     return_audio=True
    # )
    # print(f"Response: {response_text}")
    # # Save audio to file
    # sf.write("output.wav", audio.reshape(-1).detach().cpu().numpy(), samplerate=24000)
    # print("Audio saved to output.wav")
