# pip install -U transformers accelerate torch soundfile qwen-omni-utils
# On Windows, install ffmpeg (for audio-in-video):
# choco install ffmpeg  # (Admin PowerShell) or grab the static build from ffmpeg.org

import torch
import time
import soundfile as sf
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

MODEL_ID = "Qwen/Qwen2.5-Omni-3B"
USE_AUDIO_IN_VIDEO = True
USE_TALKER = True  # Set to False to disable audio generation (faster, text-only)

# 1) Load the whole model on a single GPU; avoid sharding
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,     # FP16 fits best on Windows + consumer GPUs
    device_map=None,               # don't auto-shard
    low_cpu_mem_usage=True, 
    attn_implementation="flash_attention_2" # stream from disk → reduces RAM spikes
)
model = model.to("cuda")           # put ALL params on GPU
model.eval()                       # Set to evaluation mode for faster inference

processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_ID)

# 2) Prepare a sample video conversation
# Use the default system prompt for audio output to work properly
conversation = [
    {"role": "system", "content": [{"type": "text",
      "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}]},
    {"role": "user", "content": [
      {"type": "text",
       "text": "Hello, how are you?"}
    ]},
]

# 3) Pack multimodal inputs
text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
inputs = processor(
    text=text, audio=audios, images=images, videos=videos,
    return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO,
)

# Move all tensors to the same device/dtype as the model
# Note: input_ids and other integer tensors must stay as Long/Int, not float16
for k, v in list(inputs.items()):
    if isinstance(v, torch.Tensor):
        if v.dtype in (torch.long, torch.int, torch.int32, torch.int64):
            # Integer tensors (like input_ids) should only move to device, keep integer dtype
            inputs[k] = v.to(model.device)
        else:
            # Float tensors can use model's dtype
            inputs[k] = v.to(model.device, dtype=model.dtype)

# 4) Generate (keep token budget modest to save VRAM)
start_time = time.time()
with torch.inference_mode():  # Faster inference, disables gradient computation
    if USE_TALKER:
        text_ids, audio = model.generate(
            **inputs,
            use_audio_in_video=USE_AUDIO_IN_VIDEO,
            do_sample=False,  # Greedy decoding: faster, more deterministic
            # do_sample=True, temperature=0.6,  # For more creative/diverse output
        )
    else:
        # Disable and remove talker from GPU memory (saves VRAM)
        if hasattr(model, 'disable_talker'):
            try:
                model.disable_talker()
                print("✓ Talker disabled")
            except Exception as e:
                print(f"Warning: Could not disable talker: {e}")
        # Text-only generation (talker disabled)
        text_ids = model.generate(
            **inputs,
            use_audio_in_video=USE_AUDIO_IN_VIDEO,
            do_sample=False,
            return_audio=False,  # Disable audio generation
        )
        audio = None

decoded = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
print(decoded[0])

elapsed = time.time() - start_time
print(f"✅ Generation complete! (took {elapsed:.1f} seconds)")

if USE_TALKER and audio is not None:
    sf.write("output.wav", audio.reshape(-1).detach().cpu().numpy(), samplerate=24000)
    print("✅ Audio saved to output.wav")
else:
    print("ℹ️  Audio generation disabled (USE_TALKER=False)")

# (Optional) sanity check: ensure no leftover CPU/meta tensors (except talker if disabled)
for name, p in model.named_parameters():
    if not USE_TALKER and 'talker' in name:
        continue  # Skip talker parameters if disabled
    assert p.device.type == "cuda", f"{name} still on {p.device}"
print("✅ All parameters are now on GPU" + (" (talker removed)" if not USE_TALKER else ""))