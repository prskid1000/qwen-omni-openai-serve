# pip install -U transformers accelerate torch soundfile qwen-omni-utils
# On Windows, install ffmpeg (for audio-in-video):
# choco install ffmpeg  # (Admin PowerShell) or grab the static build from ffmpeg.org

import torch, soundfile as sf
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

MODEL_ID = "Qwen/Qwen2.5-Omni-3B"
USE_AUDIO_IN_VIDEO = True

# 1) Load the whole model on a single GPU; avoid sharding
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,     # FP16 fits best on Windows + consumer GPUs
    device_map=None,               # don't auto-shard
    low_cpu_mem_usage=True,        # stream from disk → reduces RAM spikes
)
model = model.to("cuda")           # put ALL params on GPU

processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_ID)

# 2) Prepare a sample video conversation
conversation = [
    {"role": "system", "content": [{"type": "text",
      "text": "You are Qwen, a virtual human ..."}]},
    {"role": "user", "content": [
      {"type": "video",
       "video": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-Omni/draw.mp4"}
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
for k, v in list(inputs.items()):
    if isinstance(v, torch.Tensor):
        inputs[k] = v.to(model.device, dtype=model.dtype)

# 4) Generate (keep token budget modest to save VRAM)
text_ids, audio = model.generate(
    **inputs,
    use_audio_in_video=USE_AUDIO_IN_VIDEO,
    max_new_tokens=192,
    do_sample=True, temperature=0.6,
)

decoded = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
print(decoded[0])

sf.write("output.wav", audio.reshape(-1).detach().cpu().numpy(), samplerate=24000)

# (Optional) sanity check: ensure no leftover CPU/meta tensors
for name, p in model.named_parameters():
    assert p.device.type == "cuda", f"{name} still on {p.device}"
print("✅ All parameters are now on GPU")