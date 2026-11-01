# 4. Experiment 4: Implement Diffusion Models for image generation using tools like HuggingFace Diffusion library.

# Aim: Generate an image from a text prompt.

import torch
from diffusers import StableDiffusionPipeline       # generate images from text.

model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)

image = pipe(
    "A futuristic city at dusk, neon lights, digital art",
    num_inference_steps=20,
    guidance_scale=7.5
).images[0]

image.save("generated_image.png")
print("Image saved as generated_image.png")
