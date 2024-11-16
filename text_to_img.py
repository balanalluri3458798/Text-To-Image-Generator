# Execute these commands one by one

### !pip install torch diffusers transformers matplotlib pillow

### pip install torch diffusers transformers matplotlib pillow huggingface_hub 

from pathlib import Path
import torch
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt
from PIL import Image

# Configuration class
class CFG:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42
    generator = torch.Generator(device).manual_seed(seed)
    image_gen_steps = 35
    image_gen_model_id = "stabilityai/stable-diffusion-2"
    image_gen_size = (400, 400)
    image_gen_guidance_scale = 9

# Initialize the Stable Diffusion model
image_gen_model = StableDiffusionPipeline.from_pretrained(
    CFG.image_gen_model_id, torch_dtype=torch.float16,
    revision="fp16"
)
image_gen_model = image_gen_model.to(CFG.device)

# Function to generate an image based on a text prompt
def generate_image(prompt, model):
    image = model(
        prompt, num_inference_steps=CFG.image_gen_steps,
        generator=CFG.generator,
        guidance_scale=CFG.image_gen_guidance_scale
    ).images[0]

    image = image.resize(CFG.image_gen_size)
    return image

# Generate an image using the prompt "astronaut in space"
image = generate_image("a girl crossing the road", image_gen_model)

# Display the generated image
plt.imshow(image)
plt.axis('off')  # Turn off axis numbers and ticks
plt.show()
