from diffusers import StableDiffusionPipeline
import torch

class StableDiffusionGenerator:
    def __init__(self, model_name="CompVis/stable-diffusion-v1-4"):
        # Load the Stable Diffusion pipeline
        self.pipe = StableDiffusionPipeline.from_pretrained(model_name)
        self.pipe = self.pipe.to("cuda" if torch.cuda.is_available() else "cpu")

    def generate_image(self, prompt, num_inference_steps=50):
        # Generate an image based on the prompt
        return self.pipe(prompt, num_inference_steps=num_inference_steps).images[0]
