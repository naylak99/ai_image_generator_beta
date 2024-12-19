from models.clip_model import CLIPTextProcessor
from models.stable_diffusion_model import StableDiffusionGenerator
import matplotlib.pyplot as plt

def main():
    # Initialize the models
    clip_processor = CLIPTextProcessor()
    sd_generator = StableDiffusionGenerator()

    # User input
    prompt = input("Enter your prompt: ")

    # Encode the prompt with CLIP (optional, for demonstration purposes)
    encoded_prompt = clip_processor.encode_text([prompt])
    print("CLIP Text Encoding:", encoded_prompt)

    # Generate an image using Stable Diffusion
    generated_image = sd_generator.generate_image(prompt)

    # Display the generated image
    plt.imshow(generated_image)
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    main()
