from transformers import CLIPModel, CLIPTokenizer

class CLIPTextProcessor:
    def __init__(self):
        # Load the CLIP model and tokenizer from Hugging Face
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    def encode_text(self, text):
        # Tokenize and encode the text input
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        return self.model.get_text_features(**inputs)
