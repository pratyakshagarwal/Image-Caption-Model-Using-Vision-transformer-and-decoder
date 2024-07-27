import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import transforms

from tokenizer import TokenizerHF
from model import ImageCaptionModel
from config import config

# Streamlit app for deploying an image captioning model
class PredictionPipeline:
    def __init__(self, checkpoint, config, max_len, device, tokenizer=None):
        """
        Initialize the PredictionPipeline class.

        Args:
            checkpoint (str): Path to the model checkpoint file.
            config (dict): Configuration dictionary for the model.
            max_len (int): Maximum length for generated captions.
            device (torch.device): Device to run the model on (CPU or GPU).
            tokenizer (TokenizerHF, optional): Pre-initialized tokenizer. Defaults to None.
        """
        # Initialize the tokenizer if not provided
        if tokenizer is None:
            self.tokenizer = TokenizerHF(
                tokenizer_name="gpt2",
                special_tokens_dict={
                    "bos_token": "[BOS]", 
                    "eos_token": "[EOS]", 
                    "pad_token": "[PAD]"
                }
            )
        else:
            self.tokenizer = tokenizer

        # Update configuration with tokenizer information
        config['gpt_kwargs']['vocab_size'] = self.tokenizer.vocab_size
        config['gpt_kwargs']['ignore_index'] = self.tokenizer.get_vocab()[self.tokenizer.pad_token]
        self.max_len = max_len
        self.device = device

        # Load the model from the checkpoint and set it to evaluation mode
        self.model = ImageCaptionModel(config).from_pretrained(checkpoint, device)
        self.model.eval()

        # Define image transformations: resize, convert to tensor, and normalize
        self.transform = transforms.Compose([
            transforms.Resize(size=(config['img_size'], config['img_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def inference(self, image) -> str:
        """
        Generate a caption for the given image.

        Args:
            image (PIL.Image): Input image.

        Returns:
            str: Generated caption.
        """
        # Transform the image and add a batch dimension
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Generate tokens using the model
        tokens = self.model.generate(
            image_tensor,
            sos_token=self.tokenizer.get_vocab()['[BOS]'],
            eos_token=self.tokenizer.get_vocab()['[EOS]'],
            max_len=self.max_len
        )

        # Decode tokens into a caption and return it
        return self.tokenizer.decode(token_ids=[token.item() for token in tokens[1:-1]])

    def denormalize(self, image):
        """
        Denormalize the image for visualization.

        Args:
            image (PIL.Image): Input image.

        Returns:
            np.ndarray: Denormalized image.
        """
        # Apply the same transformations as in inference
        image = self.transform(image)

        # Calculate mean and std deviation for normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        # Convert the image to a numpy array and denormalize
        image = image.numpy().transpose((1, 2, 0))
        image = std * image + mean

        # Clip values to ensure they fall within [0, 1]
        image = np.clip(image, 0, 1)
        return image

def main(checkpoint, config, max_len, device):
    """
    Main function to run the Streamlit app.

    Args:
        checkpoint (str): Path to the model checkpoint file.
        config (dict): Configuration dictionary for the model.
        max_len (int): Maximum length for generated captions.
        device (torch.device): Device to run the model on (CPU or GPU).
    """
    # Set the title and description of the Streamlit app
    st.title("Image Captioning Model Deployment")
    st.write("Upload an image to generate a caption using the image captioning model.")

    # File uploader widget to allow users to upload an image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Load and display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Initialize the prediction pipeline
        pipeline = PredictionPipeline(checkpoint, config, max_len=max_len, device=device)

        # Perform inference to generate a caption and display it
        caption = pipeline.inference(image)
        st.write("**Generated Caption:**")
        st.write(caption)

        # Denormalize the image for visualization
        denormalized_image = pipeline.denormalize(image)
        
        # Plot the denormalized image with the generated caption as the title
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(denormalized_image)
        ax.set_title(f"Caption: {caption}")
        ax.axis('off')
        
        # Display the plot in the Streamlit app
        st.pyplot(fig)

if __name__ == "__main__":
    # Path to the model checkpoint
    checkpoint = "image_caption_model.pt"
    
    # Determine the device to run the model on (GPU if available, otherwise CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Maximum length of generated captions
    max_len = config['gpt_kwargs']['context_length']

    # Run the Streamlit app
    main(checkpoint=checkpoint,
         config=config,
         max_len=max_len,
         device=device)
