import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import transforms

from config import config
from model import ImageCaptionModel
from tokenizer import TokenizerHF

class PredictionPipeline:
    """
    A pipeline for predicting image captions using a pre-trained image captioning model.
    
    Attributes:
        tokenizer (TokenizerHF): Tokenizer for processing text data.
        max_len (int): Maximum length of generated captions.
        device (str): Device to run the model on ('cpu' or 'cuda').
        model (ImageCaptionModel): Pre-trained model for image captioning.
        transform (transforms.Compose): Transformations applied to input images.
    """
    
    def __init__(self, checkpoint: str, max_len: int, device: str, tokenizer=None):
        """
        Initializes the PredictionPipeline with model checkpoint and configurations.

        Args:
            checkpoint (str): Path to the model checkpoint file.
            max_len (int): Maximum length for the generated captions.
            device (str): Device to run the model on ('cpu' or 'cuda').
            tokenizer (optional): Tokenizer for processing text data. Defaults to None.
        """
        if tokenizer is None:
            # Initialize the tokenizer if not provided
            self.tokenizer = TokenizerHF(
                tokenizer_name="gpt2",
                special_tokens_dict={"bos_token": "[BOS]", "eos_token": "[EOS]", "pad_token": "[PAD]"}
            )
        else:
            self.tokenizer = tokenizer

        # Update model configuration with tokenizer-specific settings
        config['gpt_kwargs']['vocab_size'] = self.tokenizer.vocab_size
        config['gpt_kwargs']['ignore_index'] = self.tokenizer.get_vocab()[self.tokenizer.pad_token]
        self.max_len = max_len
        self.device = device

        # Load the pre-trained model from the checkpoint
        self.model = ImageCaptionModel(config).from_pretrained(checkpoint, device)
        self.model.eval()  # Set model to evaluation mode

        # Define the transformation pipeline for input images
        self.transform = transforms.Compose([
            transforms.Resize(size=(config['img_size'], config['img_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def inference(self, image_path: str) -> str:
        """
        Performs inference on a single image to generate a caption.

        Args:
            image_path (str): Path to the input image file.

        Returns:
            str: Generated caption for the image.
        """
        # Load and transform the image
        image_tensor = self.transform(Image.open(image_path)).unsqueeze(0).to(self.device)
        
        # Generate caption using the model
        tokens = self.model.generate(
            image_tensor,
            sos_token=self.tokenizer.get_vocab()['[BOS]'],
            eos_token=self.tokenizer.get_vocab()['[EOS]'],
            max_len=self.max_len
        )

        # Decode the generated token IDs to a caption string
        return self.tokenizer.decode(token_ids=[token.item() for token in tokens[1:-1]])

    def denormalize(self, image: Image) -> np.ndarray:
        """
        Denormalizes an image for visualization by reversing the normalization transformation.

        Args:
            image (PIL.Image): Input image.

        Returns:
            np.ndarray: Denormalized image array for visualization.
        """
        # Apply transformations to the image
        image = self.transform(image)
        
        # Define mean and standard deviation for denormalization
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        mean = np.array(mean)
        std = np.array(std)

        # Convert tensor to numpy array and denormalize
        image = image.numpy().transpose((1, 2, 0))
        image = std * image + mean
        image = np.clip(image, 0, 1)  # Ensure pixel values are within [0, 1]
        
        return image

    def make_prediction(self, image_paths: list, filename="prediction.png") -> None:
        """
        Generates and visualizes predictions for a list of images.

        Args:
            image_paths (list): List of paths to image files.
            filename (str): Filename for saving the visualization. Defaults to "prediction.png".
        """
        # Initialize a subplot for each image
        fig, ax = plt.subplots(len(image_paths), 1, figsize=(8, 8 * len(image_paths)))

        # Loop through each image path and plot
        for i, image_path in enumerate(image_paths):
            # Load the image
            image = Image.open(image_path)

            # Perform inference to generate a caption
            caption = self.inference(image_path)

            # Denormalize the image for visualization
            denormalized_image = self.denormalize(image)

            # Plot the denormalized image and caption
            ax[i].imshow(denormalized_image)
            ax[i].set_title(f"Caption: {caption}")
            ax[i].axis('off')  # Turn off axis labels

        # Save the entire plot as an image file
        plt.savefig(filename)

        # Display the plot
        plt.show()

if __name__ == '__main__':
    # Define model and pipeline parameters
    checkpoint = "image_caption_model.pt"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    max_len = config['img_size']  # Maximum length of captions

    # List of image paths to process
    image_paths = [
        r"images\1000092795.jpg",
        r"images\1000268201.jpg",
        r"images\1000344755.jpg",
        r"images\1000523639.jpg"
    ]

    # Initialize the prediction pipeline
    predict_pipeline = PredictionPipeline(checkpoint, max_len, device)

    # Make predictions and visualize results
    predict_pipeline.make_prediction(image_paths, "predictions.png")

