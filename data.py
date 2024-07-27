import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import Tuple
import torch
from PIL import Image
import random

def preprocess_data(csv_file_path: str, image_folder: str) -> None:
    """
    Preprocesses the data from a CSV file and saves it as a new CSV.

    This function reads the CSV file containing image paths and comments, removes duplicates, 
    formats the data, and adds special tokens for text processing. The processed data is saved to a new CSV file.

    Args:
        csv_file_path (str): Path to the input CSV file.
        image_folder (str): Path to the folder containing images.
    """
    data = pd.read_csv(csv_file_path, delimiter="|")
    data.drop_duplicates(subset=['image_name'], inplace=True)  # Remove duplicate image entries
    data.drop(columns=' comment_number', axis=1, inplace=True)  # Drop unnecessary column
    data.reset_index(drop=True, inplace=True)  # Reset index after dropping duplicates
    data.rename({" comment": "comment"}, axis=1, inplace=True)  # Rename column for consistency
    data.iloc[:, 0] = image_folder + "/" + data.iloc[:, 0]  # Prepend folder path to image names
    data['comment'] = '[BOS] ' + data['comment'] + ' [EOS]'  # Add special tokens for the comment
    data.to_csv("preprocessed_data.csv", index=False)  # Save preprocessed data

def prepare_data(train_config: dict, model_config: dict, data: pd.DataFrame, tokenizer) -> Tuple[DataLoader, DataLoader]:
    """
    Prepares the DataLoader objects for training and testing datasets.

    This function splits the data into training and testing sets, creates custom datasets, 
    and returns DataLoaders for both.

    Args:
        train_config (dict): Configuration parameters for training.
        model_config (dict): Configuration parameters for the model.
        data (pd.DataFrame): DataFrame containing the preprocessed data.
        tokenizer: Tokenizer for processing text.

    Returns:
        Tuple[DataLoader, DataLoader]: DataLoaders for the training and testing datasets.
    """
    idxs = set(range(data.shape[0]))
    
    # Randomly split indices for training and testing
    train_idxs = random.sample(sorted(idxs), k=int(len(idxs) * train_config['train_size']))
    test_idxs = list(idxs.difference(set(train_idxs)))

    # Split data into training and testing sets
    train_data = data.copy(deep=True).iloc[train_idxs, :].reset_index(drop=True)
    test_data = data.copy(deep=True).iloc[test_idxs, :].reset_index(drop=True)

    # Create dataset objects for training and testing
    train_dataset = ImageCaptionDataset(
        dataframe=train_data,
        image_size=model_config['img_size'],
        context_length=model_config['gpt_kwargs']['context_length'],
        tokenizer=tokenizer
    )

    test_dataset = ImageCaptionDataset(
        dataframe=test_data,
        image_size=model_config['img_size'],
        context_length=model_config['gpt_kwargs']['context_length'],
        tokenizer=tokenizer
    )

    # Create DataLoader objects for training and testing datasets
    train_dl = DataLoader(
        dataset=train_dataset,
        batch_size=train_config['batch_size'],
        shuffle=True,
        num_workers=2
    )

    test_dl = DataLoader(
        dataset=test_dataset,
        batch_size=train_config['batch_size'],
        shuffle=False
    )

    return train_dl, test_dl

class ImageCaptionDataset(Dataset):
    """
    Custom Dataset class for loading images and captions for image captioning.

    Attributes:
        dataframe (pd.DataFrame): DataFrame containing image paths and captions.
        image_size (int): Desired size for image resizing.
        context_length (int): Maximum context length for tokenization.
        tokenizer: Tokenizer for processing captions.
        transform (transforms.Compose): Transformations applied to images.
    """
    
    def __init__(self, dataframe: pd.DataFrame, image_size: int, context_length: int, tokenizer) -> None:
        """
        Initializes the ImageCaptionDataset with data and parameters.

        Args:
            dataframe (pd.DataFrame): DataFrame containing image paths and captions.
            image_size (int): Size to which images are resized.
            context_length (int): Maximum length for tokenization.
            tokenizer: Tokenizer used for processing text data.
        """
        assert dataframe.columns[0] == 'image_name', ValueError("The first column should be the path to the image")
        assert dataframe.columns[1] == "comment", ValueError("The second column should be named 'comment'")

        self.tokenizer = tokenizer
        self.context_length = context_length
        self.df = dataframe
        
        # Transformation pipeline for images
        self.transform = transforms.Compose([
            transforms.Resize(size=(image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return self.df.shape[0]
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns a single sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Transformed image, tokenized caption, and attention mask.
        """
        image, text = Image.open(self.df.iloc[idx, 0]), self.df.iloc[idx, 1]
        image_tensor = self.transform(image)  # Apply transformations to the image
        op = self.tokenizer(text, max_len=self.context_length + 1)  # Tokenize the caption
        tokens, attention_mask = op['input_ids'].squeeze(), op['attention_mask'].squeeze()
        return image_tensor, tokens, attention_mask

if __name__ == '__main__':
    # Define paths to the CSV file and image folder
    csv_file_path = r"results.csv"
    image_folder = r"/flickr-image-dataset/flickr30k_images"
    
    # Preprocess the data and save it to a new CSV file
    preprocess_data(csv_file_path, image_folder)