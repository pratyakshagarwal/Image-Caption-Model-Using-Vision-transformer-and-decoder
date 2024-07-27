import torch
import math
import os
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List
import timm

class PatchEmbeddings(nn.Module):
    """
    Module to generate patch embeddings from input images using convolutional layers.
    
    Attributes:
        conv_patch_layer (nn.Conv2d): Convolutional layer to extract patches from images.
        flatten (nn.Flatten): Flatten layer to reshape patches for further processing.
    """
    def __init__(self, config):
        """
        Initializes PatchEmbeddings with convolutional layer and flattening.
        
        Args:
            config (dict): Configuration dictionary containing parameters for the patch embeddings.
        """
        super().__init__()

        # Convolutional layer to create patch embeddings
        self.conv_patch_layer = nn.Conv2d(
            in_channels=config['channels'],
            out_channels=config['d_model'],
            kernel_size=config['patch_size'],
            stride=config['patch_size']
        )

        # Flatten patches into a 2D tensor for further processing
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the PatchEmbeddings layer.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Flattened patch embeddings tensor of shape (B, num_patches, d_model).
        """
        # Apply convolution to extract patch embeddings
        patched_tensor = self.conv_patch_layer(x)
        
        # Flatten the patched tensor
        flattend_tensor = self.flatten(patched_tensor)
        
        # Permute dimensions to match (B, num_patches, d_model) format
        return flattend_tensor.permute(0, 2, 1)


class ViTEmbedding(nn.Module):
    """
    Module to generate embeddings including positional and class tokens for Vision Transformer.

    Attributes:
        patch_embeddings (PatchEmbeddings): Patch embeddings module.
        class_token_embedding (nn.Parameter): Learnable class token.
        positional_embedding (nn.Parameter): Learnable positional embeddings.
        dropout (nn.Dropout): Dropout layer for regularization.
    """
    def __init__(self, config):
        """
        Initializes ViTEmbedding with patch embeddings, class token, positional embedding, and dropout.
        
        Args:
            config (dict): Configuration dictionary containing parameters for embeddings.
        """
        super().__init__()

        self.patch_embeddings = PatchEmbeddings(config)

        # Learnable class token to be prepended to patch embeddings
        self.class_token_embedding = nn.Parameter(
            data=torch.randn(size=(1, 1, config['d_model'])),
            requires_grad=True
        )

        # Learnable positional embeddings for each patch plus class token
        self.positional_embedding = nn.Parameter(
            data=torch.randn(size=(1, config['num_patches'] + 1, config['d_model'])),
            requires_grad=True
        )
        
        self.dropout = nn.Dropout(config['emb_dropout'])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the ViTEmbedding layer.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Embeddings with positional and class tokens of shape (B, num_patches + 1, d_model).
        """
        # Extract patch embeddings
        patch_embed = self.patch_embeddings(x)
        
        # Concatenate class token with patch embeddings
        patch_embeddings_with_class_token = torch.cat(
            tensors=(self.class_token_embedding.repeat(patch_embed.shape[0], 1, 1), patch_embed),
            dim=1
        )
        
        # Add positional embeddings and apply dropout
        return self.dropout(patch_embeddings_with_class_token + self.positional_embedding)


class MSABlock(nn.Module):
    """
    Multihead Self-Attention block used in the transformer architecture.

    Attributes:
        attn_block (nn.MultiheadAttention): Multihead attention layer.
        layer_norm (nn.LayerNorm): Layer normalization for the attention output.
    """
    def __init__(self, config) -> None:
        """
        Initializes the MSABlock with multihead attention and layer normalization.

        Args:
            config (dict): Configuration dictionary containing parameters for attention and normalization.
        """
        super().__init__()

        # Multihead self-attention layer
        self.attn_block = nn.MultiheadAttention(
            embed_dim=config["d_model"],
            num_heads=config["num_heads"],
            batch_first=True,
            dropout=config['attn_dropout']
        )
        
        # Layer normalization for attention output
        self.layer_norm = nn.LayerNorm(normalized_shape=config["d_model"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MSABlock layer.

        Args:
            x (torch.Tensor): Input tensor of shape (B, num_patches + 1, d_model).

        Returns:
            torch.Tensor: Output tensor with attention applied, shape (B, num_patches + 1, d_model).
        """
        # Apply multihead self-attention
        attn_output, _ = self.attn_block(x, x, x)
        
        # Add & normalize
        return self.layer_norm(x + attn_output)


class MLPBlock(nn.Module):
    """
    Feed-Forward Network (FFN) block of the transformer architecture.

    Attributes:
        dense_net (nn.Sequential): Sequential network consisting of dense layers and activation functions.
        layer_norm (nn.LayerNorm): Layer normalization for the FFN output.
    """
    def __init__(self, config) -> None:
        """
        Initializes the MLPBlock with dense layers, GELU activation, dropout, and layer normalization.

        Args:
            config (dict): Configuration dictionary containing parameters for the FFN.
        """
        super().__init__()
        d_model = config["d_model"]

        # Sequential network for the feed-forward operation
        self.dense_net = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(p=config['mlp_dropout']),
            nn.Linear(d_model * 4, d_model)
        )
        
        # Layer normalization for FFN output
        self.layer_norm = nn.LayerNorm(normalized_shape=d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLPBlock layer.

        Args:
            x (torch.Tensor): Input tensor of shape (B, num_patches + 1, d_model).

        Returns:
            torch.Tensor: Output tensor with FFN applied, shape (B, num_patches + 1, d_model).
        """
        # Apply feed-forward network and add & normalize
        return self.layer_norm(x + self.dense_net(x))


class EncoderBlock(nn.Module):
    """
    Transformer encoder block which combines Multihead Self-Attention (MSA) and Feed-Forward Network (FFN) blocks.

    Attributes:
        msa_block (MSABlock): Multihead Self-Attention block.
        mlp_block (MLPBlock): Feed-Forward Network block.
    """
    def __init__(self, config) -> None:
        """
        Initializes the EncoderBlock with MSA and FFN blocks.

        Args:
            config (dict): Configuration dictionary containing parameters for the encoder block.
        """
        super().__init__()
        self.msa_block = MSABlock(config)
        self.mlp_block = MLPBlock(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the EncoderBlock layer.

        Args:
            x (torch.Tensor): Input tensor of shape (B, num_patches + 1, d_model).

        Returns:
            torch.Tensor: Output tensor with encoder operations applied, shape (B, num_patches + 1, d_model).
        """
        # Apply MSA and FFN blocks sequentially
        return self.mlp_block(self.msa_block(x))


class Encoder(nn.Module):
    """
    The encoder part of the Vision Transformer (ViT) consisting of multiple encoder blocks.

    Attributes:
        blocks (nn.ModuleList): List of EncoderBlock modules.
    """
    def __init__(self, config) -> None:
        """
        Initializes the Encoder with a specified number of encoder blocks.

        Args:
            config (dict): Configuration dictionary containing the number of encoder blocks.
        """
        super().__init__()
        self.blocks = nn.ModuleList([EncoderBlock(config) for _ in range(config["num_encoders"])])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Encoder layer.

        Args:
            x (torch.Tensor): Input tensor of shape (B, num_patches + 1, d_model).

        Returns:
            torch.Tensor: Output tensor after applying all encoder blocks, shape (B, num_patches + 1, d_model).
        """
        # Apply each encoder block sequentially
        for block in self.blocks:
            x = block(x)
        
        return x


class ViT(nn.Module):
    """
    Vision Transformer (ViT) model which combines embedding and encoder to produce image representations.

    Attributes:
        embedding_layer (ViTEmbedding): Embedding layer that adds positional and class tokens.
        encoder (Encoder): Encoder containing multiple encoder blocks.
    """
    def __init__(self, config) -> None:
        """
        Initializes the ViT model with embedding and encoder layers.

        Args:
            config (dict): Configuration dictionary containing parameters for embedding and encoder layers.
        """
        super().__init__()
        
        self.embedding_layer = ViTEmbedding(config)
        self.encoder = Encoder(config)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Vision Transformer.

        Args:
            images (torch.Tensor): Input image tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Encoded representation of images using the [CLS] token, shape (B, d_model).
        """
        # Generate embeddings for input images
        embeddings = self.embedding_layer(images)  # Shape: (B, num_patches + 1, d_model)
        
        # Apply encoder blocks
        encoded_vectors = self.encoder(embeddings)  # Shape: (B, num_patches + 1, d_model)
        
        # Return the representation of the [CLS] token
        return encoded_vectors[:, 0, :]

    
############################################################################################################################################################33

class GPTEmbedding(nn.Module):
    """
    Embedding class for the GPT decoder. This class creates token embeddings and adds positional encodings.

    Attributes:
        token_embedding (nn.Embedding): Embedding layer for tokens.
        positional_encoding (nn.Parameter): Learnable positional encodings.
        dropout (nn.Dropout): Dropout layer for regularization.
    """
    def __init__(self, config) -> None:
        """
        Initializes GPTEmbedding with token embeddings, positional encodings, and dropout.

        Args:
            config (dict): Configuration dictionary containing parameters for embeddings.
        """
        super().__init__()
        self.token_embedding = nn.Embedding(
            num_embeddings=config["vocab_size"],
            embedding_dim=config["d_model"]
        )
        
        self.positional_encoding = nn.Parameter(
            data=torch.randn(size=(1, config["context_length"], config["d_model"])),
            requires_grad=True
        )
        self.dropout = nn.Dropout(p=config['emb_dropout'])
    
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the GPTEmbedding layer.

        Args:
            tokens (torch.Tensor): Input token tensor of shape (B, CTX_LENGTH).

        Returns:
            torch.Tensor: Embeddings with positional encodings applied, shape (B, CTX_LENGTH, d_model).
        """
        token_embeddings = self.token_embedding(tokens)
        return self.dropout(self.positional_encoding[:, :tokens.shape[1], :] + token_embeddings)


class CausalSelfAttnBlock(nn.Module):
    """
    Causal self-attention block for the GPT model. This block performs masked multi-head self-attention.

    Attributes:
        d_model (int): Dimension of the model.
        head_dim (int): Dimension of each attention head.
        num_heads (int): Number of attention heads.
        softmax_eps (float): Epsilon for numerical stability in softmax.
        projection_layer (nn.Linear): Linear projection layer for Q, K, V.
        out_layer (nn.Linear): Linear layer for output.
        layer_norm (nn.LayerNorm): Layer normalization.
        attn_dropout (nn.Dropout): Dropout layer for attention weights.
    """
    def __init__(self, config) -> None:
        """
        Initializes CausalSelfAttnBlock with projection layers, output layer, layer normalization, and dropout.

        Args:
            config (dict): Configuration dictionary containing parameters for the attention block.
        """
        super().__init__()
        assert config["d_model"] % config["num_heads"] == 0, \
            ValueError(f"{config['d_model']} d_model should be exactly divisible by {config['num_heads']} num_heads")
        
        self.d_model = config["d_model"]
        self.head_dim = config["d_model"] // config["num_heads"]
        self.num_heads = config["num_heads"]
        self.softmax_eps = config["softmax_eps"]
        
        self.projection_layer = nn.Linear(self.d_model, self.d_model * 3)
        self.out_layer = nn.Linear(self.d_model, self.d_model)
        self.layer_norm = nn.LayerNorm(normalized_shape=self.d_model)
        self.attn_dropout = nn.Dropout(p=config['attn_dropout'])

    def _safe_softmax(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes numerically stable softmax.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Softmax probabilities.
        """
        num = torch.exp(x)
        denom = torch.exp(x).sum(dim=-1, keepdim=True) + self.softmax_eps
        return num / denom
    
    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the CausalSelfAttnBlock.

        Args:
            x (torch.Tensor): Input tensor of shape (B, CTX_LENGTH, d_model).
            attn_mask (torch.Tensor): Attention mask tensor of shape (B, CTX_LENGTH, CTX_LENGTH).

        Returns:
            torch.Tensor: Output tensor with attention applied, shape (B, CTX_LENGTH, d_model).
        """
        B, CTX_LENGTH = x.shape[0], x.shape[1]
        q, k, v = self.projection_layer(x).split(self.d_model, dim=2)  # B, CTX_LENGTH, d_model
        q = q.view(B, CTX_LENGTH, self.num_heads, self.head_dim).transpose(1, 2)  # B, num_heads, CTX_LENGTH, head_dim
        k = k.view(B, CTX_LENGTH, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, CTX_LENGTH, self.num_heads, self.head_dim).transpose(1, 2)
        
        q_k_prod = (q @ k.transpose(2, 3)) + attn_mask.unsqueeze(1)  # B, num_heads, CTX_LENGTH, CTX_LENGTH
        wts = self._safe_softmax(q_k_prod / math.sqrt(self.head_dim))  # B, num_heads, CTX_LENGTH, CTX_LENGTH
        wts = self.attn_dropout(wts)
        attn_outputs = wts @ v  # B, num_heads, CTX_LENGTH, head_dim
        y = attn_outputs.transpose(1, 2).contiguous().view(B, CTX_LENGTH, -1)
        return self.layer_norm(x + self.out_layer(y))


class CrossAttnBlock(nn.Module):
    """
    Cross-attention block for the GPT model. This block performs attention between token embeddings and image encodings.

    Attributes:
        d_model (int): Dimension of the model.
        num_heads (int): Number of attention heads.
        head_dim (int): Dimension of each attention head.
        q_proj (nn.Linear): Linear projection layer for queries.
        k_proj (nn.Linear): Linear projection layer for keys.
        v_proj (nn.Linear): Linear projection layer for values.
        projection_layer (nn.Linear): Linear layer for output projection.
        layer_norm (nn.LayerNorm): Layer normalization.
        attn_dropout (nn.Dropout): Dropout layer for attention weights.
    """
    def __init__(self, config) -> None:
        """
        Initializes CrossAttnBlock with projection layers, output layer, layer normalization, and dropout.

        Args:
            config (dict): Configuration dictionary containing parameters for the cross-attention block.
        """
        super().__init__()
        assert config["d_model"] % config["num_heads"] == 0, \
            ValueError(f"{config['d_model']} d_model must be divisible by {config['num_heads']} num_heads")

        self.d_model = config['d_model']
        self.num_heads = config['num_heads']
        self.head_dim = self.d_model // self.num_heads
        self.q_proj = nn.Linear(self.d_model, self.d_model)
        self.k_proj = nn.Linear(self.d_model, self.d_model)
        self.v_proj = nn.Linear(self.d_model, self.d_model)
        self.projection_layer = nn.Linear(self.d_model, self.d_model)
        self.layer_norm = nn.LayerNorm(normalized_shape=self.d_model)
        self.attn_dropout = nn.Dropout(p=config['attn_dropout'])
    
    def forward(self, x: torch.Tensor, image_encoding: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the CrossAttnBlock.

        Args:
            x (torch.Tensor): Input tensor of shape (B, CTX_LENGTH, d_model).
            image_encoding (torch.Tensor): Image encoding tensor of shape (B, 1, d_model).

        Returns:
            torch.Tensor: Output tensor with cross-attention applied, shape (B, CTX_LENGTH, d_model).
        """
        B, CTX_LENGTH, _ = x.shape        

        q = self.q_proj(x).view(B, CTX_LENGTH, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # B, num_heads, CTX_LENGTH, head_dim
        k = self.k_proj(image_encoding).view(B, 1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # B, num_heads, 1, head_dim
        v = self.v_proj(image_encoding).view(B, 1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # B, num_heads, 1, head_dim

        wts = F.softmax((q @ k.transpose(2, 3)) / math.sqrt(self.head_dim), dim=-1)  # B, num_heads, CTX_LENGTH, 1
        wts = self.attn_dropout(wts)
        y = wts @ v  # B, num_heads, CTX_LENGTH, head_dim
        y = y.transpose(1, 2).contiguous().view(B, CTX_LENGTH, -1)  # B, CTX_LENGTH, d_model
        return self.layer_norm(x + self.projection_layer(y))


class MLPBlock(nn.Module):
    """
    Feed-Forward Network (FFN) block used in the transformer architecture.

    Attributes:
        dense_net (nn.Sequential): Sequential network consisting of dense layers and activation functions.
        layer_norm (nn.LayerNorm): Layer normalization.
    """
    def __init__(self, config) -> None:
        """
        Initializes MLPBlock with dense layers and layer normalization.

        Args:
            config (dict): Configuration dictionary containing parameters for the MLP block.
        """
        super().__init__()
        d_model = config["d_model"]
        self.dense_net = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(p=config['mlp_dropout']),
            nn.Linear(d_model * 4, d_model)
        )
        
        self.layer_norm = nn.LayerNorm(normalized_shape=d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLPBlock.

        Args:
            x (torch.Tensor): Input tensor of shape (B, CTX_LENGTH, d_model).

        Returns:
            torch.Tensor: Output tensor with feed-forward network applied, shape (B, CTX_LENGTH, d_model).
        """
        return self.layer_norm(x + self.dense_net(x))


class GPTDecoderBlock(nn.Module):
    """
    The GPT decoder block combines causal self-attention, cross-attention, and feed-forward network blocks.

    Attributes:
        csa_block (CausalSelfAttnBlock): Causal self-attention block.
        cross_attn_block (CrossAttnBlock): Cross-attention block.
        mlp_block (MLPBlock): Feed-forward network block.
    """
    def __init__(self, config) -> None:
        """
        Initializes GPTDecoderBlock with causal self-attention, cross-attention, and feed-forward network blocks.

        Args:
            config (dict): Configuration dictionary containing parameters for the decoder block.
        """
        super().__init__()
        self.csa_block = CausalSelfAttnBlock(config)
        self.cross_attn_block = CrossAttnBlock(config)
        self.mlp_block = MLPBlock(config)
    
    def forward(self, x: torch.Tensor, image_encoding: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the GPTDecoderBlock.

        Args:
            x (torch.Tensor): Input tensor of shape (B, CTX_LENGTH, d_model).
            image_encoding (torch.Tensor): Image encoding tensor of shape (B, 1, d_model).
            attn_mask (torch.Tensor): Attention mask tensor of shape (B, CTX_LENGTH, CTX_LENGTH).

        Returns:
            torch.Tensor: Output tensor after passing through the decoder block, shape (B, CTX_LENGTH, d_model).
        """
        csa_out = self.csa_block(x, attn_mask)
        cross_out = self.cross_attn_block(csa_out, image_encoding)
        mlp_out = self.mlp_block(cross_out)
        return mlp_out
    

class GPTDecoder(nn.Module):
    """
    The GPT decoder consisting of multiple decoder blocks.

    Attributes:
        decoder_blocks (nn.ModuleList): List of GPTDecoderBlock instances.
    """
    def __init__(self, config) -> None:
        """
        Initializes GPTDecoder with a stack of GPTDecoderBlock instances.

        Args:
            config (dict): Configuration dictionary containing parameters for the decoder.
        """
        super().__init__()
        self.decoder_blocks = nn.ModuleList([GPTDecoderBlock(config) for _ in range(config["num_decoders"])])
    
    def forward(self, x: torch.Tensor, image_encoding: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the GPTDecoder.

        Args:
            x (torch.Tensor): Input tensor of shape (B, CTX_LENGTH, d_model).
            image_encoding (torch.Tensor): Image encoding tensor of shape (B, 1, d_model).
            attn_mask (torch.Tensor): Attention mask tensor of shape (B, CTX_LENGTH, CTX_LENGTH).

        Returns:
            torch.Tensor: Output tensor after passing through all decoder blocks, shape (B, CTX_LENGTH, d_model).
        """
        for block in self.decoder_blocks:
            x = block(x, image_encoding, attn_mask)
        
        return x
    

class GPT(nn.Module):
    """
    GPT model for caption generation, integrating embeddings, decoder, and output classification.

    Attributes:
        device (str): Device to run the model on.
        context_length (int): Length of the input context.
        softmax_eps (float): Epsilon for numerical stability in softmax.
        embedding (GPTEmbedding): GPTEmbedding instance.
        decoder (GPTDecoder): GPTDecoder instance.
        cls_head (nn.Linear): Linear layer for final classification.
        ignore_index (int): Index to ignore in loss computation.
    """
    def __init__(self, config) -> None:
        """
        Initializes GPT with embedding, decoder, and classification head.

        Args:
            config (dict): Configuration dictionary containing parameters for the GPT model.
        """
        super().__init__()
        self.device = config["device"]
        self.context_length = config["context_length"]
        self.softmax_eps = config["softmax_eps"]
        self.embedding = GPTEmbedding(config)
        self.decoder = GPTDecoder(config)
        self.cls_head = nn.Linear(config["d_model"], config["vocab_size"])
        self.cls_head.weight = self.embedding.token_embedding.weight
        # Removed weight tying as it led to slower convergence
        self.ignore_index = config["ignore_index"]
    
    def _create_mask(self, context_length: int, attn_mask: torch.Tensor) -> torch.Tensor:
        """
        Creates an attention mask for the decoder.

        Args:
            context_length (int): Length of the context.
            attn_mask (torch.Tensor): Attention mask tensor of shape (B, CTX_LENGTH).

        Returns:
            torch.Tensor: Mask tensor for attention, shape (B, CTX_LENGTH, CTX_LENGTH).
        """
        mask = torch.triu(
            input=torch.ones(size=(context_length, context_length), requires_grad=False) * float("-inf"),
            diagonal=1
        ).unsqueeze(0).repeat(attn_mask.shape[0], 1, 1)
        mask = mask.to(self.device)
        for i in range(mask.shape[0]):
            mask[i, attn_mask[i].logical_not(), :] = float("-inf")
        return mask  # B, CTX_LENGTH, CTX_LENGTH
        
    def forward(self, tokens: torch.Tensor, image_encoding: torch.Tensor, attn_mask: torch.Tensor, targets: torch.Tensor = None) -> Tuple[torch.Tensor]:
        """
        Forward pass through the GPT model.

        Args:
            tokens (torch.Tensor): Input token tensor of shape (B, CTX_LENGTH).
            image_encoding (torch.Tensor): Image encoding tensor of shape (B, 1, d_model).
            attn_mask (torch.Tensor): Attention mask tensor of shape (B, CTX_LENGTH).
            targets (torch.Tensor, optional): Target tensor for computing loss, shape (B, CTX_LENGTH).

        Returns:
            Tuple[torch.Tensor]: Tuple containing logits (B, CTX_LENGTH, vocab_size) and loss (if targets are provided).
        """
        embeddings = self.embedding(tokens)  # B, CTX_LENGTH, d_model
        mask = self._create_mask(tokens.shape[1], attn_mask)
        decoder_out = self.decoder(embeddings, image_encoding, mask)  # B, CTX_LENGTH, d_model
        logits = self.cls_head(decoder_out)  # B, CTX_LENGTH, vocab_size
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=self.ignore_index)
        
        return logits, loss

    


#############################################################################################################################################################################################

import torch
import torch.nn as nn
import timm
from typing import Tuple, List
import os

class ImageCaptionModel(nn.Module):
    """
    Main class that integrates Vision Transformer (ViT) and GPT for image captioning.

    This model combines a Vision Transformer to encode images and a GPT model to generate captions based on the encoded image and provided tokens. 

    Attributes:
        device (str): Device on which the model is loaded ('cpu' or 'cuda').
        is_vit_pretrained (bool): Flag indicating whether the ViT model is pretrained.
        vit (nn.Module): Vision Transformer model for image encoding.
        gpt (GPT): GPT model for generating captions.
        dimension_mapping_layer (nn.Linear): Linear layer for mapping the dimension of the image encoding to match GPT's input dimension.
    """
    
    def __init__(self, config) -> None:
        """
        Initializes the ImageCaptionModel with ViT and GPT components, and sets up the dimension mapping layer.

        Args:
            config (dict): Configuration dictionary containing parameters for ViT, GPT, and other model components.
        """
        super().__init__()
        
        self.device = config['device']
        self.is_vit_pretrained = False
        
        # Initialize Vision Transformer
        if config['vit_kwargs']["pretrained_model_name"] is not None:
            self.is_vit_pretrained = True
            self.vit = timm.create_model(
                model_name=config['vit_kwargs']["pretrained_model_name"],
                pretrained=True,
                num_classes=0,
                global_pool='avg'
            )
            config["vit_kwargs"]["d_model"] = self.vit.embed_dim
        else:   
            self.vit = ViT(config['vit_kwargs'])
        
        # Initialize GPT
        self.gpt = GPT(config['gpt_kwargs'])
        
        # Linear layer to map image encoding dimension to GPT's input dimension
        self.dimension_mapping_layer = nn.Linear(config["vit_kwargs"]['d_model'], config["gpt_kwargs"]['d_model'])
        
    def forward(self, image: torch.Tensor, tokens: torch.Tensor, attn_mask: torch.Tensor, targets: torch.Tensor=None) -> Tuple[torch.Tensor]:
        """
        Forward pass through the ImageCaptionModel.

        Args:
            image (torch.Tensor): Input image tensor of shape (B, C, H, W).
            tokens (torch.Tensor): Input token tensor of shape (B, CTX_LENGTH).
            attn_mask (torch.Tensor): Attention mask tensor of shape (B, CTX_LENGTH).
            targets (torch.Tensor, optional): Target tensor for computing loss, shape (B, CTX_LENGTH).

        Returns:
            Tuple[torch.Tensor]: A tuple containing the GPT model's output logits (B, CTX_LENGTH, vocab_size) 
            and loss (if targets are provided).
        """
        # Encode image
        image_encoding = self.vit(image)  # (B, d_model)
        
        # Map image encoding to GPT's input dimension
        dimension_mapped_image_encoding = self.dimension_mapping_layer(image_encoding[:, None, :])  # (B, 1, d_model)
        
        # Forward pass through GPT
        return self.gpt(tokens, dimension_mapped_image_encoding, attn_mask, targets)

    @torch.inference_mode()
    def generate(self, 
                 image: torch.Tensor, 
                 sos_token: int,
                 eos_token: int,
                 max_len: int=40) -> List[int]:
        """
        Generates a caption for a given image using the GPT model.

        Args:
            image (torch.Tensor): Input image tensor of shape (B, C, H, W).
            sos_token (int): Start-of-sequence token ID.
            eos_token (int): End-of-sequence token ID.
            max_len (int, optional): Maximum length of the generated caption. Default is 40.

        Returns:
            List[int]: List of token IDs representing the generated caption.
        """
        # Encode image
        image_encoding = self.vit(image)  # (B, d_model)
        
        # Map image encoding to GPT's input dimension
        dimension_mapped_image_encoding = self.dimension_mapping_layer(image_encoding[:, None, :])  # (B, 1, d_model)
        
        # Initialize tokens with the start-of-sequence token
        tokens = torch.tensor([[sos_token]], requires_grad=False).to(self.device)
        attn_mask = torch.tensor([[1]], requires_grad=False).to(self.device)
        
        while tokens.shape[1] < max_len and tokens[0, -1] != eos_token:
            # Forward pass through GPT
            logits, _ = self.gpt(tokens, dimension_mapped_image_encoding, attn_mask, None)  # (1, N+1, vocab_size)
            
            # Predict the next token
            next_token = torch.argmax(logits[0, -1, :], dim=0).item()
            
            # Append the predicted token to the sequence
            tokens = torch.cat(
                (tokens, torch.tensor([[next_token]], requires_grad=False)),
                dim=-1
            ).to(self.device)
            
            # Update attention mask
            attn_mask = torch.cat(
                (attn_mask, torch.tensor([[1]], requires_grad=False)),
                dim=-1
            ).to(self.device)
        
        return list(tokens[0])
    
    @classmethod
    def from_pretrained(cls, checkpoint, device):
        """
        Loads a pre-trained ImageCaptionModel from a checkpoint file.

        Args:
            checkpoint (str): Path to the checkpoint file.
            device (str): Device to load the model onto ('cpu' or 'cuda').

        Returns:
            ImageCaptionModel: An instance of the ImageCaptionModel loaded with pre-trained weights.
        """
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(f"{checkpoint} does not exist")

        cp = torch.load(checkpoint, map_location=device)
        
        # Update device information in the model configuration
        cp['model_config']['device'] = device
        cp['model_config']['vit_kwargs']['device'] = device
        cp['model_config']['gpt_kwargs']['device'] = device

        # Initialize model with configuration and load state_dict
        model = cls(cp['model_config'])
        model.load_state_dict(cp['model_state_dict'])
        model = model.to(device)
        
        return model


if __name__ == '__main__':
    pass
