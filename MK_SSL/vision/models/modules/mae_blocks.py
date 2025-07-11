import torch
import torch.nn as nn
from einops import rearrange, repeat


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding """
    def __init__(self, freq_bins: int, time_steps: int, patch_size: int, embed_dim: int):
        super().__init__()
        assert freq_bins % patch_size == 0 and time_steps % patch_size == 0, "Patch size must divide dimensions"
        self.patch_size = patch_size
        self.num_patches = (freq_bins // patch_size) * (time_steps // patch_size)
        self.proj = nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, 1, F, T)
        x = self.proj(x)  # (B, D, F/P, T/P)
        x = rearrange(x, 'b d f t -> b (f t) d')  # (B, N, D)
        return x


class ViTBlock(nn.Module):
    """ Standard Vision Transformer Block """
    def __init__(self, dim: int, heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0])
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x


# Main MAE Architecture

class MaskedAutoencoderViT(nn.Module):
    """
    Masked Autoencoder with VisionTransformer backbone.

    This module implements the Masked Autoencoder (MAE) model for self-supervised
    learning on 2D data like images or spectrograms. The core idea is to randomly
    mask a high portion of the input patches and train the model to reconstruct
    the missing patches in the pixel/signal space.

    The architecture consists of:
    1. A PatchEmbed layer to convert the input into a sequence of tokens.
    2. A ViT-based Encoder that processes only the visible (unmasked) patches.
    3. A lightweight ViT-based Decoder that takes the encoded patches along with
       learnable mask tokens to reconstruct the full sequence.
    4. A final linear layer to project the decoder's output back to the
       original patch dimension for reconstruction.

    Args:
        input_shape (tuple): Shape of the input (frequency_bins, time_steps).
        patch_size (int): Size of each square patch.
        embed_dim (int): Embedding dimension of the encoder.
        encoder_depth (int): Number of transformer blocks in the encoder.
        encoder_heads (int): Number of attention heads in the encoder.
        decoder_embed_dim (int): Embedding dimension of the decoder.
        decoder_depth (int): Number of transformer blocks in the decoder.
        decoder_heads (int): Number of attention heads in the decoder.
        mlp_ratio (float): Ratio for the MLP hidden dimension in transformer blocks.
        mask_ratio (float): The ratio of patches to be masked.
    """
    def __init__(
        self,
        input_shape=(128, 1024),
        patch_size=16,
        embed_dim=768,       # Encoder dimension
        encoder_depth=12,
        encoder_heads=12,
        decoder_embed_dim=512, # Decoder dimension (usually smaller)
        decoder_depth=8,
        decoder_heads=16,
        mlp_ratio=4.0,
        mask_ratio=0.75,
    ):
        super().__init__()

        self.mask_ratio = mask_ratio
        self.patch_size = patch_size

        # 1. Patch Embedding section
        self.patch_embed = PatchEmbed(input_shape[0], input_shape[1], patch_size, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        # 2. MAE Encoder
        self.encoder = nn.Sequential(*[
            ViTBlock(embed_dim, encoder_heads, mlp_ratio) for _ in range(encoder_depth)
        ])
        self.encoder_norm = nn.LayerNorm(embed_dim)

        # 3. MAE Decoder
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim))
        
        self.decoder = nn.Sequential(*[
            ViTBlock(decoder_embed_dim, decoder_heads, mlp_ratio) for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)

        # 4. Final layer for patch reconstruction
        # The output must match the number of pixels in a patch
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2, bias=True)

        self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.normal_(self.pos_embed, std=.02)
        torch.nn.init.normal_(self.decoder_pos_embed, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        Converts the input into patches.
        imgs: (B, 1, F, T)
        x: (B, N, patch_size**2)
        """
        p = self.patch_size
        B, C, F, T = imgs.shape
        x = imgs.reshape(B, C, F // p, p, T // p, p)
        x = torch.einsum('bcfptq->bfn(p q)', x) # or use rearrange
        return x

    def random_masking(self, x):
        """
        Randomly masks patches.
        x: (B, N, D)
        """
        B, N, D = x.shape
        len_keep = int(N * (1 - self.mask_ratio))
        
        noise = torch.rand(B, N, device=x.device)
        
        # Sort indices based on noise
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Separate the visible indices
        ids_keep = ids_shuffle[:, :len_keep]
        
        # Extract the visible patches
        x_visible = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))
        
        return x_visible, ids_restore, ids_keep

    def forward_encoder(self, x, ids_keep):
        # Add positional embedding only to the visible patches
        x_visible_with_pos = x + torch.gather(self.pos_embed, 1, ids_keep.unsqueeze(-1).expand(-1, -1, x.shape[-1]))

        # Run the encoder
        x_encoded = self.encoder(x_visible_with_pos)
        x_encoded = self.encoder_norm(x_encoded)
        return x_encoded

    def forward_decoder(self, x_encoded, ids_restore):
        # Project to decoder space
        x_encoded = self.decoder_embed(x_encoded)

        # Prepare mask tokens
        B, len_keep, D_dec = x_encoded.shape
        N = self.decoder_pos_embed.shape[1]
        
        mask_tokens = self.mask_token.repeat(B, N - len_keep, 1)

        # Concatenate encoded patches and mask tokens
        x_full = torch.cat([x_encoded, mask_tokens], dim=1)
        
        # Restore the original order of patches
        x_full = torch.gather(x_full, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, D_dec))
        
        # Add decoder positional embeddings
        x_full = x_full + self.decoder_pos_embed

        # Run the decoder
        x_decoded = self.decoder(x_full)
        x_decoded = self.decoder_norm(x_decoded)

        # Reconstruct the patches
        x_pred = self.decoder_pred(x_decoded)
        return x_pred

    def forward_loss(self, imgs, pred, ids_restore):
        """
        Calculate loss only on the masked patches.
        imgs: [B, 1, F, T]
        pred: [B, N, p*p]
        """
        target = self.patchify(imgs)

        # Extract the masked patches from the target
        N = pred.shape[1]
        len_keep = N - int(N * self.mask_ratio)
        ids_masked = ids_restore[:, len_keep:]
        
        target_masked = torch.gather(target, dim=1, index=ids_masked.unsqueeze(-1).expand(-1, -1, target.shape[-1]))
        pred_masked = torch.gather(pred, dim=1, index=ids_masked.unsqueeze(-1).expand(-1, -1, pred.shape[-1]))
        
        loss = (pred_masked - target_masked).pow(2).mean()
        return loss

    def forward(self, imgs):
        """
        Main forward pass for the pre-training process.
        """
        # 1. Convert input to embedded patches
        latent = self.patch_embed(imgs)

        # 2. Perform random masking
        latent_visible, ids_restore, ids_keep = self.random_masking(latent)

        # 3. Run the encoder on visible patches only
        encoded = self.forward_encoder(latent_visible, ids_keep)

        # 4. Run the decoder to reconstruct masked patches
        pred = self.forward_decoder(encoded, ids_restore)

        # 5. Calculate the loss
        loss = self.forward_loss(imgs, pred, ids_restore)
        
        return loss, pred, ids_restore



class MAEDecoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 768,           # encoder output dim
        decoder_dim: int = 512,         # internal dim of decoder
        depth: int = 8,
        heads: int = 8,
        mlp_ratio: float = 4.0,
        patch_size: int = 16,
        input_dim: tuple = (128, 1024),  # (freq, time)
    ):
        super().__init__()
        self.input_dim = input_dim
        self.patch_size = patch_size
        self.num_patches = (input_dim[0] // patch_size) * (input_dim[1] // patch_size)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))

        self.proj = nn.Linear(embed_dim, decoder_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, decoder_dim))

        self.blocks = nn.Sequential(*[
            ViTBlock(decoder_dim, heads, mlp_ratio) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(decoder_dim)

        patch_dim = patch_size * patch_size  # if input is image-like (e.g., spectrogram patch)
        self.head = nn.Linear(decoder_dim, patch_dim)


    def forward(self, x_visible, ids_restore):
        """
          Args:
              x_visible (B, N_visible, D_enc): encoded visible tokens
              ids_restore (B, N_total): indices to restore full sequence
              ids_keep (B, N_visible): indices of the visible tokens
          Returns:
        """
        x_proj = self.proj(x_visible)
        B, N_visible, D = x_proj.shape
        N_total = ids_restore.shape[1]


        # 1. Create a full-length tensor filled with mask tokens
        x_full = self.mask_token.repeat(B, N_total, 1)
    
        # 2. Insert visible tokens back into their original positions
        # scatter_ works in-place
        ids_keep_expanded = ids_keep.unsqueeze(-1).expand(-1, -1, D)
        x_full.scatter_(dim=1, index=ids_keep_expanded, src=x_proj)
    
        # 3. Add positional embeddings
        x_full = x_full + self.pos_embed
    
        # 4. Decode the full sequence
        x_decoded = self.blocks(x_full)
        x_decoded = self.norm(x_decoded)
        x_rec = self.head(x_decoded)
    
        return x_rec
