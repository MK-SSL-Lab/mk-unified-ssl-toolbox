import torch
import torch.nn as nn


class InfoNCELoss(nn.Module):
    """
    InfoNCE Loss (as used in COLA): A softmax-based contrastive loss function that 
    maximizes agreement between positive pairs and contrasts them against many negatives.
    
    Similar to NT-Xent but uses bilinear similarity instead of cosine similarity.
    
    Args:
        temperature (float): Temperature scaling factor for logits.
        input_dim (int): Dimensionality of projected embeddings.
    """

    def __init__(self, temperature: float = 0.2, input_dim: int = 512, **kwargs):
        super().__init__()
        self.temperature = temperature
        self.eps = 1e-8
        self.similarity = nn.Bilinear(input_dim, input_dim, 1, bias=False)

        if abs(self.temperature) < self.eps:
            raise ValueError(f"Illegal temperature: abs({self.temperature}) < 1e-8")

    def forward(self, out0: torch.Tensor, out1: torch.Tensor) -> torch.Tensor:
        """
        Args:
            out0 (torch.Tensor): Embeddings from view 1 of shape (B, D).
            out1 (torch.Tensor): Embeddings from view 2 of shape (B, D).
        
        Returns:
            torch.Tensor: Scalar loss value.
        """
        device = out0.device
        batch_size = out0.size(0)

        # Normalize embeddings (optional, improves stability sometimes)
        out0 = nn.functional.normalize(out0, dim=1)
        out1 = nn.functional.normalize(out1, dim=1)

        # Combine: z = [z0, z1]
        z = torch.cat([out0, out1], dim=0)  # (2B, D)

        # Compute pairwise similarity using bilinear function
        logits = torch.zeros((2 * batch_size, 2 * batch_size), device=device)
        for i in range(2 * batch_size):
            logits[i] = self.similarity(z[i].unsqueeze(0).repeat(2 * batch_size, 1), z)

        # Apply temperature scaling
        logits /= self.temperature

        # Positive indices: for i in [0, B), match with i+B; for i in [B, 2B), match with i-B
        labels = torch.arange(batch_size, device=device)
        labels = torch.cat([labels + batch_size, labels], dim=0)

        # Cross-entropy loss so we don't need masking
        loss = nn.CrossEntropyLoss()(logits, labels)
        return loss
