# normal_audio_collate.py
import torch

def normal_audio_collate(batch):
    """
    Audio-only collate function.
    - Pads variable-length audio to the max length in the batch (zero-padding).
    - Returns the true (unpadded) lengths per sample.
    - Keeps labels if present. Ignores any other keys.

    Expects each item in `batch` to be a dict with:
      - "audio": Tensor [C, T] or [T]
      - optionally "label": int/float/tensor

    Returns:
      dict with:
        - "audio": Tensor [B, C, max_T]
        - "audio_lengths": LongTensor [B]
        - optional "label": LongTensor/FloatTensor [B]
    """
    # Collect audio tensors
    audios = [item["audio"] for item in batch]
    if len(audios) == 0:
        raise ValueError("Empty batch passed to normal_audio_collate.")

    # Ensure shapes and determine channels per sample (support [T] or [C, T])
    # Compute true lengths
    lengths = torch.tensor(
        [a.shape[-1] if a.dim() > 0 else 0 for a in audios],
        dtype=torch.long
    )
    max_len = int(lengths.max().item())

    # Determine channel dimension consistently from first item
    first = audios[0]
    if first.dim() == 1:
        C = 1
    elif first.dim() == 2:
        C = first.shape[0]
    else:
        raise ValueError(f"Audio tensor must be 1D or 2D, got shape {tuple(first.shape)}")

    dtype = first.dtype
    device = first.device

    # Allocate padded tensor [B, C, max_T]
    padded = torch.zeros(len(audios), C, max_len, dtype=dtype, device=device)

    # Copy each sample into padded tensor
    for i, a in enumerate(audios):
        if a.dim() == 1:
            a = a.unsqueeze(0)  # [1, T]
        T = a.shape[-1]
        padded[i, :, :T] = a

    batch_dict = {
        "audio": padded,
        "audio_lengths": lengths
    }

    # Optionally collate labels if present
    if "label" in batch[0]:
        labels = [item["label"] for item in batch]
        # Convert to tensor with best-effort dtype inference
        if isinstance(labels[0], float):
            batch_dict["label"] = torch.tensor(labels, dtype=torch.float, device=device)
        elif isinstance(labels[0], int):
            batch_dict["label"] = torch.tensor(labels, dtype=torch.long, device=device)
        else:
            batch_dict["label"] = torch.as_tensor(labels, device=device)

    return batch_dict
