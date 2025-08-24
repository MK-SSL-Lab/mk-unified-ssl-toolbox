import torch

class AudioMultimodalCollator:
    """
    Collator for audio-centric multimodal batches.
    - Pads variable-length audio to the max length in the batch (zero-padding).
    - Returns true audio lengths.
    - Passes through 'text' and/or 'image' exactly as provided (no processing).
    - Optionally collates 'label' into a tensor if present.
    """

    def __call__(self, batch):
        """
        Collates a list of samples into a batch.
        Each sample is a dict with:
          - 'audio': Tensor [C, T] or [T]  (required)
          - optionally 'text': any type (left untouched, returned as a list)
          - optionally 'image': any type (left untouched, returned as a list)
          - optionally 'label': number/tensor (collated to a tensor)

        Returns:
          dict with:
            - 'audio': Tensor [B, C, max_T]
            - 'audio_lengths': LongTensor [B]
            - passthrough keys (e.g., 'text', 'image') as lists
            - optional 'label' as a tensor
        """
        if len(batch) == 0:
            raise ValueError("Empty batch passed to AudioMultimodalCollator.")

        # --- AUDIO: pad to max length & keep true lengths ---
        audios = [item["audio"] for item in batch]
        first = audios[0]

        # Determine channels and validate shapes
        if first.dim() == 1:
            C = 1
        elif first.dim() == 2:
            C = first.shape[0]
        else:
            raise ValueError(f"'audio' must be 1D or 2D, got shape {tuple(first.shape)}")

        lengths = torch.as_tensor([a.shape[-1] for a in audios], dtype=torch.long)
        max_len = int(lengths.max().item())

        dtype = first.dtype
        device = first.device
        padded = torch.zeros(len(audios), C, max_len, dtype=dtype, device=device)

        for i, a in enumerate(audios):
            if a.dim() == 1:
                a = a.unsqueeze(0)  # [1, T]
            T = a.shape[-1]
            padded[i, :, :T] = a

        batch_dict = {
            "audio": padded,
            "audio_lengths": lengths,
        }

        # --- PASSTHROUGH: return 'text' and/or 'image' exactly as they are (lists) ---
        # If a key exists in any item, return a list aligned with the batch order.
        if "text" in batch[0]:
            batch_dict["text"] = batch_dict["text"]
        if "image" in batch[0]:
            images = [item["image"] for item in batch]           # list of [3,H,W] tensors
            batch_dict["image"] = torch.stack(images, dim=0)     # [B,3,H,W]

        # --- Optional: collate labels if present ---
        if "label" in batch[0]:
            labels = [item["label"] for item in batch]
            # Best-effort dtype inference
            if isinstance(labels[0], float):
                batch_dict["label"] = torch.tensor(labels, dtype=torch.float, device=device)
            elif isinstance(labels[0], int):
                batch_dict["label"] = torch.tensor(labels, dtype=torch.long, device=device)
            else:
                batch_dict["label"] = torch.as_tensor(labels, device=device)

        return batch_dict
