import torch
from transformers import BertTokenizer

class AudioTextCollator:
    """
    Collator class for audio-text datasets.
    Converts raw text (list[str]) into tokenized tensors using a HuggingFace tokenizer.
    Also pads variable-length audio to the max length in the batch and returns lengths.
    """

    # Tokenizer initialized once at the class level
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def __call__(self, batch):
        """
        Collates a list of samples into a batch.
        Args:
            batch (list): List of samples, each being a dict with 'audio', 'text' (or 'caption'), and optionally 'label'.
        Returns:
            dict: A batch with tokenized text, padded audio tensors, and audio lengths.
        """
        # --- AUDIO: pad to max length & keep true lengths ---
        audios = [item["audio"] for item in batch]  # each [C, T] (mono => [1, T])
        lengths = torch.tensor([a.shape[-1] for a in audios], dtype=torch.long)  # true lengths (T)
        max_len = int(lengths.max().item())

        # Determine channel dimension consistently
        first = audios[0]
        if first.dim() == 1:
            C = 1
        else:
            C = first.shape[0]

        padded = torch.zeros(len(audios), C, max_len, dtype=first.dtype)
        for i, a in enumerate(audios):
            T = a.shape[-1]
            # reshape if 1D to [1, T]
            if a.dim() == 1:
                a = a.unsqueeze(0)
            padded[i, :, :T] = a

        # --- TEXT: keep your existing behavior exactly ---
        texts = [item["text"] for item in batch] if "text" in batch[0] else None
        if texts is not None:
            if isinstance(texts, list) and isinstance(texts[0], str):
                texts = self.tokenizer(
                    texts,
                    padding='max_length',
                    truncation=True,
                    max_length=100,
                    return_tensors='pt'
                )
            elif isinstance(texts, dict):
                texts = {k: v for k, v in texts.items()}
            elif isinstance(texts, torch.Tensor):
                texts = texts
            else:
                raise TypeError(f"Unsupported text input type: {type(texts)}")

        # --- assemble batch dict (unchanged keys + new 'audio_lengths') ---
        batch_dict = {"audio": padded, "audio_lengths": lengths}
        if texts is not None:
            batch_dict["text"] = texts
        if "label" in batch[0]:
            batch_dict["label"] = torch.tensor([item["label"] for item in batch], dtype=torch.long)

        return batch_dict
