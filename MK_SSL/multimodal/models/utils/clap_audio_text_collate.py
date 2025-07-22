
import torch
from transformers import BertTokenizer

class AudioTextCollator:
    """
    Collator class for audio-text datasets.
    Converts raw text (list[str]) into tokenized tensors using a HuggingFace tokenizer.
    """

    # Tokenizer initialized once at the class level
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


    def __call__(self, batch):
        """
        Collates a list of samples into a batch.
        Args:
            batch (list): List of samples, each being a dict with 'audio', 'text', and optionally 'label'.
        Returns:
            dict: A batch with tokenized text and stacked audio tensors.
        """
        audio = torch.stack([item["audio"] for item in batch])
        texts = [item["text"] for item in batch]
    
        # === Convert raw text to tokenized tensors ===
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
    
        # Optional: collate labels
        batch_dict = {"audio": audio, "text": texts}
        if "label" in batch[0]:
            batch_dict["label"] = torch.tensor([item["label"] for item in batch])
    
        return batch_dict

