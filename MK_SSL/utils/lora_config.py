from dataclasses import dataclass
from typing import Optional
from peft import TaskType, LoraConfig


@dataclass
class LoRAArguments:
    """
    Configuration class for applying LoRA (Low-Rank Adaptation) using Hugging Face PEFT.

    This class allows users to specify how LoRA should be applied to a transformer model.
    It supports fine-tuning on various tasks such as masked language modeling, sequence classification,
    and causal language modeling.

    Attributes:
        r (int): Rank of the low-rank decomposition. Controls the capacity of the LoRA layers.
            - Typical values: 4, 8, 16, 32
            - Default: 8

        lora_alpha (int): Scaling factor applied to the LoRA outputs.
            - Usually a multiple of r (e.g., 1x, 2x, or 4x r)
            - Default: 32 (i.e., 4 * r)

        target_modules (Optional[list[str]]): List of module names to which LoRA should be applied.
            - For BERT-like models: ["query", "value"]
            - For GPT models: ["c_attn"]
            - Default: ["query", "value"] if None

        lora_dropout (float): Dropout probability applied to the LoRA paths.
            - Range: 0.0 to 0.5
            - Default: 0.1

        bias (str): Whether to train bias parameters in addition to LoRA parameters.
            - Options:
                - "none": Do not train any bias terms (recommended for pure LoRA)
                - "all": Train all bias terms
                - "lora_only": Train only bias terms inside LoRA layers
            - Default: "none"

        task_type (str): Specifies the type of downstream task. Used internally by PEFT.
            - Options:
                - "TOKEN_CLS" for token-level tasks (e.g., Masked Language Modeling)
                - "SEQ_CLS" for sequence classification
                - "CAUSAL_LM" for causal language modeling (e.g., GPT)
            - Default: "TOKEN_CLS"

    Example:
        >>> lora_args = LoRAArguments(
        ...     r=16,
        ...     lora_alpha=64,
        ...     target_modules=["query", "value", "key"],
        ...     lora_dropout=0.05,
        ...     bias="none",
        ...     task_type="TOKEN_CLS"
        ... )
        >>> config = lora_args.to_peft_config()
        >>> model = get_peft_model(model, config)

    See Also:
        - HuggingFace PEFT documentation: https://github.com/huggingface/peft
    """

    r: int = 8
    lora_alpha: int = 32
    target_modules: Optional[list[str]] = None
    lora_dropout: float = 0.1
    bias: str = "none"
    task_type: str = "FEATURE_EXTRACTION"

    def to_peft_config(self) -> LoraConfig:
        """
        Converts this configuration to a PEFT-compatible `LoraConfig` object.

        Returns:
            peft.LoraConfig: A configuration object that can be passed to `get_peft_model()`.
        """
        return LoraConfig(
            r=self.r,
            lora_alpha=self.lora_alpha,
            target_modules=self.target_modules or ["query", "value"],
            lora_dropout=self.lora_dropout,
            bias=self.bias,
            task_type=TaskType[self.task_type],
        )
