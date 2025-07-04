import os
import importlib.util
from huggingface_hub import login, whoami, hf_hub_download
from transformers import AutoModel, AutoConfig
from torch import nn

class HFHubInterface:
    """
    Interface for interacting with Hugging Face Hub:
    - Authentication
    - Loading models (modules)
    - Loading methods (registry files)
    """

    _user = None

    @staticmethod
    def authenticate(token: str = None) -> dict | None:
        """
        Authenticates the user with the Hugging Face Hub.

        Args:
            token (str, optional): Hugging Face token. If not provided, uses the
                HUGGINGFACE_HUB_TOKEN environment variable.

        Returns:
            dict or None: User information if authentication is successful, else None.
        """
        token = token or os.getenv("HUGGINGFACE_HUB_TOKEN")
        if token:
            login(token=token, add_to_git_credential=True)
            HFHubInterface._user = whoami()
            return HFHubInterface._user
        return None

    @staticmethod
    def _check_auth():
        """
        Checks whether the user is authenticated with the Hugging Face Hub.
        
        Raises:
            PermissionError: If the user is not authenticated.
        """
        if HFHubInterface._user is None:
            try:
                HFHubInterface._user = whoami()
            except Exception:
                raise PermissionError(
                    "You are not authenticated with Hugging Face Hub.\n"
                    "Please login using:\n\n"
                    "HFHubInterface.authenticate(token=\"<your_token>\")\n\n"
                    "or set the environment variable HUGGINGFACE_HUB_TOKEN."
                )

    @staticmethod
    def load_model(model_id: str, pretrained: bool = True, **kwargs) -> nn.Module:
        """
        Loads a model from Hugging Face Hub using its identifier.

        Args:
            model_id (str): Hugging Face model ID (e.g., 'facebook/wav2vec2-base').
            pretrained (bool, optional): Whether to load pretrained weights.
                Defaults to True.
            **kwargs: Additional keyword arguments passed to `from_pretrained`.

        Returns:
            nn.Module: The loaded PyTorch model.
        """
        HFHubInterface._check_auth()
        if pretrained:
            return AutoModel.from_pretrained(model_id, **kwargs)
        config = AutoConfig.from_pretrained(model_id)
        return AutoModel.from_config(config)

    @staticmethod
    def load_method(model_id: str, **config_overrides) -> nn.Module:
        """
        Loads a model architecture from Hugging Face Hub without pretrained weights.

        This method is intended for use during the pretext phase of self-supervised learning,
        where you may need to initialize a backbone architecture and define your own training logic.
        Since training procedures vary depending on the specific SSL method, this function only
        provides the model architecture. You are responsible for implementing a compatible trainer.

        Args:
            model_id (str): Hugging Face model ID (e.g., 'facebook/wav2vec2-base').
            **config_overrides: Keyword arguments to override the default configuration.

        Returns:
            nn.Module: A PyTorch model instantiated from configuration without pretrained weights.
        """
        HFHubInterface._check_auth()
        config = AutoConfig.from_pretrained(model_id, **config_overrides)
        return AutoModel.from_config(config)
