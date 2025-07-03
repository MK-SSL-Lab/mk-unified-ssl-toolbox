import os
import importlib.util
from huggingface_hub import login, whoami, hf_hub_download
from transformers import AutoModel, AutoConfig
from torch import nn


class NotAuthenticatedError(Exception):
    """Raised when a Hugging Face user is not authenticated."""
    def __init__(self):
        message = (
            "You are not authenticated with Hugging Face Hub.\n"
            "Please login using:\n\n"
            "    HFHubInterface.authenticate(token=\"<your_token>\")\n\n"
            "or set the environment variable HUGGINGFACE_HUB_TOKEN."
        )
        super().__init__(message)


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
        token = token or os.getenv("HUGGINGFACE_HUB_TOKEN")
        if token:
            login(token=token, add_to_git_credential=True)
            HFHubInterface._user = whoami()
            return HFHubInterface._user
        return None

    @staticmethod
    def _check_auth():
        if HFHubInterface._user is None:
            try:
                HFHubInterface._user = whoami()
            except Exception:
                raise NotAuthenticatedError()

    @staticmethod
    def load_module(model_id: str, pretrained: bool = True, **kwargs) -> nn.Module:
        HFHubInterface._check_auth()
        if pretrained:
            return AutoModel.from_pretrained(model_id, **kwargs)
        config = AutoConfig.from_pretrained(model_id)
        return AutoModel.from_config(config)

    @staticmethod
    def load_method(repo_id: str, filename: str = "registry.py") -> dict:
        HFHubInterface._check_auth()
        path = hf_hub_download(repo_id=repo_id, filename=filename)
        spec = importlib.util.spec_from_file_location("hf_method", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.method
