import os
import re
import torch
import numpy as np
import os
from torch import nn
from tqdm.auto import tqdm
from datetime import datetime
from torch.utils.data import Subset
import logging
from torcheval.metrics.functional import multiclass_accuracy

from MK_SSL.audio.models.utils import get_method



class Trainer:
    def __init__(
        self,
        method: str,
        backbone: nn.Module,
        
        
        
        save_dir: str = ".",
        configure_logger: bool = True,
        verbose: bool = True,
        **kwargs,
    ) -> None:
        """
        Initializes the Trainer class.
        """
        if configure_logger:
            configure_logger()

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO if verbose else logging.WARNING)
        self.logger.info("Vision Trainer initialized.")


        self.method = method.lower()
        self.backbone = backbone

        self.save_dir = os.path.join(save_dir, self.method)

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.checkpoint_path = os.path.join(self.save_dir , "Pretext")

        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_workers = os.cpu_count()

        self.logger.info(
            "\n"
            "---------------- AK_SSL: Audio ----------------\n"
            f"Number of workers : {self.num_workers}\n"
            f"Device            : {self.device}\n"
            f"Method            : {self.method}\n"
            "----------------------------------------------------"
        )

        try:
            method_cfg = get_method(self.method)
        except ValueError as e:
            self.logger.error(f"Method {self.method} not found in registry.")
            raise e