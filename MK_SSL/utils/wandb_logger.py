import wandb
import os
import logging # Import logging for integration
from typing import Optional, Dict, Any
from MK_SSL.utils import configure_logging, get_logger_handler  # Assuming this exists

class WandbLogger:
    """
    A utility class to manage Weights & Biases logging for the library.
    Handles initialization, configuration, and run management.
    """
    def __init__(self,
                 project_name: str,
                 entity: Optional[str] = None,
                 mode: str = "online", # "online", "offline", "disabled"
                 run_name: Optional[str] = None,
                 config: Optional[Dict[str, Any]] = None,
                 notes: Optional[str] = None,
                 tags: Optional[list[str]] = None,
                 verbose: bool=True,
                 **kwargs): # Allow arbitrary wandb.init kwargs
        """
        Initializes the W&B logger.

        Args:
            project_name (str): The name of the W&B project.
            entity (str, optional): The W&B entity (username or team name).
                                    Defaults to None, letting W&B pick up current user.
            mode (str): W&B logging mode. Can be "online", "offline", or "disabled".
                        - "online": Logs to W&B cloud (requires authentication).
                        - "offline": Logs to local files in the current directory.
                        - "disabled": Disables W&B logging entirely.
            run_name (str, optional): A unique name for the W&B run. If None, W&B generates one.
            config (Dict[str, Any], optional): Dictionary of hyperparameters and settings to log.
            notes (str, optional): A longer description of the run.
            tags (list[str], optional): List of tags for the run.
            **kwargs: Additional keyword arguments passed directly to wandb.init().
                      Useful for advanced settings like 'dir', 'settings', etc.
        """
        self.project_name = project_name
        self.entity = entity
        self.mode = mode
        # Generate a unique run name if not provided
        self.run_name = run_name if run_name else f"{project_name}-run-{wandb.util.generate_id()}"
        self.initial_config = config if config is not None else {}
        self.notes = notes
        self.tags = tags
        self.kwargs = kwargs
        self._run = None # Store the active W&B run object

        # Get a logger for this utility class
        configure_logging()
        
        self.logger = logging.getLogger(self.__class__.__name__)

        if not self.logger.hasHandlers():
            self.logger.addHandler(get_logger_handler())

        self.logger.setLevel(logging.INFO if verbose else logging.WARNING)
        self.logger.info("Audio Trainer initialized.")