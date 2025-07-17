import wandb
import os
import logging # Import logging for integration
from typing import Optional, Dict, Any
from MK_SSL.utils import get_logger_handler  # Assuming this exists

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
                 api_key: Optional[str] = None,
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
        
        self._logger = logging.getLogger(self.__class__.__name__)

        if not self._logger.hasHandlers():
            self._logger.addHandler(get_logger_handler())

        self._logger.setLevel(logging.INFO if verbose else logging.WARNING)
        self._logger.info("Audio Trainer initialized.")


        if self.mode == "online" and api_key:
            try:
                wandb.login(key=api_key, relogin=True)
                self._logger.info("W&B login successful.")
            except Exception as e:
                self._logger.error(f"Failed to login to W&B with provided API key: {e}")


    def init_run(self,
                 run_name: Optional[str] = None,
                 ):
        """
        Initializes a new W&B run based on the logger's configuration.
        """

        self.run_name = run_name if run_name else self.run_name

        if self._run is not None and self._run.id is not None:
            self._logger.warning("W&B run already active. Finishing previous run before starting a new one.")
            self.finish_run()

        try:
            self._run = wandb.init(
                project=self.project_name,
                entity=self.entity,
                mode=self.mode,
                name=self.run_name,
                config=self.initial_config,
                notes=self.notes,
                tags=self.tags,
                **self.kwargs
            )
            if self._run and self.mode != "disabled":
                self._logger.info(f"W&B logging initialized. Mode: {self.mode.upper()}. "
                                  f"View run at: {self._run.url}")
            elif self.mode == "disabled":
                self._logger.info("W&B logging is disabled for this run.")
            else:
                # This case typically means offline mode or a failed init without a direct URL
                self._logger.info("W&B initialization successful, but no direct URL available (e.g., offline mode).")

        except wandb.errors.UsageError as e:
            self._logger.error(f"W&B initialization error: {e}")
            self._logger.error("Please ensure you are logged in (`wandb login`) if using 'online' mode, "
                               "or check your network connection/local server setup.")
            self._run = None # Ensure _run is None if init fails
        except Exception as e:
            self._logger.error(f"An unexpected error occurred during W&B initialization: {e}")
            self._run = None

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Logs metrics to the current W&B run."""
        if self._run and self.mode != "disabled":
            try:
                wandb.log(metrics, step=step)
            except Exception as e:
                self._logger.error(f"Failed to log metrics to W&B: {e}")

    def watch_model(self, model, criterion=None, log="all", log_freq=100):
        """Watches a PyTorch model for gradients and parameters."""
        if self._run and self.mode != "disabled":
            try:
                wandb.watch(model, criterion, log=log, log_freq=log_freq)
            except Exception as e:
                self._logger.error(f"Failed to set W&B model watch: {e}")

    def save_artifact(self, file_path: str, name: str, type: str = "model", metadata: Optional[Dict] = None):
        """Saves a file as a W&B artifact."""
        if self._run and self.mode != "disabled":
            try:
                artifact = wandb.Artifact(name=name, type=type, metadata=metadata)
                artifact.add_file(file_path)
                wandb.log_artifact(artifact)
                self._logger.info(f"Artifact '{name}' (type: {type}) saved to W&B.")
            except Exception as e:
                self._logger.error(f"Failed to save artifact '{name}' to W&B: {e}")

    def finish_run(self):
        """Finishes the current W&B run."""
        if self._run and self.mode != "disabled":
            try:
                self._run.finish()
                self._logger.info("W&B run finished.")
            except Exception as e:
                self._logger.error(f"Failed to finish W&B run: {e}")
        self._run = None # Reset the run object

    @property
    def current_run(self):
        """Returns the current W&B run object."""
        return self._run

    @property
    def is_active(self) -> bool:
        """Checks if a W&B run is currently active and not disabled."""
        return self._run is not None and self.mode != "disabled"        