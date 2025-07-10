import os
import re
import torch
import numpy as np
import os
from torch import nn
from tqdm.auto import tqdm
from datetime import datetime
from torch.utils.data import Subset, DataLoader # Added DataLoader for clarity
import logging
from torcheval.metrics.functional import multiclass_accuracy

import optuna

# from torch.utils.tensorboard import SummaryWriter # Commented out: Replaced by WandbLogger


from MK_SSL.vision.models import *
from MK_SSL.vision.models.modules.losses import *
from MK_SSL.vision.models.modules.transformations import *
from MK_SSL.utils import configure_logging, get_logger_handler

from MK_SSL.vision.models.utils import get_method
from typing import Optional, Dict, Any # Added for type hinting W&B args

# Import your WandbLogger utility
# Make sure your_library.wandb_utils is accessible, e.g., in the same directory
# or properly installed as part of your package.
from MK_SSL.utils import WandbLogger


class Trainer:
    def __init__(
        self,
        method: str,
        backbone: nn.Module,
        feature_size: int,
        image_size: int,
        configure_logger: bool = True,
        save_dir: str = ".",
        checkpoint_interval: int = 10,
        reload_checkpoint: bool = False,
        verbose: bool = True,
        mixed_precision_training: bool = True,
        # W&B specific arguments
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        wandb_mode: str = "online", # "online", "offline", "disabled"
        wandb_run_name: Optional[str] = None,
        wandb_config: Optional[Dict[str, Any]] = None,
        wandb_notes: Optional[str] = None,
        wandb_tags: Optional[list[str]] = None,
        **kwargs,
    ) -> None:
        """
        Description:
            Trainer class for training a model using self-supervised learning methods. This class manages the
            training loop, model saving, and supports advanced features such as mixed precision training and
            checkpointing.

        Args:
            method (str): The self-supervised learning method to be used for training.
                          Available options include:
                          - 'BarlowTwins'
                          - 'BYOL'
                          - 'DINO'
                          - 'MoCov2'
                          - 'MoCov3'
                          - 'SimCLR'
                          - 'SimSiam'
                          - 'SwAV'
            backbone (nn.Module): The neural network module serving as the backbone of the model.
            feature_size (int): The dimensionality of the feature vector output by the backbone model.
            image_size (int): The dimensions (height, width) of the input images. This is generally expected to
                              be a square (i.e., height equals width).
            save_dir (str): Path to the directory where model checkpoints and logs will be saved. Defaults to
                            the current directory ("./").
            checkpoint_interval (int): Frequency (in epochs) at which model checkpoints are saved. For example,
                                        if set to 10, the model will be saved every 10 epochs.
            reload_checkpoint (bool): If set to True, training will resume from the latest checkpoint available
                                      in the `save_dir`. If False, training will start from scratch.
            verbose (bool): If True, detailed logs and progress updates will be printed during training.
            mixed_precision_training (bool): If True, mixed precision (using both 16-bit and 32-bit floats)
                                             will be used during training to improve performance and reduce memory usage.
            wandb_project (str, optional): W&B project name. If None, uses default from W&B.
            wandb_entity (str, optional): W&B entity (username or team name). If None, uses default.
            wandb_mode (str, optional): W&B logging mode ("online", "offline", "disabled"). Defaults to "online".
            wandb_run_name (str, optional): Custom name for the W&B run.
            wandb_config (Dict[str, Any], optional): Dictionary of hyperparameters/settings for W&B.
            wandb_notes (str, optional): Notes for the W&B run.
            wandb_tags (list[str], optional): Tags for the W&B run.
            **kwargs: Additional keyword arguments for extending functionality or overriding default settings
                      specific to the training method or the backbone architecture.
        """


        if configure_logger:
            configure_logging()


        self.logger = logging.getLogger(self.__class__.__name__)

        if not self.logger.hasHandlers():
            self.logger.addHandler(get_logger_handler())

        self.logger.setLevel(logging.INFO if verbose else logging.WARNING)
        
        self.logger.info("Vision Trainer initialized.")


        self.method = method.lower()
        self.image_size = image_size
        self.backbone = backbone
        self.feature_size = feature_size
        self.reload_checkpoint = reload_checkpoint
        self.checkpoint_interval = checkpoint_interval
        self.mixed_precision_training = mixed_precision_training

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
            "---------------- MK_SSL: Vision ----------------\n"
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
        
        model_special_overrides = {
            "barlowtwins": {"hidden_dim": self.feature_size},
            "simsiam": {
                "projection_hidden_dim": self.feature_size,
                "prediction_hidden_dim": self.feature_size // 4
            }
        }

        loss_special_overrides = {
            "dino": {
                "projection_dim": self.model.projection_dim,
                "temp_student": self.model.temp_student,
                "temp_teacher": self.model.temp_teacher,
            },
            "swav": {"num_crops": self.model.num_crops + 2,}
        }


        self.model = method_cfg["model"](
            backbone=self.backbone,
            feature_size=self.feature_size,

            **model_special_overrides.get(
                self.method, {}
            ),
            **kwargs
        )


        self.loss = method_cfg["loss"](
            **loss_special_overrides.get(
                self.method, {}
            ),
            **kwargs
        )

        self.transformation = method_cfg["transformation"](
            image_size=self.image_size, 
            **kwargs
        )

        # Only define transformation_prime if needed
        if self.method in {"byol", "barlowtwins", "simclr", "simsiam", "mocov3"}:
            self.transformation_prime = self.transformation

        if self.method in {"dino"}:
            self.transformation_global1 = self.transformation
            self.transformation_global2 = self.transformation
            self.transformation_local = self.transformation

        if self.method in {"swav"}:
            self.transformation_global = self.transformation
            self.transformation_local = self.transformation

        self.logger.info(method_cfg["logs"](self.model, self.loss))
        
        
        self.model = self.model.to(self.device)
        self.loss = self.loss.to(self.device)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision_training)


        self.logger.info(
            "\n"
            "---------------- Model Summary ----------------\n"
            f"Model parameters : {np.sum([int(np.prod(p.shape)) for p in self.model.parameters()]):,}\n"
            "----------------------------------------------"
        )

        # --- W&B Logger Initialization ---
        # Combine trainer_config with any specific wandb_config provided
        trainer_internal_config = {
            "method": self.method,
            "feature_size": self.feature_size,
            "image_size": self.image_size,
            "save_dir": save_dir,
            "checkpoint_interval": checkpoint_interval,
            "reload_checkpoint": reload_checkpoint,
            "mixed_precision_training": mixed_precision_training,
            "device": str(self.device),
            "num_workers": self.num_workers,
            **kwargs # Include any other kwargs passed to Trainer init
        }
        full_wandb_config = {**trainer_internal_config, **(wandb_config if wandb_config else {})}

        self.wandb_logger = WandbLogger(
            project_name=wandb_project if wandb_project else f"MK_SSL_Vision_{self.method}", # Default project name
            entity=wandb_entity,
            mode=wandb_mode,
            run_name=wandb_run_name,
            config=full_wandb_config,
            notes=wandb_notes if wandb_notes else f"Training {self.method} vision model with MK_SSL.",
            tags=wandb_tags if wandb_tags else [self.method, "vision", "training"],
        )

    def __del__(self):
        pass # No need for TensorBoard writer close if not used

    def get_backbone(self):
        return self.model.backbone

    def train_one_epoch(self, tepoch, optimizer, epoch_idx, total_batches_per_epoch): # Added epoch_idx, total_batches_per_epoch
        loss_hist_train = 0.0
        # Watch the model with W&B if active
        if self.wandb_logger.is_active:
            self.wandb_logger.watch_model(self.model)

        for step, (images, _) in enumerate(tepoch): # Added step for global step calculation
            images = images.to(self.device)
            if self.method.lower() in ["barlowtwins", "byol", "mocov3"]:
                with torch.cuda.amp.autocast(enabled=self.mixed_precision_training):
                    view0 = self.transformation(images)
                    view1 = self.transformation_prime(images)
                    z0, z1 = self.model(view0, view1)
                    loss = self.loss(z0, z1)
            elif self.method.lower() in ["dino"]:
                with torch.cuda.amp.autocast(enabled=self.mixed_precision_training):
                    view0 = self.transformation_global1(images)
                    view1 = self.transformation_global2(images)
                    viewc = []
                    if self.model.num_crops > 0:
                        for _ in range(self.model.num_crops):
                            viewc.append(self.transformation_local(images))
                    z0, z1 = self.model(view0, view1, viewc)
                    loss = self.loss(z0, z1)
            elif self.method.lower() in ["swav"]:
                with torch.cuda.amp.autocast(enabled=self.mixed_precision_training):
                    view0 = self.transformation_global(images)
                    view1 = self.transformation_global(images)
                    viewc = []
                    if self.model.num_crops > 0:
                        for _ in range(self.model.num_crops):
                            viewc.append(self.transformation_local(images))
                    z0, z1 = self.model(view0, view1, viewc)
                    loss = self.loss(z0, z1)
            else: # SimCLR, SimSiam, MoCov2 (assuming these use transformation twice)
                with torch.cuda.amp.autocast(enabled=self.mixed_precision_training):
                    view0 = self.transformation(images)
                    view1 = self.transformation(images)
                    z0, z1 = self.model(view0, view1)
                    loss = self.loss(z0, z1)

            optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
            loss_hist_train += loss.item()
            
            # Log batch-level metrics to W&B
            if self.wandb_logger.is_active:
                global_batch_step = (epoch_idx * total_batches_per_epoch) + step
                self.wandb_logger.log({
                    f"{self.method.upper()}/Train/Batch_Loss": loss.item(),
                    f"{self.method.upper()}/Train/LR": optimizer.param_groups[0]["lr"],
                }, step=global_batch_step)

            tepoch.set_postfix(loss=loss.item())

        return loss_hist_train

    def train(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int = 256,
        start_epoch: int = 1,
        epochs: int = 100,
        optimizer: str = "Adam",
        weight_decay: float = 1e-6,
        learning_rate: float = 1e-3,
    ):
        """
        Description:
            Train the model.

        Args:
            dataset (torch.utils.data.Dataset): Dataset to train.
            batch_size (int): Batch size.
            start_epoch (int): Epoch to start the training.
            epochs (int): Number of epochs.
            optimizer (str): Optimizer to train the model. Options: [Adam, SGD, AdamW]
            weight_decay (float): Weight decay.
            learning_rate (float): Learning rate.
        """
        # Initialize W&B run at the very beginning of the main train method
        if self.wandb_logger.is_active:
            self.wandb_logger.init_run()
            # Update W&B config with dynamic training parameters
            self.wandb_logger.current_run.config.update({
                "batch_size": batch_size,
                "start_epoch": start_epoch,
                "max_epochs": epochs,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "optimizer": optimizer,
            })
            self.logger.info(f"W&B run initialized. View run at: {self.wandb_logger.current_run.url}")
        else:
            self.logger.info("W&B logging is not active for this run.")


        self.dataset = dataset
        match optimizer.lower():
            case "adam":
                optimizer = torch.optim.Adam(
                    list(self.model.parameters()),
                    lr=learning_rate,
                    weight_decay=weight_decay,
                )
            case "sgd":
                optimizer = torch.optim.SGD(
                    list(self.model.parameters()),
                    lr=learning_rate,
                    weight_decay=weight_decay,
                )
            case "adamw":
                optimizer = torch.optim.AdamW(
                    list(self.model.parameters()),
                    lr=learning_rate,
                    weight_decay=weight_decay,
                )
            case _:
                self.logger.error(f"Unsupported Optimizer: {optimizer}")
                
                raise ValueError(f"Optimizer {optimizer} not supported")

        train_loader = torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size, shuffle=True, drop_last=True,
            num_workers=self.num_workers # Add num_workers for consistency
        )
        total_batches_per_epoch = len(train_loader) # Used for global step calculation

        self.model.train()

        if self.reload_checkpoint:
            start_epoch = self._reload_latest_checkpoint() + 1

        for epoch in tqdm( # epoch is 0-indexed loop variable (range(start-1, epochs))
            range(start_epoch - 1, epochs),
            unit="epoch",
            desc="Pretext Task Model Training",
            leave=True,
        ):
            with tqdm(train_loader, unit="batch", leave=False) as tepoch:
                tepoch.set_description(f"Epoch {epoch + 1}")
                loss_per_epoch = self.train_one_epoch(tepoch, optimizer, epoch, total_batches_per_epoch) # Pass epoch_idx, total_batches_per_epoch
            
            # To stop from full training for optuna
            if hasattr(self, "_optuna_trial"):
                self._optuna_trial.report(loss_per_epoch, epoch)
                if self._optuna_trial.should_prune():
                    raise optuna.TrialPruned()
    

            # Log epoch-level metrics to W&B
            if self.wandb_logger.is_active:
                self.wandb_logger.log({
                    f"{self.method.upper()}/Train/Epoch_Loss": loss_per_epoch / len(train_loader),
                    f"{self.method.upper()}/Train/LR": optimizer.param_groups[0]["lr"],
                }, step=epoch + 1) # Use epoch + 1 for 1-indexed epoch step


            if (epoch + 1) % self.checkpoint_interval == 0:
                model_path = self.checkpoint_path + "/{}_model_{}_epoch{}.pth".format( # Added / for path joining
                    self.method, self.timestamp, epoch + 1
                )
                torch.save(self.model.state_dict(), model_path)
                self.logger.info(f"Model checkpoint saved: {model_path}")
                # Save model checkpoint as W&B artifact
                if self.wandb_logger.is_active:
                    self.wandb_logger.save_artifact(
                        model_path,
                        name=f"{self.method}-model-epoch-{epoch+1}",
                        type="model",
                        metadata={"epoch": epoch+1, "loss": loss_per_epoch / len(train_loader)}
                    )

        # Save final model after all epochs
        # Note: 'epoch' here will be the last value from the loop, which is `epochs - 1` (0-indexed)
        # So, for the filename, it should be `epochs` (1-indexed total epochs)
        final_model_path = self.checkpoint_path + "/{}_model_{}_final.pth".format( # Changed to final
            self.method, self.timestamp
        )
        torch.save(self.model.state_dict(), final_model_path)
        self.logger.info(f"Final model saved: {final_model_path}")
        # Save final model as W&B artifact
        if self.wandb_logger.is_active:
            self.wandb_logger.save_artifact(
                final_model_path,
                name=f"{self.method}-model-final",
                type="model",
                metadata={"epochs_trained": epochs, "final_loss": loss_per_epoch / len(train_loader)} # Use final epoch's loss
            )

        # Finish W&B run at the very end of the main train method
        if self.wandb_logger.is_active:
            self.wandb_logger.finish_run()
            self.logger.info("Main training process completed and W&B run finalized.")
        else:
            self.logger.info("Main training process completed.")


    def evaluate(
        self,
        train_dataset: torch.utils.data.Dataset,
        test_dataset: torch.utils.data.Dataset,
        eval_method: str = "linear",
        top_k: int = 1,
        epochs: int = 100,
        optimizer: str = "Adam",
        weight_decay: float = 1e-6,
        learning_rate: float = 1e-3,
        batch_size: int = 256,
        fine_tuning_data_proportion: float = 1,
    ):
        """
        Description:
            Evaluate the model using the given evaluating method.

        Args:
            eval_method (str): Evaluation method. Options: [linear, finetune]
            top_k (int): Top k accuracy.
            epochs (int): Number of epochs.
            optimizer (str): Optimizer to train the model. Options: [Adam, SGD, AdamW]
            weight_decay (float): Weight decay.
            learning_rate (float): Learning rate.
            batch_size (int): Batch size.
            train_dataset (torch.utils.data.Dataset): Dataset to train the downstream model.
            test_dataset (torch.utils.data.Dataset): Dataset to test the downstream model.
            fine_tuning_data_proportion (float): Proportion of the dataset between 0 and 1 to use for fine-tuning.

        """
        # Start W&B run for evaluation if not already active (e.g., if evaluation is run standalone)
        # If train() was called before, the run might still be active.
        # This ensures evaluation metrics are logged to the same run or a new one.
        if not self.wandb_logger.is_active:
            # Re-init W&B logger for evaluation context if not already active from training
            # This might create a new run if not explicitly linked to a previous one.
            # For simplicity, we'll assume a new run if not active.
            # You might want to add a specific project/run_name for evaluation runs.
            self.logger.info("W&B logger not active, initializing for evaluation.")
            self.wandb_logger.init_run() # This will create a new run if none is active

        # Log evaluation parameters to W&B config
        if self.wandb_logger.is_active:
            self.wandb_logger.current_run.config.update({
                "eval_method": eval_method,
                "eval_top_k": top_k,
                "eval_epochs": epochs,
                "eval_optimizer": optimizer,
                "eval_weight_decay": weight_decay,
                "eval_learning_rate": learning_rate,
                "eval_batch_size": batch_size,
                "fine_tuning_data_proportion": fine_tuning_data_proportion,
            })
            self.logger.info("Evaluation parameters logged to W&B config.")


        match eval_method.lower():
            case "linear":
                net = EvaluateNet(
                    self.model.backbone,
                    self.feature_size,
                    len(train_dataset.classes),
                    True,
                )
            case "finetune":
                if not 0 <= fine_tuning_data_proportion <= 1:

                    self.logger.error(f"The fine_tuning_data_proportion parameter must be between 0 and 1.")
                    
                    raise ValueError(
                        "The fine_tuning_data_proportion parameter must be between 0 and 1."
                    )

                net = EvaluateNet(
                    self.model.backbone,
                    self.feature_size,
                    len(train_dataset.classes),
                    False,
                )

                num_samples = len(train_dataset)
                subset_size = int(num_samples * fine_tuning_data_proportion)

                indices = torch.randperm(num_samples)[:subset_size]

                train_dataset = Subset(train_dataset, indices)

        match optimizer.lower():
            case "adam":
                optimizer_eval = torch.optim.Adam(
                    net.parameters(), lr=learning_rate, weight_decay=weight_decay
                )
            case "sgd":
                optimizer_eval = torch.optim.SGD(
                    net.parameters(), lr=learning_rate, weight_decay=weight_decay
                )
            case "adamw":
                optimizer_eval = torch.optim.AdamW(
                    net.parameters(), lr=learning_rate, weight_decay=weight_decay
                )
            case _:

                self.logger.error(f"Unsupported Optimizer: {optimizer}")

                raise ValueError(f"Optimizer {optimizer} not supported")

        net = net.to(self.device)
        criterion = nn.CrossEntropyLoss()

        train_loader_ds = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=self.num_workers # Add num_workers
        )
        total_batches_per_eval_epoch = len(train_loader_ds) # For global step calculation

        net.train(True)
        scaler_eval = torch.cuda.amp.GradScaler(enabled=self.mixed_precision_training)

        for epoch in tqdm(
            range(epochs),
            unit="epoch",
            desc="Evaluate Model Training",
            leave=True,
        ):
            with tqdm(train_loader_ds, unit="batch", leave=False) as tepoch_ds:
                tepoch_ds.set_description(f"Epoch {epoch + 1}")
                loss_hist_train, acc_hist_train = 0.0, 0.0

                for step, (images, labels) in enumerate(tepoch_ds): # Added step
                    correct, total = 0, 0

                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    with torch.cuda.amp.autocast(enabled=self.mixed_precision_training):
                        outputs = net(images)
                        loss = criterion(outputs, labels)

                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    acc = 100 * correct / total
                    acc_hist_train += acc

                    tepoch_ds.set_postfix(loss=loss.item(), accuracy=f"{acc:.2f}")
                    loss_hist_train += loss.item()
                    optimizer_eval.zero_grad()
                    scaler_eval.scale(loss).backward()
                    scaler_eval.step(optimizer_eval)
                    scaler_eval.update()

                    # Log batch-level evaluation train metrics to W&B
                    if self.wandb_logger.is_active:
                        global_eval_batch_step = (epoch * total_batches_per_eval_epoch) + step
                        self.wandb_logger.log({
                            f"Downstream Task/{eval_method.capitalize()}/Batch_Loss": loss.item(),
                            f"Downstream Task/{eval_method.capitalize()}/Batch_Accuracy": acc,
                            f"Downstream Task/{eval_method.capitalize()}/LR": optimizer_eval.param_groups[0]["lr"],
                        }, step=global_eval_batch_step)



                # Log epoch-level evaluation train metrics to W&B
                if self.wandb_logger.is_active:
                    self.wandb_logger.log({
                        f"Downstream Task/{eval_method.capitalize()}/Epoch_Loss": loss_hist_train / len(train_loader_ds),
                        f"Downstream Task/{eval_method.capitalize()}/Epoch_Accuracy": acc_hist_train / len(train_loader_ds),
                    }, step=epoch + 1)


        test_loader_ds = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=True,
            num_workers=self.num_workers # Add num_workers
        )

        acc_test = 0.0
        net.eval()
        with torch.no_grad():
            for images, labels in tqdm(test_loader_ds, unit="batch"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = net(images)
                acc_test += multiclass_accuracy(outputs, labels, k=top_k).item()

        final_test_accuracy = 100 * acc_test / len(test_loader_ds)

        self.logger.info(
            "\n"
            "---------------- Test Accuracy ----------------\n"
            f"The top_{top_k} accuracy of the network on the {len(test_dataset)} test images: {final_test_accuracy:.2f}%\n" # Formatted for clarity
            "-----------------------------------------------"
        )

        # Log final test accuracy to W&B summary
        if self.wandb_logger.is_active:
            self.wandb_logger.log({
                f"Downstream Task/{eval_method.capitalize()}/Final_Test_Accuracy_top_{top_k}": final_test_accuracy
            })
            # Also add to summary for easy comparison across runs
            self.wandb_logger.current_run.summary[f"final_test_accuracy_top_{top_k}"] = final_test_accuracy

        # Finish W&B run after evaluation if it was started by evaluate and not already finished by train
        if self.wandb_logger.is_active: # Check again if it's still active
            self.wandb_logger.finish_run()
            self.logger.info("Evaluation process completed and W&B run finalized.")
        else:
            self.logger.info("Evaluation process completed.")

        return final_test_accuracy

    def load_checkpoint(self, checkpont_dir: str):
        self.model.load_state_dict(torch.load(checkpont_dir, map_location=self.device)) # Add map_location
        self.logger.info(
            "\n"
            "---------------- Checkpoint ----------------\n"
            "Checkpoint loaded.\n"
            "--------------------------------------------"
        )


    def save_backbone(self):
        # Ensure save_dir has a trailing slash or use os.path.join
        backbone_path = os.path.join(self.save_dir, "backbone.pth")
        torch.save(self.model.backbone.state_dict(), backbone_path)

        self.logger.info(
            "\n"
            "---------------- Save Backbone ----------------\n"
            "Backbone saved.\n"
            f"Backbone file path : {backbone_path}\n"
            "------------------------------------------------"
        )
        # Save backbone as W&B artifact
        if self.wandb_logger.is_active:
            self.wandb_logger.save_artifact(
                backbone_path,
                name=f"{self.method}-backbone",
                type="model_backbone",
                metadata={"feature_size": self.feature_size, "image_size": self.image_size}
            )


    def _reload_latest_checkpoint(self):
        checkpoints = os.listdir(self.checkpoint_path)
        sorted_checkpoints = sorted(
            [os.path.join(self.checkpoint_path, i) for i in checkpoints],
            key=os.path.getmtime,
        )

        if len(sorted_checkpoints) == 0:


            self.logger.error(f"No checkpoints found in the directory")
            
            raise ValueError("No checkpoints found in the directory")

        self.load_checkpoint(sorted_checkpoints[-1])

        match = re.search(r"epoch(\d+)", sorted_checkpoints[-1])
        if match:
            epoch = int(match.group(1))

            self.logger.info(
                "\n"
                "---------------- Checkpoint Reload ----------------\n"
                f"Starting Epoch : {epoch}\n"
                "---------------------------------------------------"
            )


        else:
            self.logger.error(f"No epoch number found in the checkpoint name.")
            
            raise ValueError("No epoch number found in the checkpoint name.")

        return epoch
