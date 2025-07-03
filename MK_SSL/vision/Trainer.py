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

from torch.utils.tensorboard import SummaryWriter



from MK_SSL.vision.models import *
from MK_SSL.vision.models.modules.losses import *
from MK_SSL.vision.models.modules.transformations import *
from MK_SSL.utils import configure_logging  

from MK_SSL.vision.registry import get_vision_method



class Trainer:
    def __init__(
        self,
        method: str,
        backbone: nn.Module,
        feature_size: int,
        image_size: int,
        configure_logger: bool = True,
        log_level: int = logging.INFO,
        save_dir: str = ".",
        checkpoint_interval: int = 10,
        reload_checkpoint: bool = False,
        verbose: bool = True,
        mixed_precision_training: bool = True,
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
            **kwargs: Additional keyword arguments for extending functionality or overriding default settings
                      specific to the training method or the backbone architecture.
        """


        if configure_logger:
            configure_logging()


        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)
        self.logger.info("Multimodal Trainer initialized.")

        self.logger.setLevel(logging.INFO if verbose else logging.WARNING)


        self.method = method.lower()
        self.image_size = image_size
        self.backbone = backbone
        self.feature_size = feature_size
        self.reload_checkpoint = reload_checkpoint
        self.checkpoint_interval = checkpoint_interval
        self.verbose = verbose
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
            "---------------- AK_SSL: Multimodal ----------------\n"
            f"Number of workers : {self.num_workers}\n"
            f"Device            : {self.device}\n"
            f"Method            : {self.method}\n"
            "----------------------------------------------------"
        )

        try:
            method_cfg = get_vision_method(self.method)
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
        self.writer = SummaryWriter("{}/Logs/{}".format(self.save_dir, self.timestamp))
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision_training)


        self.logger.info(
            "\n"
            "---------------- Model Summary ----------------\n"
            f"Model parameters : {np.sum([int(np.prod(p.shape)) for p in self.model.parameters()]):,}\n"
            "----------------------------------------------"
        )

    def __del__(self):
        self.writer.close()

    def get_backbone(self):
        return self.model.backbone

    def train_one_epoch(self, tepoch, optimizer):
        loss_hist_train = 0.0
        for images, _ in tepoch:
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
            else:
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
            self.dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )

        self.model.train(True)

        if self.reload_checkpoint:
            start_epoch = self._reload_latest_checkpoint() + 1

        for epoch in tqdm(
            range(start_epoch - 1, epochs),
            unit="epoch",
            desc="Pretext Task Model Training",
            leave=True,
        ):
            with tqdm(train_loader, unit="batch", leave=False) as tepoch:
                tepoch.set_description(f"Epoch {epoch + 1}")
                loss_per_epoch = self.train_one_epoch(tepoch, optimizer)

            self.writer.add_scalar(
                "Pretext Task/Loss/train",
                loss_per_epoch / len(train_loader),
                epoch + 1,
            )

            self.writer.flush()
            if (epoch + 1) % self.checkpoint_interval == 0:
                model_path = self.checkpoint_path + "{}_model_{}_epoch{}.pth".format(
                    self.method, self.timestamp, epoch + 1
                )
                torch.save(self.model.state_dict(), model_path)

        model_path = self.checkpoint_path + "{}_model_{}_epoch{}.pth".format(
            self.method, self.timestamp, epoch + 1
        )
        torch.save(self.model.state_dict(), model_path)

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
            train_dataset, batch_size=batch_size, shuffle=True
        )

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

                for images, labels in tepoch_ds:
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

                self.writer.add_scalar(
                    "Downstream Task/Loss/train",
                    loss_hist_train / len(train_loader_ds),
                    epoch + 1,
                )

                self.writer.add_scalar(
                    "Downstream Task/Accuracy/train",
                    acc_hist_train / len(train_loader_ds),
                    epoch + 1,
                )

                self.writer.flush()

        test_loader_ds = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=True
        )

        acc_test = 0.0
        net.eval()
        with torch.no_grad():
            for images, labels in tqdm(test_loader_ds, unit="batch"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = net(images)
                acc_test += multiclass_accuracy(outputs, labels, k=top_k).item()

        acc = 100 * acc_test / len(test_loader_ds)

        self.logger.info(
            "\n"
            "---------------- Test Accuracy ----------------\n"
            f"The top_{top_k} accuracy of the network on the {len(test_dataset)} test images: {acc}%\n"
            "-----------------------------------------------"
        )

        return acc

    def load_checkpoint(self, checkpont_dir: str):
        self.model.load_state_dict(torch.load(checkpont_dir))
        self.logger.info(
            "\n"
            "---------------- Checkpoint ----------------\n"
            "Checkpoint loaded.\n"
            "--------------------------------------------"
        )


    def save_backbone(self):
        torch.save(self.model.backbone.state_dict(), self.save_dir + "backbone.pth")

        self.logger.info(
            "\n"
            "---------------- Save Backbone ----------------\n"
            "Backbone saved.\n"
            f"Backbone file path : {os.path.join(self.save_dir, 'backbone.pth')}\n"
            "------------------------------------------------"
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
