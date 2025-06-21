import os
import re
import numpy as np
import torch
import logging
import torch.nn as nn
from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.clip_grad import clip_grad_norm_

from AK_SSL.multimodal.models import *
from AK_SSL.vision.models.modules.losses.nt_xent import NT_Xent
from AK_SSL.utils import configure_logging  

class Trainer:

    def __init__(
        self,
        method: str,
        image_encoder: nn.Module,
        text_encoder: nn.Module,
        configure_logger: bool = True,
        log_level: int = logging.INFO,
        mixed_precision_training: bool = True,
        save_dir: str = ".",
        checkpoint_interval: int = 10,
        reload_checkpoint: bool = False,
        verbose: bool = True,
        **kwargs,
    ) -> None:
        """
        Description:
            Initializes the Trainer class for self-supervised training of vision-language models.

        Args:
            method (str): The training method or framework to be used.
                          Options include ["CLIP", "ALBEF", "SimVLM", "SLIP", "UNITER", "VSE"].
            image_encoder (nn.Module): The neural network module responsible for extracting features from images.
            text_encoder (nn.Module): The neural network module responsible for extracting features from text.
            mixed_precision_training (bool, optional): If True, enables mixed precision training to reduce memory usage
                                                       and potentially speed up training. Defaults to True.
            save_dir (str, optional): The directory path where model checkpoints will be saved during training.
                                      Defaults to the current directory ("./").
            checkpoint_interval (int, optional): The number of training epochs between saving model checkpoints.
                                                 Defaults to 10.
            reload_checkpoint (bool, optional): If True, attempts to reload the most recent checkpoint from `save_dir`
                                                at the start of training, allowing continuation from a previous run.
                                                Defaults to False.
            verbose (bool, optional): If True, enables detailed logging and progress information during training.
                                      Defaults to True.
            **kwargs: Additional keyword arguments that can be passed to the image and text encoder models,
                      or used to customize the training process.
        """

        if configure_logger:
            configure_logging()


        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)
        self.logger.info("Multimodal Trainer initialized.")

        self.logger.setLevel(logging.INFO if verbose else logging.WARNING)


        self.method = method
        self.checkpoint_interval = checkpoint_interval
        self.reload_checkpoint = reload_checkpoint
        self.mixed_precision_training = mixed_precision_training




        
        self.save_dir = os.path.join(save_dir, self.method)

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

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


        match self.method.lower():
            case "clip":
                self.model = CLIP(
                    image_encoder=image_encoder,
                    text_encoder=text_encoder,
                    device=self.device,
                    **kwargs,
                )


                self.logger.info(
                    "\n"
                    "----------------CLIP Configuration----------------\n"
                    f"Embedding Dimension           : {self.model.embed_dim}\n"
                    f"Dimension of image features   : {self.model.image_feature_dim}\n"
                    f"Dimension of text features    : {self.model.text_feature_dim}\n"
                    f"Loss Function                 : {'SigLIP loss' if self.model.use_siglip else 'Contrastive loss'}\n"
                    f"Initial Tau                   : {self.model.init_tau}\n"
                    f"Initial Bias                  : {self.model.init_bias}"
                )


            case "albef":
                self.model = ALBEF(
                    image_encoder=image_encoder,
                    text_encoder=text_encoder,
                    device=self.device,
                    **kwargs,
                )



                self.logger.info(
                    "\n"
                    "---------------- ALBEF Configuration ----------------\n"
                    f"MLM Probability              : {self.model.mlm_probability}\n"
                    f"Embedding Dimension          : {self.model.embed_dim}\n"
                    f"Dimension of image features  : {self.model.image_feature_dim}\n"
                    f"Dimension of text features   : {self.model.text_feature_dim}\n"
                    f"Temperature                  : {self.model.temp}\n"
                    f"Queue Size                   : {self.model.queue_size}\n"
                    f"Momentum                     : {self.model.momentum}\n"
                    f"Alpha                        : {self.model.alpha}"
                )

            case "simvlm":
                self.model = SimVLM(
                    transformer_encoder=image_encoder,
                    transformer_decoder=text_encoder,
                    **kwargs,
                )

                self.logger.info(
                    "\n"
                    "---------------- SimVLM Configuration ----------------\n"
                    f"Vocabulary Size               : {self.model.vocab_size}\n"
                    f"Dimension of Features         : {self.model.feature_dim}\n"
                    f"Maximum Sequence Length       : {self.model.max_seq_len}\n"
                    f"Max Truncation Text Length    : {self.model.max_trunc_txt_len}\n"
                    f"Prefix Text Length            : {self.model.prefix_txt_len}\n"
                    f"Target Text Length            : {self.model.target_txt_len}\n"
                    f"Padding Index                 : {self.model.pad_idx}\n"
                    f"Resolution of Images          : {self.model.image_resolution}\n"
                    f"Patch Size                    : {self.model.patch_size}\n"
                    f"Number of Channels            : {self.model.num_channels}"
                )

            case "slip":
                self.model = SLIP(
                    image_encoder=image_encoder,
                    text_encoder=text_encoder,
                    device=self.device,
                    **kwargs,
                )

                self.logger.info(
                    "\n"
                    "---------------- SLIP Configuration ----------------\n"
                    f"Embedding Dimension           : {self.model.embed_dim}\n"
                    f"Dimension of image features   : {self.model.image_feature_dim}\n"
                    f"Dimension of text features    : {self.model.text_feature_dim}\n"
                    f"Dimension of the MLP          : {self.model.mlp_dim}"
                )

            case "uniter":
                self.model = UNITER(
                    image_encoder=image_encoder, text_encoder=text_encoder, **kwargs
                )

                self.logger.info(
                    "\n"
                    "---------------- UNITER Configuration ----------------\n"
                    f"Hidden Size                    : {self.model.hidden_size}\n"
                    f"Number of Answers              : {self.model.num_answer}\n"
                    f"Attention Dropout Probability  : {self.model.attention_probs_dropout_prob}\n"
                    f"Initializer Range              : {self.model.initializer_range}\n"
                    f"Pooler                         : {self.model.pooler}\n"
                    f"Encoder                        : {self.model.encoder}"
                )

            case "vse":
                self.model = VSE(
                    image_encoder=image_encoder, text_encoder=text_encoder, **kwargs
                )

                self.logger.info(
                    "\n"
                    "---------------- VSE Configuration ----------------\n"
                    f"Margin : {self.model.margin}"
                )

            case _:
                self.logger.error(f"Unsupported method: {self.method}")
                raise ValueError(f"Method {self.method} not supported")

        self.model = self.model.to(self.device)

        self.logger.info(
            "\n"
            "---------------- Model Parameters ----------------\n"
            f"Total Parameters : {np.sum([int(np.prod(p.shape)) for p in self.model.parameters()]):,}\n"
            "--------------------------------------------------"
        )


        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.writer = SummaryWriter("{}/Logs/{}".format(self.save_dir, self.timestamp))
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision_training)

    def __del__(self):
        self.writer.close()

    def _train_clip(self, tepoch, optimizer):
        epoch_loss = 0.0
        for step, (batch) in enumerate(tepoch):
            batch = {
                k: v.to(self.device)
                for k, v in batch.items()
                if k in ["input_ids", "attention_mask", "image"]
            }

            with torch.cuda.amp.autocast(enabled=self.mixed_precision_training):
                logits = self.model(**batch)
                if self.model.use_siglip:
                    loss = self.model.criterion_siglip_loss(logits)
                else:
                    loss = self.model.criterion_contrastive_loss(logits)

            optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()

            epoch_loss += loss.item()
            tepoch.set_postfix(
                loss=loss.item(),
                temp=self.model.t_prime.exp().item(),
                bias=self.model.b.item(),
                lr=optimizer.param_groups[0]["lr"],
            )

        return epoch_loss

    def _train_slip(self, tepoch, optimizer):
        epoch_loss = 0.0
        for step, (batch) in enumerate(tepoch):
            batch = {
                k: v.to(self.device)
                for k, v in batch.items()
                if k in ["input_ids", "attention_mask", "image"]
            }

            with torch.cuda.amp.autocast(enabled=self.mixed_precision_training):
                logits = self.model(**batch)
                ssl_loss = NT_Xent(temperature=0.1)
                ssl_loss = ssl_loss(logits["aug1_embed"], logits["aug2_embed"])
                clip_loss = self.model.clip.criterion_contrastive_loss(
                    logits["clip_output"]
                )

                loss = self.model.criterion(
                    ssl_scale=1.0, ssl_loss=ssl_loss, clip_loss=clip_loss
                )

            optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()

            epoch_loss += loss.item()
            tepoch.set_postfix(
                loss=loss.item(),
                temp=self.model.clip.t_prime.exp().item(),
                bias=self.model.clip.b.item(),
                lr=optimizer.param_groups[0]["lr"],
            )

        return epoch_loss

    def _train_simvlm(self, tepoch, optimizer):
        epoch_loss = 0.0
        for step, (batch) in enumerate(tepoch):
            batch = {
                k: v.to(self.device) for k, v in batch.items() if k in ["text", "image"]
            }

            with torch.cuda.amp.autocast(enabled=self.mixed_precision_training):
                logits, labels = self.model(**batch)
                loss = self.model.criterion(logits, labels)

            optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()

            epoch_loss += loss.item()
            tepoch.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])

        return epoch_loss

    def _train_vse(self, tepoch, optimizer):
        epoch_loss = 0.0
        num_negs = []
        for step, (batch) in enumerate(tepoch):
            batch = {
                k: v.to(self.device)
                for k, v in batch.items()
                if k in ["image", "image_lengths", "text", "text_lengths"]
            }

            with torch.cuda.amp.autocast(enabled=self.mixed_precision_training):
                img_emb, txt_emb, txt_lens = self.model(**batch)
                loss, tmp_num_negs = self.model.conterastive_loss(
                    img_emb, txt_emb, txt_lens
                )
                num_negs.extend(tmp_num_negs)

            optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            clip_grad_norm_(self.model.enc_params, 2.0)
            self.scaler.step(optimizer)
            self.scaler.update()

            epoch_loss += loss.item()
            tepoch.set_postfix(loss=loss.item(), epoch_negs=np.mean(num_negs))

        return epoch_loss

    def _train_albef(self, tepoch, optimizer, epoch):
        epoch_loss = 0.0
        for step, (batch) in enumerate(tepoch):
            batch = {
                k: v.to(self.device)
                for k, v in batch.items()
                if k in ["input_ids", "attention_mask", "image"]
            }
            if epoch > 0:
                alpha = self.model.alpha
            else:
                alpha = self.model.alpha * min(1, step / len(tepoch))
            with torch.cuda.amp.autocast(enabled=self.mixed_precision_training):
                loss_mlm, loss_ita, loss_itm = self.model(
                    image=batch["image"],
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    alpha=alpha,
                )
                loss = loss_mlm + loss_ita + loss_itm

            optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()

            epoch_loss += loss.item()
            tepoch.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])

        return epoch_loss

    def _train_unitervqa(self, tepoch, optimizer):
        epoch_loss = 0.0
        for step, (batch) in enumerate(tepoch):
            with torch.cuda.amp.autocast(enabled=self.mixed_precision_training):
                logits = self.model(**batch)
                loss = self.model.criterion(batch["targets"], logits)

            optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()

            epoch_loss += loss.item()
            tepoch.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])

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

        number_of_epochs = epochs - start_epoch + 1

        match optimizer.lower():
            case "adam":
                optimizer = torch.optim.Adam(
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
            case "sgd":
                optimizer = torch.optim.SGD(
                    list(self.model.parameters()),
                    lr=learning_rate,
                    weight_decay=weight_decay,
                )
            case _:
                raise ValueError(f"Optimizer {optimizer} not supported")

        if self.reload_checkpoint:
            start_epoch = self._reload_latest_checkpoint() + 1

        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

        self.model.train()

        match self.method.lower():
            case "clip":
                tmax = number_of_epochs * len(train_loader) + len(train_loader) // 4
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer=optimizer, T_max=tmax, eta_min=1e-8
                )

                for epoch in tqdm(
                    range(start_epoch - 1, epochs),
                    unit="epoch",
                    desc="CLIP Training",
                    leave=True,
                ):
                    with tqdm(train_loader, unit="batch", leave=False) as tepoch:
                        tepoch.set_description(f"Epoch {epoch + 1}")
                        loss_per_epoch = self._train_clip(tepoch, optimizer)
                        lr_scheduler.step()

                    self.writer.add_scalar(
                        f"{self.method.upper()}/Train/Loss",
                        loss_per_epoch / len(train_loader),
                        epoch + 1,
                    )
                    self.writer.flush()
                    if (epoch + 1) % self.checkpoint_interval == 0:
                        model_path = self.save_dir + "{}_model_{}_epoch{}.pth".format(
                            self.method, self.timestamp, epoch + 1
                        )
                        torch.save(self.model.state_dict(), model_path)

            case "slip":
                tmax = number_of_epochs * len(train_loader) + len(train_loader) // 4
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer=optimizer, T_max=tmax, eta_min=1e-5
                )

                for epoch in tqdm(
                    range(start_epoch - 1, epochs),
                    unit="epoch",
                    desc="SLIP Training",
                    leave=True,
                ):
                    with tqdm(train_loader, unit="batch", leave=False) as tepoch:
                        tepoch.set_description(f"Epoch {epoch + 1}")
                        loss_per_epoch = self._train_slip(tepoch, optimizer)
                        lr_scheduler.step()

                    self.writer.add_scalar(
                        f"{self.method.upper()}/Train/Loss",
                        loss_per_epoch / len(train_loader),
                        epoch + 1,
                    )
                    self.writer.flush()
                    if (epoch + 1) % self.checkpoint_interval == 0:
                        model_path = self.save_dir + "{}_model_{}_epoch{}.pth".format(
                            self.method, self.timestamp, epoch + 1
                        )
                        torch.save(self.model.state_dict(), model_path)

            case "albef":
                tmax = number_of_epochs * len(train_loader) + len(train_loader) // 4
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=tmax, eta_min=1e-5
                )

                for epoch in tqdm(
                    range(start_epoch - 1, epochs),
                    unit="epoch",
                    desc="ALBEF Training",
                    leave=True,
                ):
                    with tqdm(train_loader, unit="batch", leave=False) as tepoch:
                        tepoch.set_description(f"Epoch {epoch + 1}")
                        loss_per_epoch = self._train_albef(tepoch, optimizer, epoch)
                        lr_scheduler.step()

                    self.writer.add_scalar(
                        f"{self.method.upper()}/Train/Loss",
                        loss_per_epoch / len(train_loader),
                        epoch + 1,
                    )
                    self.writer.flush()
                    if (epoch + 1) % self.checkpoint_interval == 0:
                        model_path = self.save_dir + "{}_model_{}_epoch{}.pth".format(
                            self.method, self.timestamp, epoch + 1
                        )
                        torch.save(self.model.state_dict(), model_path)

            case "simvlm":
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer, T_0=2000
                )

                for epoch in tqdm(
                    range(start_epoch - 1, epochs),
                    unit="epoch",
                    desc="SimVLM Training",
                    leave=True,
                ):
                    with tqdm(train_loader, unit="batch", leave=False) as tepoch:
                        tepoch.set_description(f"Epoch {epoch + 1}")
                        loss_per_epoch = self._train_simvlm(tepoch, optimizer)
                        lr_scheduler.step()

                    self.writer.add_scalar(
                        f"{self.method.upper()}/Train/Loss",
                        loss_per_epoch / len(train_loader),
                        epoch + 1,
                    )
                    self.writer.flush()
                    if (epoch + 1) % self.checkpoint_interval == 0:
                        model_path = self.save_dir + "{}_model_{}_epoch{}.pth".format(
                            self.method, self.timestamp, epoch + 1
                        )
                        torch.save(self.model.state_dict(), model_path)

            case "uniter_vqa":
                for epoch in tqdm(
                    range(start_epoch - 1, epochs),
                    unit="epoch",
                    desc="Uniter For VQA Training",
                    leave=True,
                ):
                    with tqdm(train_loader, unit="batch", leave=False) as tepoch:
                        tepoch.set_description(f"Epoch {epoch + 1}")
                        loss_per_epoch = self._train_unitervqa(tepoch, optimizer)

                    self.writer.add_scalar(
                        f"{self.method.upper()}/Train/Loss",
                        loss_per_epoch / len(train_loader),
                        epoch + 1,
                    )
                    self.writer.flush()
                    if (epoch + 1) % self.checkpoint_interval == 0:
                        model_path = self.save_dir + "{}_model_{}_epoch{}.pth".format(
                            self.method, self.timestamp, epoch + 1
                        )
                        torch.save(self.model.state_dict(), model_path)

            case "vse":
                for epoch in tqdm(
                    range(start_epoch - 1, epochs),
                    unit="epoch",
                    desc="VSE Training",
                    leave=True,
                ):
                    with tqdm(train_loader, unit="batch", leave=False) as tepoch:
                        tepoch.set_description(f"Epoch {epoch + 1}")
                        loss_per_epoch = self._train_vse(tepoch, optimizer)

                    self.writer.add_scalar(
                        f"{self.method.upper()}/Train/Loss",
                        loss_per_epoch / len(train_loader),
                        epoch + 1,
                    )
                    self.writer.flush()
                    if (epoch + 1) % self.checkpoint_interval == 0:
                        model_path = self.save_dir + "{}_model_{}_epoch{}.pth".format(
                            self.method, self.timestamp, epoch + 1
                        )
                        torch.save(self.model.state_dict(), model_path)

            case _:
                raise ValueError(f"Method {self.method} not supported")

        model_path = self.save_dir + "{}_model_{}_epoch{}.pth".format(
            self.method, self.timestamp, epoch + 1
        )
        torch.save(self.model.state_dict(), model_path)

    def load_checkpoint(self, checkpont_dir: str):
        self.model.load_state_dict(torch.load(checkpont_dir))


        self.logger.info(
            "\n"
            "---------------- Checkpoint ----------------\n"
            "Checkpoint loaded.\n"
            "--------------------------------------------"
        )


    def _reload_latest_checkpoint(self):
        checkpoints = os.listdir(self.save_dir)
        sorted_checkpoints = sorted(
            [os.path.join(self.save_dir, i) for i in checkpoints],
            key=os.path.getmtime,
        )

        if len(sorted_checkpoints) == 0:
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
            self.logger.error("No epoch number found in the checkpoint name.")
            raise ValueError("No epoch number found in the checkpoint name.")

        return epoch
