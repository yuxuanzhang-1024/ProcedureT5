"""Model for Language Modeling."""

import logging
from typing import Any, Dict, Type, Union

import sentencepiece as _sentencepiece
import pytorch_lightning as pl
import torch
import torch.optim as optim
from torch import Tensor
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    XLNetLMHeadModel,
)
from time import time
from datetime import datetime
from analysis import partial_accuracy, original_bleu, levenshtein_similarity
# sentencepiece has to be loaded before lightning to avoid segfaults
_sentencepiece

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class BaseLightningModule(pl.LightningModule):
    """Pytorch lightning base model."""

    def __init__(
        self,
        model_args: Dict[str, Any],
    ) -> None:
        """Construct a Pytorch lightning base model.
        Args:
            model_args: model's arguments.
        """
        super().__init__()

        self.model_args = model_args
        self.model: torch.nn.Module
        self.val_results = {'pred': [], 'target': [], 'loss': []}
        self.train_loss_list = []

    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        """Forward pass.
        Args:
            x: tensor of shape (batch_size, seq_length) containing the input_ids.
        Returns:
            logits of the model.
        """
        return self.model(x).logits  # type:ignore

    def configure_optimizers(self) -> Dict[str, object]:  # type:ignore
        """Create and return the optimizer.
        Returns:
            output (dict of str: Any):
                - optimizer: the optimizer used to update the parameter.
                - ls_scheduler: the scheduler used to reduce the learning rate in every epoch.
                - monitor: the metric that the scheduler will track over the training.
        """

        if not isinstance(self.model_args["lr"], float):
            raise ValueError("Learning rate should be float")

        if not isinstance(self.model_args["lr_decay"], float):
            raise ValueError("Learning rate decay rate should be float")

        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.model_args["lr"],
            weight_decay=self.model_args["weight_decay"],
        )

        scheduler = optim.lr_scheduler.StepLR(optimizer, 1, self.model_args["lr_decay"])

        output = {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }
        return output

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:  # type: ignore
        """
        Training step which encompasses the forward pass and the computation of the loss value.
        Args:
            batch: dictionary containing the input_ids and optionally the token_type_ids and the attention_type.
            batch_idx: index of the current batch, unused.
        Returns:
            loss computed on the batch.
        """
        loss = self.model(**batch).loss  # type:ignore
        self.train_loss_list.append(loss)
        if self.global_step % self.model_args["train_loss_log_interval"] == 0:
            train_loss = torch.tensor(self.train_loss_list).mean().to(self.device)
            self.train_loss_list = []
            self.log("train_loss", train_loss, sync_dist=True, prog_bar=True, on_epoch=True)
            if self.model_args["display_mode"] == "remote":
                # log to terminal
                t = time()
                t = datetime.fromtimestamp(t).strftime('%Y-%m-%d %H:%M:%S')
                step = self.global_step
                total_steps = self.train_total_steps
                logger.info(f"Time: {t}, Epoch: {self.current_epoch}, Step: {step}/{total_steps}, Train Loss: {train_loss:.4f}")
        return loss

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:  # type: ignore
        """
        Validation step which encompasses the forward pass and the computation of the loss value.
        Args:
            batch: dictionary containing the input_ids and optionally the token_type_ids and the attention_type.
            batch_idx: index of the current batch, unused.
        Returns:
            loss computed on the batch.
        """
        results = self.model(**batch)
        logits, loss = results.logits, results.loss
        pred = torch.argmax(logits, dim=-1)
        # decode batches into sequences
        pred_str = self.tokenizer.batch_decode(pred, skip_special_tokens=True)
        target_str = self.tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)
        self.val_results['pred'].extend(pred_str)
        self.val_results['target'].extend(target_str)
        self.val_results['loss'].append(loss)
        return loss

    def on_validation_epoch_end(self) -> None:
        """Function called at the end of the validation epoch."""

        val_loss = torch.tensor(self.val_results['loss']).mean().to(self.device)
        acc_list = levenshtein_similarity(self.val_results['pred'], self.val_results['target'])
        acc_100 = sum([1 if acc==1.0 else 0 for acc in acc_list])/len(acc_list)
        acc_90 = sum([1 if acc>=0.9 else 0 for acc in acc_list])/len(acc_list)
        acc_75 = sum([1 if acc>=0.75 else 0 for acc in acc_list])/len(acc_list)
        acc_50 = sum([1 if acc>=0.50 else 0 for acc in acc_list])/len(acc_list)
        bleu = original_bleu(self.val_results['pred'], self.val_results['target'])
        self.log("val_loss", val_loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("val_acc_100", acc_100, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("val_acc_90", acc_90, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("val_acc_75", acc_75, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("val_acc_50", acc_50, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("val_bleu", bleu, prog_bar=True, on_epoch=True, sync_dist=True)
        if self.model_args["display_mode"] == "remote":
            # log to terminal with time(real time)
            t = time()
            t = datetime.fromtimestamp(t).strftime('%Y-%m-%d %H:%M:%S')
            logger.info(f"Time: {t}, Epoch: {self.current_epoch}, Val Loss: {val_loss}, Val Acc 100: {acc_100}, Val Acc 90: {acc_90}, Val Acc 75: {acc_75}, Val Acc 50: {acc_50}, Val BLEU: {bleu}")
        self.val_results = {'pred': [], 'target': [], 'loss': []}
        

class LMModule(BaseLightningModule):
    """Pytorch lightning model for LM training."""

    def __init__(
        self,
        model_args: Dict[str, Union[float, int, str]],
    ) -> None:
        """Construct an LM lightning module.
        Args:
            model_args: model's arguments.
        """
        super().__init__(model_args)

        self.model: AutoModel
        self.tokenizer: AutoTokenizer

        self.cache_dir = None
        if "cache_dir" in model_args:
            self.cache_dir = model_args["cache_dir"]

        self.init_model()

    def init_model(self) -> None:
        """Initialize an AutoModel."""

        if self.model_args["model_name_or_path"] is not None:
            self.model = AutoModel.from_pretrained(
                self.model_args["model_name_or_path"],
                cache_dir=self.cache_dir,
            )
        else:
            config = AutoConfig.from_pretrained(
                self.model_args["model_config_name"], cache_dir=self.cache_dir
            )

            self.model = AutoModel.from_config(config)

            logger.info("Training from scratch")


class MLMModule(LMModule):
    """Pytorch lightning model for MLM training."""

    def init_model(self) -> None:
        """Initialize a MLM model."""

        if self.model_args["model_name_or_path"] is not None:
            self.model = AutoModelForMaskedLM.from_pretrained(
                self.model_args["model_name_or_path"], cache_dir=self.cache_dir
            )
        else:
            config = AutoConfig.from_pretrained(
                self.model_args["model_config_name"], cache_dir=self.cache_dir
            )

            self.model = AutoModelForMaskedLM.from_config(config)

            logger.info("Training from scratch")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_args["tokenizer"], use_fast=False
        )

        self.model.resize_token_embeddings(len(self.tokenizer))  # type: ignore


class CGMModule(LMModule):
    """Pytorch lightning model for conditional generation training."""

    def init_model(self) -> None:
        """Initialize a model for conditional generation."""

        if self.model_args["model_name_or_path"] is not None:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_args["model_name_or_path"],  # type:ignore
                cache_dir=self.cache_dir,
            )
        else:
            config = AutoConfig.from_pretrained(
                self.model_args["model_config_name"], cache_dir=self.cache_dir
            )

            self.model = AutoModelForSeq2SeqLM.from_config(config)

            logger.info("Training from scratch")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_args["tokenizer"], use_fast=False
        )

        self.model.resize_token_embeddings(len(self.tokenizer))  # type: ignore


class CLMModule(LMModule):
    """Pytorch lightning model for CLM training."""

    def init_model(self) -> None:
        """Initialize a CLM model."""

        if self.model_args["model_name_or_path"] is not None:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_args["model_name_or_path"], cache_dir=self.cache_dir
            )
        else:
            config = AutoConfig.from_pretrained(
                self.model_args["model_config_name"], cache_dir=self.cache_dir
            )

            self.model = AutoModelForCausalLM.from_config(config)

            logger.info("Training from scratch")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_args["tokenizer"],
            sep_token="<|sep|>",
            pad_token="<|pad|>",
            use_fast=False,
        )

        self.model.resize_token_embeddings(len(self.tokenizer))  # type: ignore


class PLMModule(LMModule):
    """Pytorch lightning model for PLM training."""

    def init_model(self) -> None:
        """Initialize a PLM model."""

        if self.model_args["model_name_or_path"] is not None:
            self.model = XLNetLMHeadModel.from_pretrained(
                self.model_args["model_name_or_path"],  # type:ignore
                cache_dir=self.cache_dir,
            )
        else:
            config = AutoConfig.from_pretrained(
                self.model_args["model_config_name"], cache_dir=self.cache_dir
            )

            self.model = XLNetLMHeadModel.from_config(config)

            logger.info("Training from scratch")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_args["tokenizer"], use_fast=False
        )

        self.model.resize_token_embeddings(len(self.tokenizer))  # type: ignore


LM_MODULE_FACTORY: Dict[str, Type[LMModule]] = {
    "lm": LMModule,
    "mlm": MLMModule,
    "clm": CLMModule,
    "cgm": CGMModule,
    "plm": PLMModule,
}
