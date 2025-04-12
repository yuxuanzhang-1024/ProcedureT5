"""PyTorch Lightning training utilities."""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import sentencepiece as _sentencepiece
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# sentencepiece has to be loaded before lightning to avoid segfaults
_sentencepiece

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class PyTorchLightningTrainingPipeline:
    """PyTorch lightining training pipelines."""

    def train(  # type: ignore
        self,
        pl_trainer_args: Dict[str, Any],
        model_args: Dict[str, Union[float, str, int]],
        dataset_args: Dict[str, Union[float, str, int]],
    ) -> None:
        """Generic training function for PyTorch Lightning-based training.
        Args:
            pl_trainer_args: pytorch lightning trainer arguments passed to the configuration.
            model_args: model arguments passed to the configuration.
            dataset_args: dataset arguments passed to the configuration.
        # """
        ckpt_path = pl_trainer_args["ckpt_path"] if "ckpt_path" in list(pl_trainer_args.keys()) else None
        logger.info(f"Trainer arguments: {pl_trainer_args}")

        # if pl_trainer_args["dirpath"] is not None and not pl_trainer_args[
        #     "dirpath"
        # ].endswith(".ckpt"):
        #     pl_trainer_args["dirpath"] = None

        pl_trainer_args["callbacks"] = {
            "model_checkpoint_callback": {
                "dirpath": pl_trainer_args["dirpath"],
                "monitor": pl_trainer_args["monitor"],
                "save_top_k": pl_trainer_args["save_top_k"],
                "mode": pl_trainer_args["mode"],
                "every_n_train_steps": pl_trainer_args["every_n_train_steps"],
                "every_n_epochs": pl_trainer_args["every_n_epochs"],
                "save_last": pl_trainer_args["save_last"],
            }
        }

        if model_args['display_mode'] == 'remote':
            pl_trainer_args['enable_progress_bar'] = False
        
        del (
            pl_trainer_args["monitor"],
            pl_trainer_args["dirpath"],
            pl_trainer_args["save_top_k"],
            pl_trainer_args["mode"],
            pl_trainer_args["every_n_train_steps"],
            pl_trainer_args["save_last"],
            pl_trainer_args["every_n_epochs"],
            pl_trainer_args["ckpt_path"],
        )

        pl_trainer_args["callbacks"] = self.add_callbacks(pl_trainer_args["callbacks"])

        pl_trainer_args["logger"] = TensorBoardLogger(
            pl_trainer_args["save_dir"], name=pl_trainer_args["basename"]
        )
        del (pl_trainer_args["save_dir"], pl_trainer_args["basename"])

        trainer = Trainer(**pl_trainer_args)
        data_module, model_module = self.get_data_and_model_modules(
            model_args, dataset_args
        )
        model_module.train_total_steps = int(len(data_module.train_dataloader())/2)
        trainer.fit(model_module, data_module, ckpt_path=ckpt_path) if ckpt_path else trainer.fit(model_module, data_module)
        
    def get_data_and_model_modules(
        self,
        model_args: Dict[str, Union[float, str, int]],
        dataset_args: Dict[str, Union[float, str, int]],
    ) -> Tuple[LightningDataModule, LightningModule]:
        """Get data and model modules for training.
        Args:
            model_args: model arguments passed to the configuration.
            dataset_args: dataset arguments passed to the configuration.
        Returns:
            the data and model modules.
        """
        raise NotImplementedError(
            "Can't get data and model modules for an abstract training pipeline."
        )

    def add_callbacks(self, callback_args: Dict[str, Any]) -> List[Any]:
        """Create the requested callbacks for training.
        Args:
            callback_args: callback arguments passed to the configuration.
        Returns:
            list of pytorch lightning callbacks.
        """

        callbacks: List[Any] = []
        if "early_stopping_callback" in callback_args:
            callbacks.append(EarlyStopping(**callback_args["early_stopping_callback"]))

        if "model_checkpoint_callback" in callback_args:
            callbacks.append(
                ModelCheckpoint(**callback_args["model_checkpoint_callback"])
            )

        return callbacks

