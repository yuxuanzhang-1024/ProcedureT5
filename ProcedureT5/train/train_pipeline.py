
"""Language modeling training utilities."""

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Union

import sentencepiece as _sentencepiece
from pytorch_lightning import LightningDataModule, LightningModule

from datamodules import CGMDataModule, CLMDataModule, MLMDataModule, PLMDataModule
from models import LM_MODULE_FACTORY, CGMModule, CLMModule, MLMModule, PLMModule
from pl_trainer import PyTorchLightningTrainingPipeline

# sentencepiece has to be loaded before lightning to avoid segfaults
_sentencepiece

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class LanguageModelingTrainingPipeline(PyTorchLightningTrainingPipeline):
    """Language modeling training pipelines."""

    def get_data_and_model_modules(
        self,
        model_args: Dict[str, Union[float, str, int]],
        dataset_args: Dict[str, Union[float, str, int]],
        **kwargs,
    ) -> Tuple[LightningDataModule, LightningModule]:
        """Get data and model modules for training.
        Args:
            model_args: model arguments passed to the configuration.
            dataset_args: dataset arguments passed to the configuration.
        Returns:
            the data and model modules.
        """

        if (
            model_args["model_config_name"] is None
            and model_args["model_name_or_path"] is None
        ):
            raise ValueError("Model config or model name/path should be provided")

        if (
            model_args["model_config_name"] is not None
            and model_args["model_name_or_path"] is not None
        ):
            logger.warning(
                "Config name is omitted. Start fine-tuning using {}".format(
                    model_args["model_name_or_path"]
                )
            )

        if model_args["tokenizer"] is None:
            if model_args["model_name_or_path"] is not None:
                model_args["tokenizer"] = model_args["model_name_or_path"]
            else:
                model_args["tokenizer"] = model_args["model_config_name"]
                logger.warning(
                    "{} tokenizer is going to be used in the training".format(
                        model_args["tokenizer"]
                    )
                )

        logger.info(f"Model arguments: {model_args}")
        logger.info(f"Dataset arguments: {dataset_args}")

        if model_args["type"] == "mlm":
            data_module, model_module = self.get_mlm_modules(model_args, dataset_args)
        elif model_args["type"] == "clm":
            data_module, model_module = self.get_clm_modules(model_args, dataset_args)  # type: ignore
        elif model_args["type"] == "plm":
            data_module, model_module = self.get_plm_modules(model_args, dataset_args)  # type: ignore
        elif model_args["type"] == "cgm":
            data_module, model_module = self.get_cgm_modules(model_args, dataset_args)  # type: ignore
        else:
            raise ValueError(f"LM training type {model_args['type']} not supported")

        model_module.model.resize_token_embeddings(len(data_module.tokenizer))  # type: ignore

        return data_module, model_module

    def get_mlm_modules(
        self,
        model_args: Dict[str, Union[float, str, int]],
        dataset_args: Dict[str, Union[float, str, int]],
    ) -> Tuple[MLMDataModule, MLMModule]:
        """Get model and data module for clm.
        Args:
            model_args: dictionary containing all the parameters for the mode configuration.
            dataset_args: dictionary containing all the necessary parameters for the dataset creation.
        Returns:
            model and data module for clm.
        """

        model_module = MLMModule(model_args)
        data_module = MLMDataModule(dataset_args, tokenizer=model_module.tokenizer)

        return data_module, model_module

    def get_clm_modules(
        self,
        model_args: Dict[str, Union[float, str, int]],
        dataset_args: Dict[str, Union[float, str, int]],
    ) -> Tuple[CLMDataModule, CLMModule]:
        """Get model and data module for clm.
        Args:
            model_args: dictionary containing all the parameters for the mode configuration.
            dataset_args: dictionary containing all the necessary parameters for the dataset creation.
        Returns:
            model and data module for clm.
        """

        model_module = CLMModule(model_args)
        data_module = CLMDataModule(dataset_args, tokenizer=model_module.tokenizer)

        return data_module, model_module

    def get_plm_modules(
        self,
        model_args: Dict[str, Union[float, str, int]],
        dataset_args: Dict[str, Union[float, str, int]],
    ) -> Tuple[PLMDataModule, PLMModule]:
        """Get model and data module for plm.
        Args:
            model_args: dictionary containing all the parameters for the mode configuration.
            dataset_args: dictionary containing all the necessary parameters for the dataset creation.
        Returns:
            model and data module for plm.
        """

        model_module = PLMModule(model_args)
        data_module = PLMDataModule(dataset_args, tokenizer=model_module.tokenizer)

        return data_module, model_module

    def get_cgm_modules(
        self,
        model_args: Dict[str, Union[float, str, int]],
        dataset_args: Dict[str, Union[float, str, int]],
    ) -> Tuple[CGMDataModule, CGMModule]:
        """Get model and data module for Conditional Generation model.
        Args:
            model_args: dictionary containing all the parameters for the mode configuration.
            dataset_args: dictionary containing all the necessary parameters for the dataset creation.
        Returns:
            model and data module for plm.
        """

        model_module = CGMModule(model_args)
        data_module = CGMDataModule(
            dataset_args, model=model_module.model, tokenizer=model_module.tokenizer
        )

        return data_module, model_module

