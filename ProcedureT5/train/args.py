from dataclasses import dataclass, field
from typing import Optional, Union
from models import LM_MODULE_FACTORY

@dataclass
class TrainerArguments:
    """Trainer arguments."""

    __name__ = "trainer_base_args"

    configuration_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "Configuration file for the training in JSON format. It can be used to completely by-pass pipeline specific arguments."
        },
    )

@dataclass
class PytorchLightningTrainingArguments:
    """
    Arguments related to pytorch lightning trainer.
    """

    __name__ = "pl_trainer_args"

    accelerator: Optional[str] = field(
        default="auto",
        metadata={
            "help": "Training accelerator ('cpu', 'gpu', 'tpu', 'ipu', 'hpu', 'mps', 'auto')"
        },
    )
    strategy: Optional[str] = field(
        default="auto",
        metadata={
            "help": "Training strategy ('ddp', 'ddp_spawn', 'deepspeed', 'auto')"
        },
    )
    accumulate_grad_batches: int = field(
        default=1,
        metadata={
            "help": "Accumulates grads every k batches or as set up in the dict."
        },
    )
    val_check_interval: int = field(
        default=5000, metadata={"help": " How often to check the validation set."}
    )
    save_dir: Optional[str] = field(
        default="logs", metadata={"help": "Save directory for logs and output."}
    )
    basename: Optional[str] = field(
        default="lightning_logs", metadata={"help": "Experiment name."}
    )
    gradient_clip_val: float = field(
        default=0.0, metadata={"help": "Gradient clipping value."}
    )
    limit_val_batches: int = field(
        default=500, metadata={"help": "How much of validation dataset to check."}
    )
    log_every_n_steps: int = field(
        default=500, metadata={"help": "How often to log within steps."}
    )
    max_epochs: int = field(
        default=3,
        metadata={"help": "Stop training once this number of epochs is reached."},
    )
    dirpath: Optional[str] = field(
        default=None,
        metadata={"help": "Path/URL to save checkpoints"},
    )
    ckpt_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path/URL to to the chekpoint be resumed"},
    )
    # num_nodes: Optional[int] = field(
    #     default=-1,
    #     metadata={"help": "Number of devices (including gpus) to train on."},
    # )
    devices: Optional[int] = field(
        default=-1,
        metadata={"help": "Number of devices (including gpus) to train on."},
    )
    # Callbacks for saving checkpoints
    monitor: Optional[str] = field(
        default=None,
        metadata={"help": "Quantity to monitor in order to store a checkpoint."},
    )
    save_last: Optional[bool] = field(
        default=None,
        metadata={
            "help": "When True, always saves the model at the end of the epoch to a file last.ckpt"
        },
    )
    save_top_k: int = field(
        default=1,
        metadata={
            "help": "The best k models according to the quantity monitored will be saved."
        },
    )
    mode: str = field(
        default="min",
        metadata={"help": "Quantity to monitor in order to store a checkpoint."},
    )
    every_n_train_steps: Optional[int] = field(
        default=None,
        metadata={"help": "Number of training steps between checkpoints."},
    )
    every_n_epochs: Optional[int] = field(
        default=None,
        metadata={"help": "Number of epochs between checkpoints."},
    )

@dataclass
class LanguageModelingModelArguments:
    """
    Arguments pertaining to which model/config we are going to fine-tune, or train from scratch.
    """

    __name__ = "model_args"

    type: str = field(
        metadata={"help": "The language modeling type, for example mlm."},
    )
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization, for example bert-base-uncased."
        },
    )
    model_config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained config name or path."},
    )
    tokenizer: Optional[str] = field(
        default=None,
        metadata={
            "help": "Name of the tokenizer to be used, default: tokenizer of utilizing model."
        },
    )
    lr: float = field(
        default=2e-5,
        metadata={"help": "The learning rate."},
    )
    lr_decay: float = field(
        default=0.5,
        metadata={"help": "The learning rate decay."},
    )
    weight_decay: float = field(
        default=0.01,
        metadata={"help": "Weight decay (L2)."},
    )
    cache_dir: Union[str, None] = field(
        default=None,
        metadata={"help": "Cache directory for HF models."},
    )
    # Display options
    display_mode: str = field(
        default="local",
        metadata={"help": "Display mode for training. local or remote"},
    )
    # show_val_bleu: bool = field(
    #     default=False,
    #     metadata={"help": "Whether to show validation BLEU score."},
    # )
    # show_val_acc: bool = field(
    #     default=False,
    #     metadata={"help": "Whether to show validation accuracy."},
    # )
    train_loss_log_interval: int = field(
        default=500,
        metadata={"help": "Interval for logging training loss."},
    )

@dataclass
class LanguageModelingDataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    __name__ = "dataset_args"

    train_file: str = field(
        metadata={
            "help": "The input training data file (a text file), for example path/to/file."
        }
    )
    validation_file: str = field(
        metadata={
            "help": "The input evaluation data file to evaluate the perplexity on (a text file), for example path/to/file."
        },
    )
    max_length: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated."
        },
    )
    mlm_probability: float = field(
        default=0.15,
        metadata={"help": "Ratio of tokens to mask for masked language modeling loss"},
    )
    plm_probability: float = field(
        default=0.16666,
        metadata={
            "help": "Ratio of length of a span of masked tokens to surrounding context length for "
            "permutation language modeling."
        },
    )
    max_span_length: int = field(
        default=5,
        metadata={
            "help": "Maximum length of a span of masked tokens for permutation language modeling."
        },
    )
    batch_size: int = field(
        default=8,
        metadata={"help": "Ratio of tokens to mask for masked language modeling loss."},
    )

@dataclass
class LanguageModelingSavingArguments:
    """Saving arguments related to LM trainer."""

    __name__ = "saving_args"

    hf_model_path: str = field(
        metadata={"help": "Path to the converted HF model."},
        default="/tmp/gt4sd_lm_saving_tmp",
    )
    training_type: Optional[str] = field(
        metadata={
            "help": f"Training type of the converted model, supported types: {', '.join(LM_MODULE_FACTORY.keys())}."
        },
        default=None,
    )
    model_name_or_path: Optional[str] = field(
        metadata={
            "help": "Model name or path.",
        },
        default=None,
    )
    ckpt: Optional[str] = field(metadata={"help": "Path to checkpoint."}, default=None)
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Tokenizer name or path. If not provided defaults to model_name_or_path."
        },
    )

