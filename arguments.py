import typing
import transformers
from typing import Optional
from dataclasses import dataclass, field
from transformers import TrainingArguments as OriginalTrainingArguments


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model: str = field(
        default="BERT",
        metadata={"help": "Model name (BERT, BART, ALBERT, ... )"}
    )

    model_name_or_path: str = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )

    metric_base_model_name_or_path: str = field(
        default='gpt2',
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )

    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )

    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )

    cache_dir: Optional[str] = field(
        default=".cache", metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )

    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    train_mask_percentage: float = field(default=0.3, metadata={"help": "RanMask train mask rate."})
    infer_mask_percentage: float = field(default=0.3, metadata={"help": "RanMask inference mask rate."})
    ensemble_num: float = field(default=100, metadata={"help": "RanMask inference ensemble number."})
    ensemble_method: str = field(default="votes", metadata={"help": "RanMask inference ensemble method."})


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task_name: str = field(default="classification", metadata={"help": "The name of the task"})
    data_files: str = field(default="data_in", metadata={"help": "Should contain the data files for the task."})
    num_labels: int = field(default=2, metadata={"help": "The number of labels on dataset"})
    max_seq_length: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )


@dataclass
class TrainingArguments(OriginalTrainingArguments):
    
    do_train: bool = field(default=True, metadata={"help": "Whether to run training."})
    do_predict: bool = field(default=False, metadata={"help": "Whether to run predictions on the test set."})
    output_dir: str = field(
        default="data_out",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."}
    )
    logging_dir: Optional[str] = field(default="data_out", metadata={"help": "Tensorboard log dir."})
    eval_delay: Optional[float] = 0
    evaluation_strategy: typing.Union[transformers.trainer_utils.IntervalStrategy, str] = 'epoch'
    save_strategy: typing.Union[transformers.trainer_utils.IntervalStrategy, str] = 'epoch'
    logging_strategy: typing.Union[transformers.trainer_utils.IntervalStrategy, str] = 'epoch'
    lr_scheduler_type: typing.Union[transformers.trainer_utils.SchedulerType, str] = 'linear'