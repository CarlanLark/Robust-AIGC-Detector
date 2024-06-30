from utils.utils import set_logger, path_checker, metrics_fn, compute_metrics

import torch
import numpy as np
import random
import pickle
import datetime
import json

from transformers import (AutoConfig, AutoModelForSequenceClassification, Trainer, HfArgumentParser, set_seed, 
AutoTokenizer, DataCollatorForSeq2Seq, AutoModelForCausalLM)

from arguments import ModelArguments, DataTrainingArguments, TrainingArguments
from datasets import load_dataset
from utils.scrn_model import SCRNModel, SCRNTrainer
from utils.metric_based import get_ll, get_rank, get_entropy, get_rank_GLTR, run_threshold_experiment, run_GLTR_experiment
from utils.metric_utils import load_base_model_and_tokenizer
from utils.flooding_model import FloodingTrainer
from utils.rdrop import RDropTrainer
from utils.ranmask_model import RanMaskModel
from utils.utils import mask_tokens

import wandb
import os

os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB__SERVICE_WAIT"] = "300"

class CustomDataCollatorForSeqCLS(DataCollatorForSeq2Seq):    
    def __call__(self, features, return_tensors=None): 
        if return_tensors is None:
            return_tensors = self.return_tensors

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        return features


def metrics_fn(outputs):
    y_true = outputs.label_ids
    y_pred = outputs.predictions.argmax(-1)
    y_score = torch.tensor(outputs.predictions).softmax(-1).numpy()[:, 1]
    return compute_metrics(y_true, y_pred, y_score)    

def main():
    supervised_model_list = ['bert-base', 'roberta-base', 'deberta-base', 'ChatGPT-Detector', 'flooding', 'rdrop', 'ranmask', 'scrn']
    metric_based_model_list = ["Log-Likelihood", "Rank", "Log-Rank", "Entropy", "GLTR"]

    # Get arguments
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_abbr = training_args.output_dir.split('/')[-1]
    dataset_abbr = data_args.data_files.split('/')[-1]
    training_args.output_dir = training_args.output_dir + '_' + dataset_abbr
    
    # Path check and set logger
    # path_checker(training_args)
    try:
        os.mkdir(training_args.output_dir)
    except:
        print('Output directory already exists: %s'%training_args.output_dir)
    logger = set_logger(training_args)

    # Set seed
    set_seed(training_args.seed)

    # Load dataset
    raw_dataset = load_dataset(
            'json',
            data_files={"train": data_args.data_files + "/train.json", 
                        "test": data_args.data_files + "/test.json", },
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    if model_abbr in supervised_model_list:
        # Load model
        config = AutoConfig.from_pretrained(model_args.model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            model_max_length=data_args.max_seq_length,
            padding_side="right",
            use_fast=False,
        )
        if model_abbr == 'scrn':
            model = SCRNModel(model_args.model_name_or_path, config)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(model_args.model_name_or_path, config=config)
        
        
        def preprocess_function_for_ranmask(examples):
            examples["text"] = mask_tokens(examples["text"], mask_token=tokenizer.mask_token)
            inputs = tokenizer(examples["text"], truncation=True)
            model_inputs = inputs
            return model_inputs
        
        def preprocess_function_for_seq_cls(examples):
            inputs = tokenizer(examples["text"], truncation=True)
            model_inputs = inputs
            return model_inputs
        
        if model_abbr == 'ranmask':
            train_data_preprocess_fn = preprocess_function_for_ranmask
            infer_data_preprocess_fn = preprocess_function_for_seq_cls
        else:
            train_data_preprocess_fn = preprocess_function_for_seq_cls
            infer_data_preprocess_fn = preprocess_function_for_seq_cls


        
        # Preprocess dataset
        train_dataset, test_dataset = raw_dataset["train"], raw_dataset["test"]

        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                train_data_preprocess_fn,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )
            test_dataset = test_dataset.map(
                infer_data_preprocess_fn,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on test dataset",
            )
        
        data_collator = CustomDataCollatorForSeqCLS(tokenizer, model=model, pad_to_multiple_of=8 if training_args.fp16 else None,)


        # Set trainer
        if model_abbr == 'scrn':
            trainer_fn = SCRNTrainer
        elif model_abbr == 'flooding':
            trainer_fn = FloodingTrainer
        elif model_abbr == 'rdrop':
            trainer_fn = RDropTrainer
        else:
            trainer_fn = Trainer
        trainer = trainer_fn(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            eval_dataset=test_dataset,
            compute_metrics=metrics_fn,
        )

        # Training
        if training_args.do_train:
            train_result = trainer.train()
            # trainer.save_state()
            trainer.save_model()

        # Predict
        if training_args.do_predict:
            if model_abbr == 'ranmask':
                config = AutoConfig.from_pretrained(training_args.output_dir)
                model = RanMaskModel.from_pretrained(training_args.output_dir)
                # set params for ensemble inference
                model.tokenizer = tokenizer
                model.mask_percentage = model_args.infer_mask_percentage
                model.ensemble_num = model_args.ensemble_num
                model.ensemble_method = model_args.ensemble_method
            elif model_abbr == 'scrn':
                config = AutoConfig.from_pretrained(training_args.output_dir)
                model = SCRNModel(model_args.model_name_or_path, config=config)
                model.load_state_dict(torch.load(os.path.join(training_args.output_dir,'pytorch_model.bin')))
            else:
                config = AutoConfig.from_pretrained(training_args.output_dir)
                model = AutoModelForSequenceClassification.from_pretrained(training_args.output_dir)
            trainer = trainer_fn(
                model=model,
                args=training_args,
                tokenizer=tokenizer,
                data_collator=data_collator,
                eval_dataset=test_dataset,
                compute_metrics=metrics_fn,
            )
            predict_results = trainer.evaluate()
            trainer.save_metrics("predict", predict_results)

    elif model_abbr in metric_based_model_list:
        DEVICE = 'cuda'
        START_DATE = datetime.datetime.now().strftime('%Y-%m-%d')
        START_TIME = datetime.datetime.now().strftime('%H-%M-%S-%f')

        # get generative model and set device
        # gpt-2
        base_model, base_tokenizer = load_base_model_and_tokenizer(model_args.metric_base_model_name_or_path)
        base_model.to(DEVICE)

        # build features

        def ll_criterion(text): return get_ll(text, base_model, base_tokenizer, DEVICE)

        def rank_criterion(text): return -get_rank(text, base_model, base_tokenizer, DEVICE, log=False)

        def logrank_criterion(text): return -get_rank(text, base_model, base_tokenizer, DEVICE, log=True)

        def entropy_criterion(text): return get_entropy(text, base_model, base_tokenizer, DEVICE)

        def GLTR_criterion(text): return get_rank_GLTR(text, base_model, base_tokenizer, DEVICE)
    
        outputs = []
        data = raw_dataset
        if model_abbr == "Log-Likelihood":
            outputs.append(run_threshold_experiment(data, ll_criterion, "likelihood", logger=logger))
        elif model_abbr == "Rank":
            outputs.append(run_threshold_experiment(data, rank_criterion, "rank", logger=logger))
        elif model_abbr == "Log-Rank":
            outputs.append(run_threshold_experiment(data, logrank_criterion, "log_rank", logger=logger))
        elif model_abbr == "Entropy":
            outputs.append(run_threshold_experiment(data, entropy_criterion, "entropy", logger=logger))
        elif model_abbr == "GLTR":
            outputs.append(run_GLTR_experiment(data, GLTR_criterion, "rank_GLTR", logger=logger))
        clf = outputs[0]['clf']
        filename = training_args.output_dir + '/classifier.bin'
        pickle.dump(clf, open(filename, 'wb'))
        # save metrics
        test_metrics = {'eval_%s'%k:v for k, v in outputs[0]['general_test'].items()}
        file_name = training_args.output_dir + '/predict_results.json'
        json.dump(test_metrics, open(file_name, 'w'))
    
    
    
    
    else:
        raise ValueError("Invalid model abbreviation")


if __name__ == "__main__":
    main()
