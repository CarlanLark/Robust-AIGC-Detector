import argparse
import textattack
import pickle
import random
import torch
import numpy as np
import os
import json
from attack.sklearn_utils import CustomSklearnModelWrapper, CustomSklearnTokenizer
from textattack.models.wrappers import HuggingFaceModelWrapper
from attack import HuggingFaceModelMaskEnsembleWrapper
from transformers import AutoTokenizer, AutoModelForCausalLM
from textattack.attack_recipes import PWWSRen2019, Pruthi2019, DeepWordBugGao2018
from attack.attack_recipe import PWWSRen2019_threshold
from textattack import Attacker
from datasets import load_dataset
from utils.metric_utils import load_base_model_and_tokenizer
from utils.scrn_model import SCRNModel
from attack.custom_dataset import load_attack_dataset
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification

parser = argparse.ArgumentParser()
parser.add_argument('--model_type', type=str, default="hf") # hf/sklearn
parser.add_argument('--ensemble_num', type=int, default=1) 
parser.add_argument('--mask_percentage', type=float, default=0.30) 
parser.add_argument('--transfer_dataset_abbr', type=str, default="self")
parser.add_argument('--num_examples', type=int, default=10)
parser.add_argument('--attack_class', type=str, default="ai")
parser.add_argument('--attack_recipe', type=str, default="pwws")
parser.add_argument('--data_files', type=str, default="./data_in")
parser.add_argument('--output_dir', type=str, default="./data_out")
parser.add_argument('--bert_name_or_path', type=str, default="bert-base-uncased")
parser.add_argument('--metric_base_model_name_or_path', type=str, default="gpt2")
parser.add_argument('--seed', type=int, default=2020)
parser.add_argument('--log_summary', type=str, default='yes')
args = parser.parse_args()


random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


model_abbr, dataset_abbr = args.output_dir.split('/')[-1].split('_')
if args.transfer_dataset_abbr!= "self":
    dataset_abbr = args.transfer_dataset_abbr
args.data_files = args.data_files + '/' + dataset_abbr
# dataset
dataset = load_attack_dataset(data_files=args.data_files, attack_class=args.attack_class)


if args.model_type == 'hf':
    # load config and tokenizer
    config = AutoConfig.from_pretrained(args.output_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        args.output_dir,
        model_max_length=512,
        padding_side="right",
        use_fast=False,
    )
    # load model
    if model_abbr == 'scrn':
        model = SCRNModel(args.bert_name_or_path, config=config)
        model.load_state_dict(torch.load(os.path.join(args.output_dir,'pytorch_model.bin')))
    else:
        model = AutoModelForSequenceClassification.from_pretrained(args.output_dir, config=config)
    # select model_wrapper
    if args.ensemble_num > 1:
        model_wrapper = HuggingFaceModelMaskEnsembleWrapper(model, tokenizer, ensemble_num=args.ensemble_num, mask_percentage=args.mask_percentage)
    else:
        model_wrapper = HuggingFaceModelWrapper(model, tokenizer)
elif args.model_type =='sklearn':
    # model
    DEVICE = 'cuda'
    base_model, base_tokenizer = load_base_model_and_tokenizer(args.metric_base_model_name_or_path)
    base_model.to(DEVICE)
    tokenizer = CustomSklearnTokenizer(base_model, base_tokenizer, DEVICE, feature_fn=model_abbr)
    filename = args.output_dir + '/' + 'classifier.bin'
    # load the model from disk
    model = pickle.load(open(filename, 'rb'))
    model_wrapper = CustomSklearnModelWrapper(model, tokenizer)
else:
    raise ValueError('Unknown model type %s'%args.model_type)

if args.num_examples == -1:
    num_examples = len(dataset)
else:
    num_examples = args.num_examples

max_num_word_swaps = np.mean([len(x[0]['text'].split(' ')) for x in dataset][:num_examples]) // 20
if max_num_word_swaps >= 10:
    max_num_word_swaps = 10
elif max_num_word_swaps <= 1:
    max_num_word_swaps = 1
else:
    _ = 0

if args.attack_recipe == 'pwws': # word sub
    attack = PWWSRen2019.build(model_wrapper)
elif args.attack_recipe == 'pwwsTaip': # add threshold ai as positive
    # get threshold
    with open(f"{args.output_dir}/predict_results.json", "r") as fin:
        metrics = json.load(fin)
    if args.attack_class == "ai":
        target_max_score = metrics["eval_aip_threshold_chatgpt"]
    elif args.attack_class == "human":
        target_max_score = metrics["eval_aip_threshold_human"]
    else:
        raise ValueError('Unknown attack class %s'%args.attack_class)
    attack = PWWSRen2019_threshold.build(model_wrapper, target_max_score=target_max_score)
elif args.attack_recipe == 'pwwsThp': # add threshold human as positive
    with open(f"{args.output_dir}/predict_results.json", "r") as fin:
        metrics = json.load(fin)
    if args.attack_class == "ai":
        target_max_score = metrics["eval_hp_threshold_chatgpt"]
    elif args.attack_class == "human":
        target_max_score = metrics["eval_hp_threshold_human"]
    else:
        raise ValueError('Unknown attack class %s'%args.attack_class)
    attack = PWWSRen2019_threshold.build(model_wrapper, target_max_score=target_max_score)
elif args.attack_recipe == 'pruthi': # char sub delete insert etc
    attack = Pruthi2019.build(model_wrapper, max_num_word_swaps=max_num_word_swaps)
elif args.attack_recipe == 'deep-word-bug': # word sub, char sub, word del, word insert etc
    attack = DeepWordBugGao2018.build(model_wrapper)
else:
    raise ValueError('Unknown attack recipe %s'%args.attack_recipe)

attack_args = textattack.AttackArgs(
    num_examples=num_examples,
    log_to_csv='%s/attack_results_%s_%s_%s.csv'%(args.output_dir, dataset_abbr, args.attack_class, args.attack_recipe),
    csv_coloring_style='html', 
)
attacker = Attacker(attack, dataset, attack_args)
results = attacker.attack_dataset()
if args.log_summary == 'yes':
    attacker.attack_log_manager.add_output_file(filename="%s/attack_summary_%s_%s_%s.log"%(args.output_dir, dataset_abbr, args.attack_class, args.attack_recipe), color_method="file")
    attacker.attack_log_manager.log_summary()
