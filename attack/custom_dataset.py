import textattack
import pandas as pd
import numpy as np
import json
import random
from datasets import load_dataset

def default_load_json(json_file_path, encoding='utf-8', **kwargs):
    with open(json_file_path, 'r', encoding=encoding) as fin:
        tmp_json = json.load(fin, **kwargs)
    return tmp_json

def dump_jsonline(json_file_path, data, encoding="utf-8"):
    with open(json_file_path, "wt", encoding=encoding) as fout:
        for ins in data:
            fout.write(f"{json.dumps(ins, ensure_ascii=False)}\n")
    fout.close()
    return 0

def load_attack_dataset(data_files, attack_class='ai'):
    dataset_abbr = data_files.split('/')[-1]
    if dataset_abbr in ["in-domain", "cross-domain", "cross-genre", "mixed-source"]: 
        # these datasets have been shuffled in train/test split
        data = load_dataset(
                    'json',
                    data_files={"train": data_files + "/train.json", 
                                "test": data_files + "/test.json", },
                )["test"]
        if attack_class == 'ai':
            dataset = []
            for x in data:
                if x['labels'] == 1:
                    dataset.append((x['text'], x['labels']))
        elif attack_class == 'human':
            dataset = []
            for x in data:
                if x['labels'] == 0:
                    dataset.append((x['text'], x['labels']))
        else:
            raise ValueError('Dataset not exist: %s'%data_files)
    else:
        raise ValueError('Attack class not exist: %s'%attack_class)
    
    return textattack.datasets.Dataset(dataset)