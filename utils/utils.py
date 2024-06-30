import logging
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

def set_logger(training_args):

    # Setup logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
        handlers=[logging.FileHandler(training_args.output_dir + "/train.log", 'w', encoding='utf-8'),
                  logging.StreamHandler()]
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    return logger


def path_checker(training_args):
    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    if not os.path.exists(training_args.logging_dir):
        os.mkdir(training_args.logging_dir)


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def metrics_fn(p):
    preds = np.argmax(p.predictions, axis=1)
    return {"acc": simple_accuracy(preds, p.label_ids)}


def prediction(logit):
    return np.argmax(logit, axis=1)

def find_thres(fpr, thresholds, target_fpr=0.01):
    idx = 0
    while fpr[idx+1] <= target_fpr:
        idx += 1
    return {'fpr': fpr[idx], 'threshold': thresholds[idx]}

def compute_metrics(y_true, y_pred, y_score):
    clf_report = classification_report(y_true, y_pred, output_dict=True)
    auc = roc_auc_score(y_true, y_score)
    hp_fpr, hp_tpr, hp_thresholds = roc_curve(y_true, 1-y_score, pos_label=0)# human as positive samples
    hp_fpr_thres = find_thres(hp_fpr, hp_thresholds, target_fpr=0.01)
    aip_fpr, aip_tpr, aip_thresholds = roc_curve(y_true, y_score, pos_label=1) # ai as positive samples
    aip_fpr_thres = find_thres(aip_fpr, aip_thresholds, target_fpr=0.01)
    # con_mat = confusion_matrix(y_true, preds)
    return {
        "AUC": auc,
        "hp_fpr": hp_fpr_thres['fpr'], 
        "hp_threshold_chatgpt": 1 - hp_fpr_thres['threshold'],
        "hp_threshold_human": hp_fpr_thres['threshold'], 
        "aip_fpr": aip_fpr_thres['fpr'], 
        "aip_threshold_chatgpt": aip_fpr_thres['threshold'],
        "aip_threshold_human": 1 - aip_fpr_thres['threshold'],
        "acc": clf_report['accuracy'],
        "precision_overall_weighted": clf_report['weighted avg']['precision'],
        "recall_overall_weighted": clf_report['weighted avg']['recall'],
        "fscore_overall_weighted": clf_report['weighted avg']['f1-score'],
        "precision_chatgpt": clf_report['1']['precision'],
        "recall_chatgpt": clf_report['1']['recall'],
        "fscore_chatgpt": clf_report['1']['f1-score'],
        "support_chatgpt": clf_report['1']['support'],
        "precision_human": clf_report['0']['precision'],
        "recall_human": clf_report['0']['recall'],
        "fscore_human": clf_report['0']['f1-score'],
        "support_human": clf_report['0']['support'],
        # "confusion_matrix": con_mat.tolist()
    }

def mask_tokens(strings, mask_percentage=0.3, mask_token='<mask>'):
    masked_strings = []

    for string in strings:
        tokens = np.array(string.split())
        num_tokens = len(tokens)
        num_masked_tokens = int(num_tokens * mask_percentage)

        masked_indices = np.random.choice(num_tokens, num_masked_tokens, replace=False)
        masked_tokens = np.where(np.isin(np.arange(num_tokens), masked_indices), mask_token, tokens)
        masked_string = ' '.join(masked_tokens)
        masked_strings.append(masked_string)

    return masked_strings

def ensemble_mask_tokens(strings, mask_percentage=0.3, ensemble_num=3, mask_token='<mask>'):
    """
    strings: (list[str]): List of strings
    Returns: (list[str]): List of strings
    """
    masked_strings = []
    for string in strings:
        for iter_idx in range(ensemble_num):
            tokens = np.array(string.split())
            num_tokens = len(tokens)
            num_masked_tokens = int(num_tokens * mask_percentage)

            masked_indices = np.random.choice(num_tokens, num_masked_tokens, replace=False)
            masked_tokens = np.where(np.isin(np.arange(num_tokens), masked_indices), mask_token, tokens).tolist()
            masked_string = ' '.join(masked_tokens)
            masked_strings.append(masked_string)
    return masked_strings