import numpy as np
import torch
import torch.nn.functional as F
import time
from utils.metric_utils import timeit, get_clf_results, cut_length, cal_metrics
from tqdm import tqdm


def get_ll(text, base_model, base_tokenizer, DEVICE):
    with torch.no_grad():
        if len(base_tokenizer.encode(text)) == 1:
            text += ' %s'%(base_tokenizer.pad_token)
        tokenized = base_tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt").to(DEVICE)
        labels = tokenized.input_ids
        return -base_model(**tokenized, labels=labels).loss.item()
        # https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py#L1317


def get_lls(texts, base_model, base_tokenizer, DEVICE):
    return [get_ll(_, base_model, base_tokenizer, DEVICE) for _ in texts]


# get the average rank of each observed token sorted by model likelihood
def get_rank(text, base_model, base_tokenizer, DEVICE, log=False):
    with torch.no_grad():
        if len(base_tokenizer.encode(text)) == 1:
            text += ' %s'%(base_tokenizer.pad_token)
        tokenized = base_tokenizer(
            text,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(DEVICE)
        logits = base_model(**tokenized).logits[:, :-1]
        labels = tokenized.input_ids[:, 1:]

        # get rank of each label token in the model's likelihood ordering
        matches = (logits.argsort(-1, descending=True)
                   == labels.unsqueeze(-1)).nonzero()

        assert matches.shape[
            1] == 3, f"Expected 3 dimensions in matches tensor, got {matches.shape}"

        ranks, timesteps = matches[:, -1], matches[:, -2]

        # make sure we got exactly one match for each timestep in the sequence
        assert (timesteps == torch.arange(len(timesteps)).to(
            timesteps.device)).all(), "Expected one match per timestep"

        ranks = ranks.float() + 1  # convert to 1-indexed rank
        if log:
            ranks = torch.log(ranks)

        return ranks.float().mean().item()


def get_ranks(texts, base_model, base_tokenizer, DEVICE, log=False):
    return [get_rank(_, base_model, base_tokenizer, DEVICE, log)
            for _ in texts]


def get_rank_GLTR(text, base_model, base_tokenizer, DEVICE, log=False):
    with torch.no_grad():
        if len(base_tokenizer.encode(text)) == 1:
            text += ' %s'%(base_tokenizer.pad_token)
        tokenized = base_tokenizer(
            text,
            truncation=True,
            max_length=512,
            return_tensors="pt").to(DEVICE)
        logits = base_model(**tokenized).logits[:, :-1]
        labels = tokenized.input_ids[:, 1:]

        # get rank of each label token in the model's likelihood ordering
        matches = (logits.argsort(-1, descending=True)
                   == labels.unsqueeze(-1)).nonzero()

        assert matches.shape[
            1] == 3, f"Expected 3 dimensions in matches tensor, got {matches.shape}"

        ranks, timesteps = matches[:, -1], matches[:, -2]

        # make sure we got exactly one match for each timestep in the sequence
        assert (timesteps == torch.arange(len(timesteps)).to(
            timesteps.device)).all(), "Expected one match per timestep"
        ranks = ranks.float()
        res = np.array([0.0, 0.0, 0.0, 0.0])
        for i in range(len(ranks)):
            if ranks[i] < 10:
                res[0] += 1
            elif ranks[i] < 100:
                res[1] += 1
            elif ranks[i] < 1000:
                res[2] += 1
            else:
                res[3] += 1
        if res.sum() > 0:
            res = res / res.sum()

        return res


# get average entropy of each token in the text
def get_entropy(text, base_model, base_tokenizer, DEVICE):
    with torch.no_grad():
        if len(base_tokenizer.encode(text)) == 1:
            text += ' %s'%(base_tokenizer.pad_token)
        tokenized = base_tokenizer(
            text,
            truncation=True,
            max_length=512,
            return_tensors="pt").to(DEVICE)
        logits = base_model(**tokenized).logits[:, :-1]
        neg_entropy = F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)
        return -neg_entropy.sum(-1).mean().item()


@timeit
def run_threshold_experiment(data, criterion_fn, name, logger=None):
    torch.manual_seed(0)
    np.random.seed(0)

    # get train data
    train_text = data['train']['text']
    train_label = data['train']['labels']
    t1 = time.time()
    train_criterion = [
        criterion_fn(
            train_text[idx]) for idx in tqdm(
            range(
                len(train_text)),
            desc="Train criterion")]
    x_train = np.array(train_criterion)

    y_train = np.array(train_label)

    test_text = data['test']['text']
    test_label = data['test']['labels']
    test_criterion = [
        criterion_fn(
            test_text[idx]) for idx in tqdm(
            range(
                len(test_text)),
            desc="Test criterion")]
    x_test = np.array(test_criterion)

    y_test = np.array(test_label)

    # remove nan values
    select_train_index = ~np.isnan(x_train)
    select_test_index = ~np.isnan(x_test)
    x_train = x_train[select_train_index]
    y_train = y_train[select_train_index]
    x_test = x_test[select_test_index]
    y_test = y_test[select_test_index]
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    # import pdb;pdb.set_trace()
    clf, train_res, test_res = get_clf_results(
        x_train, y_train, x_test, y_test)

    print('-----  train  -----')
    print(train_res)
    print('-----  test  -----')
    print(test_res)
    if logger:
        logger.info('-----  train  -----')
        logger.info(train_res)
        logger.info('-----  test  -----')
        logger.info(test_res)

    return {
        'name': f'{name}_threshold',
        'predictions': {'train': train_criterion, 'test': test_criterion},
        'general_train': train_res, 
        'general_test': test_res, 
        'clf': clf
    }


@timeit
def run_threshold_experiment_multiple_test_length(
    clf, data, criterion_fn, name, lengths=[
        10, 20, 50, 100, 200, 500, -1]):
    torch.manual_seed(0)
    np.random.seed(0)
    res = {}
    for length in lengths:
        test_text = data['test']['text']
        test_label = data['test']['labels']
        test_criterion = [
            criterion_fn(
                cut_length(
                    test_text[idx],
                    length)) for idx in tqdm(
                range(
                    len(test_text)),
                desc="Test criterion")]
        x_test = np.array(test_criterion)
        y_test = np.array(test_label)

        # remove nan values
        select_test_index = ~np.isnan(x_test)
        x_test = x_test[select_test_index]
        y_test = y_test[select_test_index]
        x_test = np.expand_dims(x_test, axis=-1)

        y_test_pred = clf.predict(x_test)
        y_test_pred_prob = clf.predict_proba(x_test)
        y_test_pred_prob = [_[1] for _ in y_test_pred_prob]
        acc_test, precision_test, recall_test, f1_test, auc_test = cal_metrics(
            y_test, y_test_pred, y_test_pred_prob)
        test_res = acc_test, precision_test, recall_test, f1_test, auc_test

        print(f"{name} {length} acc_test: {acc_test}, precision_test: {precision_test}, recall_test: {recall_test}, f1_test: {f1_test}, auc_test: {auc_test}")
        res[length] = test_res
    
    return res


@timeit
def run_GLTR_experiment(data, criterion_fn, name, logger=None):
    torch.manual_seed(0)
    np.random.seed(0)

    train_text = data['train']['text']
    train_label = data['train']['labels']
    train_criterion = [criterion_fn(train_text[idx])
                       for idx in range(len(train_text))]
    x_train = np.array(train_criterion)
    y_train = train_label

    test_text = data['test']['text']
    test_label = data['test']['labels']
    test_criterion = [criterion_fn(test_text[idx])
                      for idx in range(len(test_text))]
    x_test = np.array(test_criterion)
    y_test = test_label

    clf, train_res, test_res = get_clf_results(
        x_train, y_train, x_test, y_test)

    print('-----  train  -----')
    print(train_res)
    print('-----  test  -----')
    print(test_res)
    if logger:
        logger.info('-----  train  -----')
        logger.info(train_res)
        logger.info('-----  test  -----')
        logger.info(test_res)

    return {
        'name': f'{name}_threshold',
        'predictions': {'train': train_criterion, 'test': test_criterion},
        'general_train': train_res, 
        'general_test': test_res, 
        'clf': clf
    }


@timeit
def run_GLTR_experiment_multiple_test_length(
    clf, data, criterion_fn, name, lengths=[
        10, 20, 50, 100, 200, 500, -1]):
    torch.manual_seed(0)
    np.random.seed(0)

    res = {}
    for length in lengths:
        test_text = data['test']['text']
        test_label = data['test']['labels']
        test_criterion = [
            criterion_fn(
                cut_length(
                    test_text[idx],
                    length)) for idx in tqdm(
                range(
                    len(test_text)),
                desc="Test criterion")]
        x_test = np.array(test_criterion)
        y_test = np.array(test_label)

        y_test_pred = clf.predict(x_test)
        y_test_pred_prob = clf.predict_proba(x_test)
        y_test_pred_prob = [_[1] for _ in y_test_pred_prob]
        acc_test, precision_test, recall_test, f1_test, auc_test = cal_metrics(
            y_test, y_test_pred, y_test_pred_prob)
        test_res = acc_test, precision_test, recall_test, f1_test, auc_test

        print(f"{name} {length} acc_test: {acc_test}, precision_test: {precision_test}, recall_test: {recall_test}, f1_test: {f1_test}, auc_test: {auc_test}")
        res[length] = test_res

    return res