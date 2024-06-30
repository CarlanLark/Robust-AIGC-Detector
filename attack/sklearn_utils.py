import pickle
import textattack
from textattack.models.wrappers import SklearnModelWrapper
import numpy as np
from sklearn.linear_model import LogisticRegression
import sys
sys.path.append("..")
from utils.metric_based import get_ll, get_rank, get_entropy, get_rank_GLTR
from utils.metric_utils import cut_length
from transformers import AutoTokenizer, AutoModelForCausalLM

class CustomSklearnModelWrapper(SklearnModelWrapper):
    """
    subclass of SklearnModelWrapper
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def __call__(self, text_input_list, batch_size=None):
        x_test = self.tokenizer.transform(text_input_list)
        return self.model.predict_proba(x_test)

    def get_grad(self, text_input):
        raise NotImplementedError()

class CustomSklearnTokenizer(object):
    
    def __init__(self, base_model, base_tokenizer, device, feature_fn='Log-Likelihood', max_length=512):
        self.base_model = base_model
        self.base_tokenizer = base_tokenizer
        self.feature_fn = feature_fn
        self.max_length = max_length
        self.device = device
        
    def transform(self, text_list):
        
        if self.feature_fn == 'Log-Likelihood':
            x_test = [get_ll(cut_length(text,self.max_length), self.base_model, self.base_tokenizer, self.device) for text in text_list]
        elif self.feature_fn == 'Rank':
            x_test = [-get_rank(cut_length(text,self.max_length), self.base_model, self.base_tokenizer, self.device, log=False) for text in text_list]
        elif self.feature_fn == 'Log-Rank':
            x_test = [-get_rank(cut_length(text,self.max_length), self.base_model, self.base_tokenizer, self.device, log=True) for text in text_list]
        elif self.feature_fn == 'Entropy':
            x_test = [get_entropy(cut_length(text,self.max_length), self.base_model, self.base_tokenizer, self.device) for text in text_list]
        elif self.feature_fn == 'GLTR':
            x_test = [get_rank_GLTR(cut_length(text,self.max_length), self.base_model, self.base_tokenizer, self.device) for text in text_list]
        else:
            raise ValueError("Invalid feature function")

        x_test = np.array(x_test)
        if self.feature_fn in ["Log-Likelihood", "Rank", "Log-Rank", "Entropy"]:
            x_test = np.expand_dims(x_test, axis=-1)

        return x_test