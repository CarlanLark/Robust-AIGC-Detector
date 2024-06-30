"""
HuggingFace Model Wrapper
--------------------------
"""
import os
import torch
import transformers

import textattack
import numpy as np
from textattack.models.wrappers import PyTorchModelWrapper

from typing import List, Tuple
from scipy.special import softmax
from sklearn.preprocessing import normalize
from torch import nn as nn
from transformers import PreTrainedTokenizer, AutoModelForMaskedLM, RobertaTokenizer

class HuggingFaceModelMaskEnsembleWrapper(PyTorchModelWrapper):
    """Loads a HuggingFace ``transformers`` model and tokenizer."""
    def __init__(self, model, tokenizer, mask_percentage=0.30, ensemble_num=3, ensemble_method="vote", batch_size=32):
        self.model = model
        self.tokenizer = tokenizer
        self.mask_percentage = mask_percentage
        self.ensemble_num = ensemble_num
        self.batch_size = batch_size
        self.ensemble_method = ensemble_method

    def __call__(self, text_input_list):
        """Passes inputs to HuggingFace models as keyword arguments.

        (Regular PyTorch ``nn.Module`` models typically take inputs as
        positional arguments.)
        """
        # Default max length is set to be int(1e30), so we force 512 to enable batching.
        max_length = (
            512
            if self.tokenizer.model_max_length == int(1e30)
            else self.tokenizer.model_max_length
        )
        # start ensemble
        ensemble_mask_text_input_list = self.ensemble_mask_tokens(text_input_list, 
                                                                    mask_percentage=self.mask_percentage,
                                                                    ensemble_num=self.ensemble_num, 
                                                                    mask_token=self.tokenizer.mask_token)
        outputs_list = []
        i = 0
        while i < len(ensemble_mask_text_input_list):
            batched_text_input_list = ensemble_mask_text_input_list[i : i + self.batch_size]
            inputs_dict = self.tokenizer(
                batched_text_input_list,
                add_special_tokens=True,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            model_device = next(self.model.parameters()).device
            inputs_dict.to(model_device)

            with torch.no_grad():
                outputs = self.model(**inputs_dict)

            if isinstance(outputs[0], str):
                # HuggingFace sequence-to-sequence models return a list of
                # string predictions as output. In this case, return the full
                # list of outputs.
                outputs_list.append(outputs)
            else:
                # HuggingFace classification models return a tuple as output
                # where the first item in the tuple corresponds to the list of
                # scores for each input.
                outputs_list.append(outputs.logits) 
            i += self.batch_size
        # logits ensemble
        output_logits = torch.cat(outputs_list, dim=0).cpu().numpy() #[bsz, label_num]
        ensemble_logits_for_each_input = np.split(output_logits, indices_or_sections=len(text_input_list), axis=0)
        logits_list = []
        for logits in ensemble_logits_for_each_input:
            if self.ensemble_method == 'votes':
                voted_label = np.argmax(np.bincount(np.argmax(logits, axis=-1), minlength=logits.shape[-1]))
                voted_logits_array = logits[np.where(np.argmax(logits, axis=-1)==voted_label)[0]]
                voted_logits = np.mean(voted_logits_array, axis=0, keepdims=True) #[1, num_labels]
                logits_list.append(torch.from_numpy(voted_logits))
            else:
                avg_logits = np.mean(logits, axis=0, keepdims=True)
                logits_list.append(torch.from_numpy(avg_logits))
        return torch.cat(logits_list, dim=0).to(model_device)

    def get_grad(self, text_input):
        """Get gradient of loss with respect to input tokens.

        Args:
            text_input (str): input string
        Returns:
            Dict of ids, tokens, and gradient as numpy array.
        """
        if isinstance(self.model, textattack.models.helpers.T5ForTextToText):
            raise NotImplementedError(
                "`get_grads` for T5FotTextToText has not been implemented yet."
            )

        self.model.train()
        embedding_layer = self.model.get_input_embeddings()
        original_state = embedding_layer.weight.requires_grad
        embedding_layer.weight.requires_grad = True

        emb_grads = []

        def grad_hook(module, grad_in, grad_out):
            emb_grads.append(grad_out[0])

        emb_hook = embedding_layer.register_backward_hook(grad_hook)

        self.model.zero_grad()
        model_device = next(self.model.parameters()).device
        input_dict = self.tokenizer(
            [text_input],
            add_special_tokens=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        input_dict.to(model_device)
        predictions = self.model(**input_dict).logits

        try:
            labels = predictions.argmax(dim=1)
            loss = self.model(**input_dict, labels=labels)[0]
        except TypeError:
            raise TypeError(
                f"{type(self.model)} class does not take in `labels` to calculate loss. "
                "One cause for this might be if you instantiatedyour model using `transformer.AutoModel` "
                "(instead of `transformers.AutoModelForSequenceClassification`)."
            )

        loss.backward()

        # grad w.r.t to word embeddings
        grad = emb_grads[0][0].cpu().numpy()

        embedding_layer.weight.requires_grad = original_state
        emb_hook.remove()
        self.model.eval()

        output = {"ids": input_dict["input_ids"], "gradient": grad}

        return output


    def _tokenize(self, inputs):
        """Helper method that for `tokenize`
        Args:
            inputs (list[str]): list of input strings
        Returns:
            tokens (list[list[str]]): List of list of tokens as strings
        """
        return [
            self.tokenizer.convert_ids_to_tokens(
                self.tokenizer([x], truncation=True)["input_ids"][0]
            )
            for x in inputs
        ]

    def ensemble_mask_tokens(self, strings, mask_percentage=0.3, ensemble_num=3, mask_token='<mask>'):
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