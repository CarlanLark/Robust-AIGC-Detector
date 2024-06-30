import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel, RobertaPreTrainedModel, Trainer, AutoModel, RobertaModel
from typing import List, Optional, Tuple, Union
import pdb
from transformers.modeling_outputs import SequenceClassifierOutput
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from .utils import ensemble_mask_tokens
import numpy as np
from sklearn.preprocessing import normalize


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class RanMaskModel(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.classifier = RobertaClassificationHead(config)

        self.tokenizer = None
        self.infer_mask_percentage = 0.05
        self.ensemble_num = 5
        self.ensemble_method = "votes"

        # Initialize weights and apply final processing
        self.post_init()

    # ensemble forward
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # ensemble infer
        input_strings = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        ensemble_strings = ensemble_mask_tokens(input_strings, self.mask_percentage, self.ensemble_num, mask_token=self.tokenizer.mask_token)
        model_device = input_ids.device
        batch_size = 32#len(input_strings)
        i = 0
        ensemble_logits_list = []
        while i < len(ensemble_strings):
            batch_ensemble_strings = ensemble_strings[i:i+batch_size]
            batch_inputs = self.tokenizer(batch_ensemble_strings, return_tensors="pt", padding=True, truncation=True)
            batch_inputs = {key: value.to(model_device) for key, value in batch_inputs.items()}

            outputs = self.roberta(**batch_inputs, return_dict=return_dict,)
            sequence_output = outputs[0]
            logits = self.classifier(sequence_output)

            ensemble_logits_list.append(logits)
            i += batch_size
        ensemble_logits = torch.cat(ensemble_logits_list, dim=0).cpu().numpy() #[bsz, label_num]
        # get ensembled logits
        ensemble_logits_for_each_input = np.split(ensemble_logits, indices_or_sections=len(input_strings), axis=0)
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

        logits = torch.cat(logits_list, dim=0).to(model_device)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )

    