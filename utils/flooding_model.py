import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel, AutoModelForSequenceClassification, Trainer, AutoModel, AutoModelForCausalLM
from transformers.modeling_outputs import SequenceClassifierOutput

class FloodingTrainer(Trainer):
        
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        loss, logits = outputs.loss, outputs.logits
        loss = (loss - 0.15).abs() + 0.15
        
        outputs = SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )
        return (loss, outputs) if return_outputs else loss