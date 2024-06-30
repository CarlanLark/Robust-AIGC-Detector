import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel, AutoModelForSequenceClassification, Trainer, AutoModel, AutoModelForCausalLM
from transformers.modeling_outputs import SequenceClassifierOutput


def compute_kl_loss(p, q, pad_mask=None):
    
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')
    
    # pad_mask is for seq-level tasks
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)

    # You can choose whether to use function "sum" and "mean" depending on your task
    p_loss = p_loss.mean()
    q_loss = q_loss.mean()

    loss = (p_loss + q_loss) / 2
    return loss

class RDropTrainer(Trainer):
        
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        loss_fct = nn.CrossEntropyLoss()

        logits = model(**inputs).logits
        logits2 = model(**inputs).logits

        ce_loss = 0.5 * (loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1)) + loss_fct(logits2.view(-1, self.model.config.num_labels), labels.view(-1)))
        kl_loss = compute_kl_loss(logits, logits2)
        loss = ce_loss + kl_loss
        
        outputs = SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )
        return (loss, outputs) if return_outputs else loss