import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel, AutoModelForSequenceClassification, Trainer, AutoModel, AutoModelForCausalLM
from typing import List, Optional, Tuple, Union
import pdb
import json
from transformers.modeling_outputs import SequenceClassifierOutput

class Disentangle_Layer(nn.Module):
    def __init__(self, input_dim = 768, latent_dim = 64, hidden_dim = 512):
        super(Disentangle_Layer, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.squeezer = nn.ModuleList([nn.Linear(self.input_dim, self.hidden_dim), nn.ReLU()])
        self.semantic_proj = nn.Linear(self.hidden_dim, self.latent_dim)
        self.perturbation_proj = nn.Linear(self.hidden_dim, 1)

    def forward(self, input):
        latent_rep = input
        for layer in self.squeezer:
            latent_rep = layer(latent_rep) # [B, T, D]
        senmantic_rep = self.semantic_proj(latent_rep)
        perturbation_log_rep = self.perturbation_proj(latent_rep)

        return senmantic_rep, perturbation_log_rep

class Reconstruction_Layer(nn.Module):
    def __init__(self, output_dim = 768, latent_dim = 64, hidden_dim = 512):
        super(Reconstruction_Layer, self).__init__()
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.recon_layers = nn.ModuleList([nn.Linear(self.latent_dim, self.hidden_dim), nn.ReLU(), nn.Linear(self.hidden_dim, self.output_dim)])

    def forward(self, latent):
        recon_rep = latent
        for layer in self.recon_layers:
            recon_rep = layer(recon_rep) # [B, T, D]
        return recon_rep
    
class Reconstruction_Network(nn.Module):
    def __init__(self, input_dim = 768, latent_dim = 64):
        super(Reconstruction_Network, self).__init__()
        
        self.encoder = Disentangle_Layer(input_dim, latent_dim)
        self.decoder = Reconstruction_Layer(input_dim, latent_dim)


    def forward(self, input, beta = 0.5):
        senmantic_rep, perturbation_log_rep = self.encoder(input)

        noised_rep = self.gaussian_random_perturb(senmantic_rep, torch.exp(0.5 * perturbation_log_rep))
        output = self.decoder(noised_rep)
        mse_loss = self.recon_loss(output, input)
        reg_loss = self.regularization_loss(senmantic_rep, perturbation_log_rep)
        
        loss = mse_loss + beta * reg_loss # [B, T]
        return output, loss.mean()

    def gaussian_random_perturb(self, semantic_rep, perturbation_log_rep):
        gaussian_noise = torch.randn_like(perturbation_log_rep)
        return semantic_rep + gaussian_noise * perturbation_log_rep
    
    def recon_loss(self, output, input):
        return F.mse_loss(output, input, reduction="none").mean(dim = -1)
    
    def regularization_loss(self, semantic_rep, perturbation_log_rep, alpha = -1):
        return torch.mean(semantic_rep.pow(2) + perturbation_log_rep.exp() + alpha * perturbation_log_rep, dim=-1)


class ClassificationHead(nn.Module):
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
        x = features.max(dim = 1)[0]  
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class Calibrator(nn.Module):
    def __init__(self, symmetry=True):
        super(Calibrator, self).__init__()
        self.symmetry = symmetry
        self.kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=False)
    
    def forward(self, logits_p, logits_q):

        log_dist_p, log_dist_q = F.log_softmax(logits_p, dim=-1), F.log_softmax(logits_q, dim=-1)
        dist_p, dist_q = F.softmax(logits_p, dim=-1), F.softmax(logits_q, dim=-1)
        if self.symmetry:
            calib_loss = 0.5 * (self.kl_loss(log_dist_p, dist_q) + self.kl_loss(log_dist_q, dist_p))
        else:
            calib_loss = self.kl_loss(dist_p, dist_q)
        return calib_loss

class SCRNModel(PreTrainedModel):

    def __init__(self, model_name, config):
        super(SCRNModel, self).__init__(config)

        self.bert = AutoModel.from_pretrained(model_name, config=config)
        self.classifier = ClassificationHead(config=config)
        self.reconNN = Reconstruction_Network(input_dim = config.hidden_size, latent_dim = 512) 
    
  

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output = self.bert(input_ids = input_ids ,
                            attention_mask = attention_mask,
                            token_type_ids = token_type_ids,
                            position_ids = position_ids,
                            inputs_embeds = inputs_embeds,
                            output_attentions = output_attentions,
                            output_hidden_states = True,
                            return_dict = return_dict)
        last_hidden_state = output.last_hidden_state
        recon_output, recon_loss = self.reconNN(last_hidden_state)
        logits = self.classifier(recon_output)
        
        return SequenceClassifierOutput(
            loss=recon_loss,
            logits=logits,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
        )

class SCRNTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calibrator = Calibrator()

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        
        # siamese branch 1
        outputs_p = model.forward(**inputs)
        recon_loss_p, logits_p = outputs_p.loss.mean(), outputs_p.logits
        cls_loss_p = F.cross_entropy(logits_p, labels)

        # siamese branch 2
        outputs_q = model.forward(**inputs)
        recon_loss_q, logits_q = outputs_q.loss.mean(), outputs_q.logits
        cls_loss_q = F.cross_entropy(logits_q, labels)

        # cablibration
        calib_loss = self.calibrator(logits_p, logits_q)

        # final loss
        loss = 0.5 * (cls_loss_p + cls_loss_q) + 0.5 * calib_loss + 0.01 * (recon_loss_p + recon_loss_q)
        outputs = SequenceClassifierOutput(
            loss=loss,
            logits=logits_p,
            hidden_states=None,
            attentions=None,
        )
        return (loss, outputs) if return_outputs else loss