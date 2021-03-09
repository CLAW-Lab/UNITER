import math
from collections import defaultdict

import torch
from torch import nn

from .pretrain_vcr import UniterForPretrainingForVCR


def _gelu_python(x):
    """
    Original Implementation of the GELU activation function in Google BERT repo when initially
    created. For information: OpenAI GPT's GELU is slightly different (and gives slightly
    different results): 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 *
    torch.pow(x, 3)))) This is now written in C in torch.nn.functional Also see the Gaussian
    Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class ElectraLoss(nn.Module):
    def __init__(self, loss_weights=(1.0, 50.0)):
        super().__init__()
        self.loss_weights = loss_weights
        self.gen_loss_fn = nn.CrossEntropyLoss(reduction='none')
        self.disc_loss_fn = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, gen_logits, disc_logits, attention_mask, gt_labels,
                is_masked, is_replaced):
        # import pdb; pdb.set_trace()
        # gen_loss = self.gen_loss_fn(gen_logits, gt_labels[is_masked])
        # disc_logits = disc_logits.masked_select(attention_mask)
        # is_replaced = is_replaced.masked_select(attention_mask)
        disc_loss = self.disc_loss_fn(disc_logits.half(), is_replaced.half())
        # return gen_loss * self.loss_weights[0] + disc_loss * self.loss_weights[1]
        return disc_loss


class ElectraDiscriminatorPredictions(nn.Module):
    """Prediction module for the discriminator, made up of two dense layers."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense_prediction = nn.Linear(config.hidden_size, 1)
        self.config = config

    def forward(self, discriminator_hidden_states):
        hidden_states = self.dense(discriminator_hidden_states)
        hidden_states = _gelu_python(hidden_states)
        logits = self.dense_prediction(hidden_states).squeeze(-1)
        return logits


class UniterDiscriminatorForElectraPretraining(UniterForPretrainingForVCR):
    def __init__(self, config, img_dim, img_label_dim):
        super().__init__(config, img_dim, img_label_dim)
        self.mlm_discriminator_predictions = ElectraDiscriminatorPredictions(config)
        self.mrfr_discriminator_predictions = ElectraDiscriminatorPredictions(config)

    def forward_mlm(self, input_ids, position_ids, txt_type_ids, img_feat,
                    img_pos_feat, attention_mask, gather_index,
                    txt_labels, compute_loss=True):
        sequence_output = self.uniter(input_ids, position_ids,
                                      img_feat, img_pos_feat,
                                      attention_mask, gather_index,
                                      output_all_encoded_layers=False,
                                      txt_type_ids=txt_type_ids)
        sequence_output = sequence_output[:, :input_ids.size(1), :]
        logits = self.mlm_discriminator_predictions(sequence_output)
        return logits

    def forward_mrfr(self, input_ids, position_ids, txt_type_ids,
                     img_feat, img_pos_feat,
                     attention_mask, gather_index, img_masks, img_mask_tgt,
                     feat_targets, compute_loss=True):
        sequence_output = self.uniter(input_ids, position_ids,
                                      img_feat, img_pos_feat,
                                      attention_mask, gather_index,
                                      output_all_encoded_layers=False,
                                      img_masks=img_masks,
                                      txt_type_ids=txt_type_ids)
        sequence_output = sequence_output[:, -img_masks.size(1):, :]
        logits = self.mrfr_discriminator_predictions(sequence_output)
        return logits


class UniterForElectraPretraining(nn.Module):
    def __init__(self, config, ckpt, img_dim, img_label_dim):
        super(UniterForElectraPretraining, self).__init__()
        self.generator = UniterForPretrainingForVCR.from_pretrained(
            config, ckpt, img_dim, img_label_dim)
        self.discriminator = UniterDiscriminatorForElectraPretraining.from_pretrained(
            config, ckpt, img_dim, img_label_dim)
        self.gumbel_dist = torch.distributions.gumbel.Gumbel(0., 1.)
        self.electra_loss = ElectraLoss()

    def forward(self, batch, task, compute_loss=True):
        batch = defaultdict(lambda: None, batch)
        input_ids = batch['input_ids']
        position_ids = batch['position_ids']
        img_feat = batch['img_feat']
        img_pos_feat = batch['img_pos_feat']
        attention_mask = batch['attn_masks']
        gather_index = batch['gather_index']
        txt_type_ids = batch['txt_type_ids']
        if task == 'mlm':
            with torch.no_grad():
                txt_labels = batch['txt_labels']  # (B, L)
                is_mlm_applied = (txt_labels != -1)  # -1 if masking is not applied
                gen_logits = self.generator.forward_mlm(
                    input_ids, position_ids, txt_type_ids, img_feat,
                    img_pos_feat, attention_mask, gather_index, txt_labels, False)
                gen_tokens = (gen_logits.float() + self.gumbel_dist.sample(
                    gen_logits.shape).to("cuda")).argmax(dim=-1)  # (B, num_mlm_pos)
                generated = input_ids.clone()
                generated[txt_labels != -1] = gen_tokens  # (B, L)

                is_replaced = is_mlm_applied.clone()
                is_replaced[is_mlm_applied] = (gen_tokens != txt_labels[is_mlm_applied])
            disc_logits = self.discriminator.forward_mlm(
                generated, position_ids, txt_type_ids, img_feat, img_pos_feat,
                attention_mask, gather_index, is_replaced, False)
            if compute_loss:
                return self.electra_loss.forward(
                    gen_logits, disc_logits, attention_mask, txt_labels,
                    is_mlm_applied, is_replaced)
            else:
                return disc_logits
        elif task == 'mrfr':
            with torch.no_grad():
                img_mask_tgt = batch['img_mask_tgt']
                img_masks = batch['img_masks']
                mrfr_feat_target = batch['feat_targets']
                mrfr_gen_feats = self.generator.forward_mrfr(
                    input_ids, position_ids, txt_type_ids, img_feat,
                    img_pos_feat, attention_mask, gather_index, img_masks,
                    img_mask_tgt, mrfr_feat_target, False)
                gen_feats = img_feat.clone()
                gen_feats[img_masks] = mrfr_gen_feats
                is_replaced = img_masks.clone()
            mrfr_disc_logits = self.discriminator.forward_mrfr(
                input_ids, position_ids, txt_type_ids, img_feat, img_pos_feat,
                attention_mask, gather_index, img_masks, img_mask_tgt,
                mrfr_feat_target, False)
            if compute_loss:
                return self.electra_loss.forward(
                    mrfr_gen_feats, mrfr_disc_logits, attention_mask, img_masks,
                    img_feat, is_replaced)
            else:
                return mrfr_disc_logits
        elif task == 'itm':
            # No Change: ITM Objective is already discriminative
            targets = batch['targets']
            ot_inputs = batch['ot_inputs']
            return self.discriminator.forward_itm(
                input_ids, position_ids, img_feat, img_pos_feat, attention_mask,
                gather_index, targets, ot_inputs, compute_loss)
        elif task.startswith('mrc'):
            # No change: MRC objective is already discriminative
            img_mask_tgt = batch['img_mask_tgt']
            img_masks = batch['img_masks']
            mrc_label_target = batch['label_targets']
            return self.discriminator.forward_mrc(
                input_ids, position_ids, img_feat, img_pos_feat, attention_mask,
                gather_index, img_masks, img_mask_tgt, mrc_label_target, task,
                compute_loss)
        else:
            raise ValueError('invalid task')
