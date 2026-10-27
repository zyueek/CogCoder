"""Core token-weighted supervised loss extracted from the CogCoder trainer.

The caller supplies token weights and masks prompt labels with -100.
"""
from torch.nn import CrossEntropyLoss
from transformers import Trainer


class WeightedLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):  # type: ignore[override]
        labels = inputs.pop("labels")
        weights = inputs.pop("weights", None)
        outputs = model(**inputs, use_cache=False)
        logits = outputs.logits

        if labels.device != logits.device:
            labels = labels.to(logits.device)
        if weights is not None and weights.device != logits.device:
            weights = weights.to(logits.device)

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss(reduction="none")
        flat_logits = shift_logits.view(-1, shift_logits.shape[-1])
        flat_labels = shift_labels.view(-1)
        loss_per_token = loss_fct(flat_logits, flat_labels)

        active_loss_mask = flat_labels != -100
        active_losses = loss_per_token[active_loss_mask]
        if weights is not None:
            shift_weights = weights[..., 1:].contiguous().view(-1)
            active_weights = shift_weights[active_loss_mask]
            loss = (active_losses * active_weights).sum() / (active_weights.sum() + 1e-9)
        else:
            loss = active_losses.mean()

        return (loss, outputs) if return_outputs else loss
