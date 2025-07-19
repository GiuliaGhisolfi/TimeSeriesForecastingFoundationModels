from transformers import Trainer

class ChronosTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        target = inputs.pop("target_ids")  # recupera la ground truth
        outputs = model(**inputs)

        # Assumiamo che il modello abbia un metodo `compute_loss(preds, target)`
        # oppure che tu possa calcolare la loss manualmente
        if hasattr(model, "compute_loss"):
            loss = model.compute_loss(outputs, target)
        else:
            # fallback generico: adatta a come è fatto il tuo modello
            logits = outputs["prediction"] if isinstance(outputs, dict) else outputs
            loss_fn = torch.nn.MSELoss()  # adatta alla tua task
            loss = loss_fn(logits, target)

        return (loss, outputs) if return_outputs else loss
