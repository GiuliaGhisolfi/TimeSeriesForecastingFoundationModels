from src.uni2ts_predictor.model.moirai import MoiraiForecast, MoiraiModule
from src.uni2ts_predictor.model.moirai import MoiraiFinetune, MoiraiModule


def load_predictor(checkpoint: str, module: str, prediction_length: int=1, target_dim: int=1, 
    device_map: str="cuda"):
    """
    Load a model by its name.
    
    Args:
        name (str): The name of the model to load.
        
    Returns:
        object: The loaded model.
    """
    pretrained_module = MoiraiModule.from_pretrained(module)#.to(device_map)

    finetuned_model = MoiraiFinetune.load_from_checkpoint(
        checkpoint_path=checkpoint,
        module=pretrained_module,
        min_patches=16,
        min_mask_ratio=0.2,
        max_mask_ratio=0.5,
        max_dim=1024,
        num_training_steps=10000,
        num_warmup_steps=1000,
        module_kwargs=None,
        num_samples=100,
        beta1=0.9,
        beta2=0.98,
        lr=1e-7,
        weight_decay=1e-5,
        log_on_step=False,
    )#.to(device_map)

    model = MoiraiForecast(
        module=finetuned_model.module,
        prediction_length=prediction_length,
        context_length=4000,
        patch_size=32,
        num_samples=20,
        target_dim=target_dim,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
    )#.to(device_map)
    
    return model.create_predictor(batch_size=32)