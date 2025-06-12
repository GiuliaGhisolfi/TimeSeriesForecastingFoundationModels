from uni2ts.model.moirai import MoiraiFinetune, MoiraiForecast, MoiraiModule

MODEL_PATH = "Salesforce/moirai-1.0-R-small" # "Salesforce/moirai-1.0-R-base", "Salesforce/moirai-1.0-R-large"
DEVICE_MAP = "cuda"


# Load model from checkpoint
model = MoiraiForecast(
        module=MoiraiModule.from_pretrained(MODEL_PATH).to(DEVICE_MAP),
        prediction_length=prediction_length,
        context_length=4000,
        patch_size=32,
        num_samples=20,
        target_dim=target_dim,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
    )