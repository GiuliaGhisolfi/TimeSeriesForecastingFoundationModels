class ToTorch(Transformation):
    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        # Converte ogni array nel target in torch.Tensor
        data_entry["target"] = [torch.tensor(arr, dtype=torch.float32) for arr in data_entry["target"]]
        return data_entry
