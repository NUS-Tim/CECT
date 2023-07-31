from cect import CECT


def model_sel(model_name, device):
    model_dict = {
        'CECT': CECT,
    }
    if model_name in model_dict:
        model = model_dict[model_name]()
        model = model.to(device)
        return model
    else:
        raise ValueError(f"Model {model_name} not found.")
