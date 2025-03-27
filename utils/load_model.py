import torch
import os


def load_model():
    model_path = './tester/trained_model.pt'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = torch.load(model_path, weights_only=False)

    return model