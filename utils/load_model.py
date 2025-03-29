import torch
import os


def load_model():
    model_path = './tester/trained_model.pt'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    model_data = torch.load(model_path, weights_only=False)

    model = model_data['model']
    optimizer = model_data['optimizer'] # Optimizer is just a string with logging purposes.

    return model, optimizer
