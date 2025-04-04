import torch
import os


def load_model():
    model_path = find_pt_file()

    model_data = torch.load(model_path, weights_only=False)

    model = model_data['model']
    optimizer = model_data['optimizer']

    return model, optimizer


def find_pt_file(directory='./tester', extension='.pt'):
    pt_files = [f for f in os.listdir(directory) if f.endswith(extension)]
    if len(pt_files) != 1:
        raise FileNotFoundError(f"Expected exactly one {extension} file in {directory}, found: {pt_files}")
    model_path = os.path.join(directory, pt_files[0])

    return model_path
