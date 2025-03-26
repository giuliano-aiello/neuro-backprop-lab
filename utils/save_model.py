import os
import torch


def save_model(model):
    if not os.path.exists('./test'):
        os.makedirs('./test')
    torch.save(model, './test/trained_model.pt')
