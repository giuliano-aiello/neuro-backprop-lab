import os
import torch


def save_model(model):
    if not os.path.exists('./tester'):
        os.makedirs('./tester')
    torch.save(model, './tester/trained_model.pt')
