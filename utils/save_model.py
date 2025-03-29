import os
import torch


def save_model(model, optimizer):
    if not os.path.exists('./tester'):
        os.makedirs('./tester')

    model_data = {
        'model': model,
        'optimizer': optimizer
    }

    torch.save(model_data, './tester/trained_model.pt')
