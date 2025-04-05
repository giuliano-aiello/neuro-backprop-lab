import os
import torch
from utils.config_loader import get_optimizer_name


def delete_existing_model(directory='./tester', extension='.pt'):
    for f in os.listdir(directory):
        if f.endswith(extension):
            os.remove(os.path.join(directory, f))


def save_model(model, optimizer, n_epoch, eval_loss):
    if not os.path.exists('./tester'):
        os.makedirs('./tester')

    model_data = {
        'model': model,
        'optimizer': optimizer,
        'n_epoch': n_epoch,
        'eval_loss': eval_loss
    }

    delete_existing_model('./tester', '.pt')

    filename = f"{get_optimizer_name(optimizer.__class__.__name__)}_epoch{n_epoch + 1}_eval-loss{eval_loss:.4f}.pt"
    torch.save(model_data, os.path.join('./tester', filename))
