import os
import torch


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

    filename = f"./tester/trained-model_epoch{n_epoch}_eval-loss{eval_loss:.4f}.pt"
    torch.save(model_data, filename)
