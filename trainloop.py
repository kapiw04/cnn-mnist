import torch
from my_utils import eta
import wandb
from hyperparameters import *

def init_wandb(project_name, config):
    wandb.init(
        project=project_name,
        config=config
    )


def train_model(model, loss_fn, optimizer, train, val):
    phases = {
        "train": train,
        "val": val,
    }
    model = model.to(DEVICE)

    for phase in phases.keys():
        for epoch in range(NUM_EPOCHS):
            if phase == "train":
                model.train()
                for i, (X, y) in enumerate(train):
                    X, y = X.to(DEVICE), y.to(DEVICE)
                    pred = model(X)
                    loss = loss_fn(pred, y)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    eta(epoch, i, NUM_EPOCHS, train)
                
            else:
                model.eval()
                total_val_loss = 0
                val_correct = 0
                total_val_samples = 0
                for i, (X, y) in enumerate(val):
                    with torch.no_grad():
                        X, y = X.to(DEVICE), y.to(DEVICE)
                        pred = model(X)
                        loss = loss_fn(pred, y)

                        total_val_loss += loss.item()
                        val_correct += (pred.argmax(1) == y).type(torch.float).sum().item()
                        total_val_samples += y.size(0)
                        wandb.log({
                            'val_loss': total_val_loss / total_val_samples,
                            'val_accuracy': val_correct / total_val_samples
                        }, step=epoch)

    return model.to(DEVICE)