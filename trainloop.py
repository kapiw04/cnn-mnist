import torch
from my_utils import eta
import wandb
from hyperparameters import DEVICE, NUM_EPOCHS

def init_wandb(project_name, config):
    wandb.init(project=project_name, config=config)
    wandb.config.update(config)


def train_model(model, loss_fn, optimizer, train_dataset: torch.utils.data.DataLoader, val_dataset: torch.utils.data.DataLoader, use_wandb=False):
    phases = {
        "train": train_dataset,
        "val": val_dataset,
    }
    model = model.to(DEVICE)

    for phase in phases.keys():
        for epoch in range(NUM_EPOCHS):
            if phase == "train":
                model.train()
                for i, (X, y) in enumerate(train_dataset):
                    X, y = X.to(DEVICE), y.to(DEVICE)
                    pred = model(X)
                    loss = loss_fn(pred, y)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    eta(epoch, i, NUM_EPOCHS, train_dataset)
                
            else:
                model.eval()
                total_val_loss = 0
                val_correct = 0
                total_val_samples = 0
                for i, (X, y) in enumerate(val_dataset):
                    with torch.no_grad():
                        X, y = X.to(DEVICE), y.to(DEVICE)
                        pred = model(X)
                        loss = loss_fn(pred, y)

                        total_val_loss += loss.item()
                        val_correct += (pred.argmax(1) == y).type(torch.float).sum().item()
                        total_val_samples += y.size(0)
                        if use_wandb:
                            wandb.log({
                                'val_loss': total_val_loss / total_val_samples,
                                'val_accuracy': val_correct / total_val_samples
                            }, step=epoch)

    return model.to(DEVICE)