import torch
import torch.nn.functional as F
from tqdm import tqdm

def train_one_epoch(model, optimizer, dataloader, device):
    model.train()
    total, correct, loss_sum = 0, 0, 0
    for imgs, labels in tqdm(dataloader, leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return 100 * correct / total, loss_sum / len(dataloader)
