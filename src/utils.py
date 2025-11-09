import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_accuracy(baseline_accs, sparse_accs, save_path):
    plt.plot(baseline_accs, label='Dense ViT')
    plt.plot(sparse_accs, label='Sparse ViT')
    plt.xlabel("Epochs"); plt.ylabel("Accuracy (%)")
    plt.title("Accuracy Comparison")
    plt.legend(); plt.savefig(save_path); plt.close()

def plot_head_importance(model, save_path):
    scores = model.blocks[0].attn.head_scores.detach().cpu().numpy()
    plt.bar(range(len(scores)), scores)
    plt.xlabel("Head"); plt.ylabel("Importance")
    plt.title("Attention Head Importance")
    plt.savefig(save_path); plt.close()
