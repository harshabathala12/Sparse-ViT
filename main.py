import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from src.models.baseline_vit import build_baseline_vit
from src.models.sparse_vit import SparseViT
from src.train import train_one_epoch
from src.evaluate import evaluate
from src.utils import plot_accuracy, plot_head_importance
from ptflops import get_model_complexity_info
import matplotlib.pyplot as plt


# -----------------------------
# 1️⃣ Device setup
# -----------------------------
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {device}")

# -----------------------------
# 2️⃣ Dataset setup (CIFAR-10)
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

trainset = datasets.CIFAR10(root="data", train=True, download=False, transform=transform)
testset = datasets.CIFAR10(root="data", train=False, download=False, transform=transform)

trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

print(f"Dataset loaded successfully: {len(trainset)} training samples, {len(testset)} test samples.")

# -----------------------------
# 3️⃣ Model setup
# -----------------------------
print("\nInitializing models...")
baseline = build_baseline_vit(num_classes=10).to(device)
sparse = SparseViT(img_size=224, patch_size=16, num_classes=10,
                   embed_dim=192, depth=6, num_heads=6, K=32).to(device)

# -----------------------------
# ⚙️  Model Efficiency Stats
# -----------------------------
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

print("\n📊 Model Efficiency Comparison")
print("=" * 45)

macs_dense, _ = get_model_complexity_info(baseline, (3, 224, 224), as_strings=True,
                                          print_per_layer_stat=False, verbose=False)
macs_sparse, _ = get_model_complexity_info(sparse, (3, 224, 224), as_strings=True,
                                           print_per_layer_stat=False, verbose=False)

print(f"Dense ViT  → Params: {count_params(baseline):.2f}M | FLOPs: {macs_dense}")
print(f"Sparse ViT → Params: {count_params(sparse):.2f}M | FLOPs: {macs_sparse}")
print("=" * 45)

# -----------------------------
# 4️⃣ Training Loop
# -----------------------------
epochs = 3
base_accs, sparse_accs = [], []

opt1 = torch.optim.Adam(baseline.parameters(), lr=3e-4)
opt2 = torch.optim.Adam(sparse.parameters(), lr=3e-4)

for e in range(epochs):
    print(f"\nEpoch {e+1}/{epochs}")
    bacc, bloss = train_one_epoch(baseline, opt1, trainloader, device)
    sacc, sloss = train_one_epoch(sparse, opt2, trainloader, device)

    base_eval = evaluate(baseline, testloader, device)
    sparse_eval = evaluate(sparse, testloader, device)

    base_accs.append(base_eval)
    sparse_accs.append(sparse_eval)

    print(f"[Epoch {e+1}] Dense ViT: {base_eval:.2f}% | Sparse ViT: {sparse_eval:.2f}%")

# -----------------------------
# 5️⃣ Save Accuracy + Head Importance
# -----------------------------
plot_accuracy(base_accs, sparse_accs, "results/accuracy_plot.png")
plot_head_importance(sparse, "results/head_importance.png")

# -----------------------------
# 6️⃣ Attention Visualization (Fixed)
# -----------------------------
print("\nVisualizing sparse attention mask...")

sparse.eval()
with torch.no_grad():
    # 1️⃣ Load one test image
    sample_img, _ = testset[0]
    sample_img_batch = sample_img.unsqueeze(0).to(device)  # [1, 3, 224, 224]

    # 2️⃣ Convert image into patch embeddings
    x = sparse.patch_embed(sample_img_batch)  # [1, N, embed_dim]

    # 3️⃣ Add class + positional embeddings
    cls_token = sparse.cls_token.expand(x.shape[0], -1, -1)
    x = torch.cat((cls_token, x), dim=1)
    x = x + sparse.pos_embed
    x = sparse.pos_drop(x)

    # 4️⃣ Pass through first transformer block
    first_block = list(sparse.blocks)[0]
    qkv = first_block.attn.qkv(x)
    qkv = qkv.reshape(1, -1, 3, sparse.num_heads, sparse.embed_dim // sparse.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]

    # 5️⃣ Generate attention mask
    mask = first_block.attn.gate(q, k)[0].cpu().numpy()  # [N, N]

# Plot heatmap
plt.figure(figsize=(6, 5))
plt.imshow(mask, cmap='hot', interpolation='nearest')
plt.title("Sparse Attention Mask (First Block)")
plt.xlabel("Key Patches")
plt.ylabel("Query Patches")
plt.colorbar()
plt.tight_layout()
plt.savefig("results/sparse_attention_heatmap.png")
plt.close()

print("🖼 Saved Sparse Attention Heatmap → results/sparse_attention_heatmap.png")

# -----------------------------
# 7️⃣ Final Summary
# -----------------------------
print("\n✅ Training complete. Results saved in /results/")
print("\n📄 Final Summary Table")
print("=" * 65)
print(f"{'Model':<15}{'Params (M)':<15}{'FLOPs':<20}{'Accuracy (%)':<15}")
print("=" * 65)
print(f"{'Dense ViT':<15}{count_params(baseline):<15.2f}{macs_dense:<20}{base_accs[-1]:<15.2f}")
print(f"{'Sparse ViT':<15}{count_params(sparse):<15.2f}{macs_sparse:<20}{sparse_accs[-1]:<15.2f}")
print("=" * 65)
