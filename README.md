# SparseViT: Sparse Training from Scratch 🧠⚡

This project implements a **Sparse Vision Transformer (ViT)** trained from scratch with **adaptive attention and dynamic head pruning** — achieving higher efficiency and accuracy without post-training compression.

---

## 👥 Team Project
This project was developed as a collaborative team effort.

**Contributors:**
- **Harsha Bathala**
- Omryuo

This repository hosts the same project on my GitHub profile with full credit to all contributors.

## 🚀 Features
- Dynamic Top-K Sparse Attention via Gumbel-Softmax
- Continuous Attention Head Importance (L1 regularization)
- Progressive Sparsification during training
- Visualizations:
  - Accuracy vs Epochs
  - Attention Head Importance
  - Sparse Attention Heatmaps
- CIFAR-10/100 ready (plug-and-play)

---

## 🧱 Project Structure

```bash
sparse-vit/
│
├── main.py
├── requirements.txt
├── README.md
├── data/                      # (contains CIFAR-10 if needed)
├── results/                   # accuracy_plot.png, heatmaps, etc.
│
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baseline_vit.py
│   │   └── sparse_vit.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
│
└── venv/                      # (optional, ignored in .gitignore)
```

---

## ⚙️ Installation

```bash
git clone https://github.com/<your-username>/sparse-vit.git
cd sparse-vit
python -m venv venv
source venv/bin/activate   # (Mac/Linux)
# or venv\Scripts\activate # (Windows)

pip install -r requirements.txt
```

---

## 🧠 Run the Prototype

```bash
python main.py
```

### Expected Output:

```bash
Dense ViT  → Params: 5.53M | FLOPs: 912.7 MMac | Acc: 50.21%
Sparse ViT → Params: 2.86M | FLOPs: 557.31 MMac | Acc: 53.30%
```

### Results will be saved in /results/:

- accuracy_plot.png

- head_importance.png

- sparse_attention_heatmap.png

---

📊 Results Summary

Model	Params (M)	FLOPs (MMac)	Accuracy (%)
Dense ViT	5.53	912.7	50.21
Sparse ViT (ours)	2.86	557.3	53.30

---

🧩 Future Work

- Train on ImageNet for large-scale validation

- Adaptive K per layer

- On-device deployment (Jetson / mobile)

---

## 🔗 Related Repository
Original team repository:
https://github.com/Omryuo/Sparse-ViT

---

## 📜 License

This project is licensed under the MIT License.

---
# Thank you!!!

**Suggestion and Contributions are always welcome!** <br> Please fork the repository and create a pull request with your changes.
