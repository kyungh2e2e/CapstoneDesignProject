import math
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

# ---------------------------
# Utility Modules
# ---------------------------

def exists(val):
    return val is not None

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim)
        )
    def forward(self, x):
        return self.net(x)

class LinformerAttention(nn.Module):
    def __init__(
        self,
        dim,
        seq_len_q,
        k_max,
        seq_len_kv=None,
        heads=8,
        dim_head=64,
        dropout=0.1
    ):
        super().__init__()
        self.seq_len_q = seq_len_q
        self.seq_len_kv = seq_len_kv if exists(seq_len_kv) else seq_len_q
        self.k_max = k_max
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        inner_dim = heads * dim_head
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        # low-rank projection matrices & gating
        self.proj_k = nn.Parameter(torch.randn(self.seq_len_kv, k_max))
        self.proj_v = nn.Parameter(torch.randn(self.seq_len_kv, k_max))
        self.gate_k = nn.Parameter(torch.ones(k_max) * 0.5)
        self.gate_v = nn.Parameter(torch.ones(k_max) * 0.5)
        self.alpha_k = nn.Parameter(torch.tensor(1.0))
        self.alpha_v = nn.Parameter(torch.tensor(1.0))

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x, context=None):
        b, n_q, _ = x.shape
        context = x if context is None else context
        _, n_kv, _ = context.shape

        q = self.to_q(x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        eff_k = self.proj_k * (torch.sigmoid(self.gate_k) * self.alpha_k)
        eff_v = self.proj_v * (torch.sigmoid(self.gate_v) * self.alpha_v)

        k = torch.einsum('b n d, n k -> b k d', k, eff_k)
        v = torch.einsum('b n d, n k -> b k d', v, eff_v)

        h = self.heads
        q = rearrange(q, 'b n (h d) -> b h n d', h=h)
        k = rearrange(k, 'b k (h d) -> b h k d', h=h)
        v = rearrange(v, 'b k (h d) -> b h k d', h=h)

        attn = (q @ k.transpose(-1, -2)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        out = attn @ v
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# ---------------------------
# Sinusoidal Positional Encoding
# ---------------------------

class SinusoidalPE(nn.Module):
    def __init__(self, seq_len, dim):
        super().__init__()
        pe = torch.zeros(seq_len, dim)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float) * -(math.log(10000.0) / dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)  # non-trainable

    def forward(self, x):
        # x: (B, seq_len, dim)
        return x + self.pe.unsqueeze(0)

# ---------------------------
# PerceiverIO with Hybrid PE
# ---------------------------

class PerceiverIO(nn.Module):
    def __init__(
        self,
        seq_len: int,
        num_latents: int = 64,
        latent_dim: int = 64,
        depth: int = 2,
        latent_heads: int = 4,
        latent_dim_head: int = 16,
        dropout: float = 0.1,
        k_max: int = 128,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.latent_dim = latent_dim

        # 1) learnable pos emb, but zero-initialized (so it starts neutral)
        self.pos_emb = nn.Parameter(torch.zeros(seq_len, latent_dim))
        # 2) sinusoidal PE
        self.sin_pe = SinusoidalPE(seq_len, latent_dim)

        # global latents
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))
        self.layers = nn.ModuleList()

        for i in range(depth):
            if i == 0:
                attn = LinformerAttention(
                    dim=latent_dim,
                    seq_len_q=num_latents,
                    seq_len_kv=seq_len,
                    k_max=k_max,
                    heads=latent_heads,
                    dim_head=latent_dim_head,
                    dropout=dropout
                )
            else:
                attn = LinformerAttention(
                    dim=latent_dim,
                    seq_len_q=num_latents,
                    k_max=k_max,
                    heads=latent_heads,
                    dim_head=latent_dim_head,
                    dropout=dropout
                )
            ff = FeedForward(latent_dim)
            self.layers.append(nn.ModuleList([attn, ff]))

        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, x, context=None):
        # x: (B, seq_len, latent_dim)
        # apply both PEs:
        x = x + self.pos_emb.unsqueeze(0)     # learnable (starts zero)
        x = self.sin_pe(x)                    # fixed sin-cos

        b = x.size(0)
        latents = repeat(self.latents, 'n d -> b n d', b=b)

        for i, (attn, ff) in enumerate(self.layers):
            if i == 0:
                latents = attn(latents, context=x) + latents
            else:
                latents = attn(latents) + latents
            latents = ff(latents) + latents

        return self.norm(latents)

# ---------------------------
# Training & Evaluation
# ---------------------------
def train_one_epoch(model, embed, classifier, loader, optimizer, device):
    model.train(); embed.train(); classifier.train()
    total_loss, total = 0, 0
    for batch_idx, (imgs, labels) in enumerate(loader, start=1):
        imgs, labels = imgs.to(device), labels.to(device)
        b = imgs.size(0)

        tokens = imgs.permute(0,2,3,1).reshape(b, -1, 3)
        tokens = embed(tokens)
        latents = model(tokens)
        pooled = latents.mean(dim=1)
        logits = classifier(pooled)

        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item() * b
        total += b
    return total_loss / total

def evaluate(model, embed, classifier, loader, device):
    model.eval(); embed.eval(); classifier.eval()
    total_loss, total_correct, total = 0, 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            b = imgs.size(0)

            tokens = imgs.permute(0,2,3,1).reshape(b, -1, 3)
            tokens = embed(tokens)
            latents = model(tokens)
            pooled = latents.mean(dim=1)
            logits = classifier(pooled)

            loss = F.cross_entropy(logits, labels)
            preds = logits.argmax(dim=1)

            total_loss += loss.item() * b
            total_correct += (preds == labels).sum().item()
            total += b
    return total_loss / total, total_correct / total

# ---------------------------
# Main
# ---------------------------

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cifar_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616)),
    ])
    train_ds = datasets.CIFAR10('./data',True,transform=cifar_transform,download=True)
    test_ds  = datasets.CIFAR10('./data',False,transform=cifar_transform,download=True)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=4)
    test_loader  = DataLoader(test_ds,  batch_size=128, shuffle=False, num_workers=4)

    seq_len, latent_dim, num_classes = 32*32, 64, 10
    model = PerceiverIO(seq_len=seq_len, latent_dim=latent_dim).to(device)
    embed = nn.Linear(3, latent_dim).to(device)
    classifier = nn.Linear(latent_dim, num_classes).to(device)

    optimizer = optim.AdamW(
        list(model.parameters()) + list(embed.parameters()) + list(classifier.parameters()),
        lr=2e-4, weight_decay=1e-2
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)

    epochs = 50
    history = {
        'train_loss': [], 
        'val_loss':   [], 
        'val_acc':    [], 
        'lr':         []
    }
    k_history = []

    for epoch in range(1, epochs+1):
        tl = train_one_epoch(model, embed, classifier, train_loader, optimizer, device)
        vl, va = evaluate(model, embed, classifier, test_loader, device)
        scheduler.step()

        # 현재 effective k 계산
        gate_k  = model.layers[0][0].gate_k.detach()
        alpha_k = model.layers[0][0].alpha_k.detach()
        eff_k = (torch.sigmoid(gate_k) * alpha_k).mean().item()
        k_max = model.layers[0][0].proj_k.size(1)  # proj_k: [seq_len_kv, k_max]
        eff_k *= k_max
        # 기록
        history['train_loss'].append(tl)
        history['val_loss'].append(vl)
        history['val_acc'].append(va)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        k_history.append(eff_k)

        print(f"Epoch {epoch:03d} | "
              f"TrainLoss {tl:.4f} | ValLoss {vl:.4f} | ValAcc {va:.4f} | "
              f"LR {optimizer.param_groups[0]['lr']:.1e} | k={eff_k:.2f}")

    # ---------------------------
    # Visualization
    # ---------------------------

    epochs_range = range(1, epochs+1)

    plt.figure()
    plt.plot(epochs_range, history['train_loss'], label='Train Loss')
    plt.plot(epochs_range, history['val_loss'],   label='Val Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend(); plt.grid()
    plt.savefig("./cifar_loss.png")

    plt.figure()
    plt.plot(epochs_range, history['val_acc'], label='Val Accuracy')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy')
    plt.title('Validation Accuracy over Epochs')
    plt.legend(); plt.grid()
    plt.savefig("./cifar_acc.png")

    plt.figure()
    plt.plot(epochs_range, history['lr'], label='Learning Rate')
    plt.xlabel('Epoch'); plt.ylabel('LR')
    plt.title('Learning Rate Schedule')
    plt.legend(); plt.grid()

    plt.figure()
    plt.plot(epochs_range, k_history, label='Mean Effective k')
    plt.xlabel('Epoch'); plt.ylabel('Effective k')
    plt.title('Adaptive k')
    plt.legend(); plt.grid()
    plt.savefig("./cifar_k.png")

    plt.show()
