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

        # projection matrices for q and context (kv)
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
        # context sequence length
        context = x if context is None else context
        b_c, n_kv, _ = context.shape

        q = self.to_q(x)  # (b, n_q, inner_dim)
        k, v = self.to_kv(context).chunk(2, dim=-1)  # (b, n_kv, inner_dim)

        # adaptive k/v projection
        eff_k = self.proj_k * (torch.sigmoid(self.gate_k) * self.alpha_k)  # (n_kv, k_max)
        eff_v = self.proj_v * (torch.sigmoid(self.gate_v) * self.alpha_v)

        # project k, v into lower dimension
        # 'b n d, n k -> b k d'
        k = torch.einsum('b n d, n k -> b k d', k, eff_k)
        v = torch.einsum('b n d, n k -> b k d', v, eff_v)

        # reshape for multihead
        h = self.heads
        q = rearrange(q, 'b n (h d) -> b h n d', h=h)
        k = rearrange(k, 'b k (h d) -> b h k d', h=h)
        v = rearrange(v, 'b k (h d) -> b h k d', h=h)

        # scaled dot-product
        attn = (q @ k.transpose(-1, -2)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        out = attn @ v
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# ---------------------------
# PerceiverIO Definition
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
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))
        self.layers = nn.ModuleList()

        for i in range(depth):
            # 첫 레이어는 cross-attn: seq_len_q=num_latents, seq_len_kv=seq_len(input)
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
                # 이후 레이어는 self-attn: seq_len_q=seq_len_kv=num_latents
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
        # x: (B, seq_len, latent_dim), context for cross-attn
        b = x.size(0)
        latents = repeat(self.latents, 'n d -> b n d', b=b)

        for i, (attn, ff) in enumerate(self.layers):
            if i == 0:
                # 첫 블록에만 cross-attn 사용
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
    for imgs, labels in tqdm(loader, desc="Train"):  
        imgs, labels = imgs.to(device), labels.to(device)
        b = imgs.size(0)
        imgs = imgs.view(b, -1, 1)

        optimizer.zero_grad()
        tokens = embed(imgs)  # (b, seq_len, latent_dim)
        latents = model(tokens)  # cross-attn 내부에서 context=tokens 적용
        pooled = latents.mean(dim=1)
        logits = classifier(pooled)

        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()

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
            imgs = imgs.view(b, -1, 1)

            tokens = embed(imgs)
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
    print("Using device:", device)

    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_ds  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False)

    # Model, Embedding, Classifier
    model = PerceiverIO(seq_len=28*28).to(device)
    embed = nn.Linear(1, 64).to(device)
    classifier = nn.Linear(64, 10).to(device)

    optimizer = optim.Adam(
        list(model.parameters()) + list(embed.parameters()) + list(classifier.parameters()),
        lr=5e-4
    )

    # 기록용 리스트
    k_history = []
    train_loss_history = []
    test_loss_history = []
    test_acc_history = []
    epochs = 50

    for epoch in range(1, epochs+1):
        train_loss = train_one_epoch(model, embed, classifier, train_loader, optimizer, device)
        test_loss, test_acc = evaluate(model, embed, classifier, test_loader, device)
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Test Loss={test_loss:.4f}, Test Acc={test_acc:.4f}")

        # 기록
        train_loss_history.append(train_loss)
        test_loss_history.append(test_loss)
        test_acc_history.append(test_acc)
        gate_k = model.layers[0][0].gate_k.detach()
        alpha_k = model.layers[0][0].alpha_k.detach()
        effective_k = (torch.sigmoid(gate_k) * alpha_k).mean().item()
        k_history.append(effective_k)

    # Plot: Adaptive k 변화
    plt.figure()
    plt.plot(range(1, epochs+1), k_history, label='Mean Effective k')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Effective k')
    plt.title('Adaptive k')
    plt.grid(True)
    plt.legend()
    plt.savefig("./mnist_k.png")
    plt.show()

    # Plot: Loss
    plt.figure()
    plt.plot(range(1, epochs+1), train_loss_history, label='Train Loss')
    plt.plot(range(1, epochs+1), test_loss_history, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.grid(True)
    plt.legend()
    plt.savefig("./mnist_loss.png")
    plt.show()

    # Plot: Accuracy
    plt.figure()
    plt.plot(range(1, epochs+1), test_acc_history, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy over Epochs')
    plt.grid(True)
    plt.legend()
    plt.savefig("./mnist_accuracy.png")
    plt.show()
