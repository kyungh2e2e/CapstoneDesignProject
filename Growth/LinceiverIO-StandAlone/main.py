import argparse
import os
import time

import torch
from torch import nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from perceiver_io_linstyle import PerceiverIOLinstyle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_one_epoch(model, embedding, query, dataloader, optimizer, criterion):
    model.train()
    embedding.train()

    total_loss = 0.0
    total_samples = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        batch_size = images.size(0)
        images = images.view(batch_size, -1, 1)
        tokens = embedding(images)
        queries = query.expand(batch_size, -1, -1)

        optimizer.zero_grad()
        logits = model(tokens, queries=queries).squeeze(1)
        loss = criterion(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * batch_size
        total_samples += batch_size

    avg_loss = total_loss / total_samples
    return avg_loss

def evaluate(model, embedding, query, dataloader, criterion):
    model.eval()
    embedding.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            batch_size = images.size(0)
            images = images.view(batch_size, -1, 1)
            tokens = embedding(images)
            queries = query.expand(batch_size, -1, -1)
            logits = model(tokens, queries=queries).squeeze(1)

            loss = criterion(logits, labels)
            total_loss += loss.item() * batch_size

            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += batch_size

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='perceiver-io-linstyle')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--k', type=int, default=128)
    parser.add_argument('--test_no', type=int, default=None)
    args = parser.parse_args()

    DEPTH = 2
    MODEL_DIM = 64
    QUERIES_DIM = 64
    SEQ_LEN = 784
    K = args.k 
    NUM_CLASSES = 10
    NUM_LATENTS = 64
    CROSS_HEADS = 1
    LATENT_HEADS = 4
    CROSS_DIM_HEAD = 16
    LATENT_DIM_HEAD = 16

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset  = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    os.makedirs(f"./results/{args.k}/{args.test_no}", exist_ok=True)
    
    if args.model.lower() == 'perceiver-io-linstyle':
        model = PerceiverIOLinstyle(
            depth=DEPTH, 
            dim=MODEL_DIM,
            queries_dim=QUERIES_DIM,
            logits_dim=NUM_CLASSES,
            num_latents=NUM_LATENTS,
            latent_dim=MODEL_DIM,
            cross_heads=CROSS_HEADS,
            latent_heads=LATENT_HEADS,
            cross_dim_head=CROSS_DIM_HEAD,
            latent_dim_head=LATENT_DIM_HEAD,
            seq_len=SEQ_LEN,
            k=K,
            decoder_ff=True
        ).to(device)
        model_save_path = f"./results/{args.k}/{args.test_no}/perceiver_io_linstyle.pth"
        result_txt_path = f"./results/{args.k}/{args.test_no}/perceiver_io_linstyle_results.txt"
    else:
        raise ValueError('지원되지 않는 모델 타입입니다.')

    embedding = nn.Linear(1, MODEL_DIM).to(device)
    query = nn.Parameter(torch.randn(1, QUERIES_DIM, device=device) * 0.1)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        list(model.parameters()) + list(embedding.parameters()) + [query],
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    train_loss_history = []
    test_loss_history = []
    test_acc_history = []

    start_time = time.time()

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, embedding, query, train_loader, optimizer, criterion)
        test_loss, test_acc = evaluate(model, embedding, query, test_loader, criterion)

        train_loss_history.append(train_loss)
        test_loss_history.append(test_loss)
        test_acc_history.append(test_acc)

        print(f"[Epoch {epoch+1}/{args.epochs}] "
              f"Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

    end_time = time.time()
    total_time = end_time - start_time

    torch.save(model.state_dict(), model_save_path)

    model_size_mb = os.path.getsize(model_save_path) / (1024 * 1024)

    with open(result_txt_path, 'w', encoding='utf-8') as f:
        f.write(f"===== {args.model} 학습 결과 =====\n\n")
        for i in range(args.epochs):
            f.write(f"Epoch {i+1}: Train Loss={train_loss_history[i]:.4f}, "
                    f"Test Loss={test_loss_history[i]:.4f}, Test Acc={test_acc_history[i]:.4f}\n")
        f.write("\n")
        f.write(f"총 학습 시간 (초): {total_time:.2f}\n")
        f.write(f"모델 크기 (MB): {model_size_mb:.2f}\n")

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, args.epochs + 1), train_loss_history, label='Train Loss', linewidth=2)
    plt.plot(range(1, args.epochs + 1), test_loss_history, label='Test Loss', linewidth=2, linestyle='dashed')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f"{args.model} Loss")
    plt.xticks(range(1, args.epochs + 1))
    plt.legend()
    plt.grid(True)
    loss_plot_path = f"./results/{args.k}/{args.test_no}/perceiver_io_linstyle_lossplot.png"
    plt.savefig(loss_plot_path)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, args.epochs + 1), [acc * 100 for acc in test_acc_history], label='Test Accuracy', color='green', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title(f"{args.model} Accuracy")
    plt.xticks(range(1, args.epochs + 1))
    plt.yticks(range(0, 101, 10))
    plt.ylim([0, 100])
    plt.legend()
    plt.grid(True)
    accuracy_plot_path = f"./results/{args.k}/{args.test_no}/perceiver_io_linstyle_accplot.png"
    plt.savefig(accuracy_plot_path)
    plt.close()

    print(f"\n>>> 학습 완료! 결과 저장 경로:")
    print(f" - 모델 파라미터: {model_save_path}")
    print(f" - 텍스트 결과:   {result_txt_path}")
    print(f" - Loss Plot:     {loss_plot_path}")
    print(f" - Accuracy Plot: {accuracy_plot_path}\n")

if __name__ == "__main__":
    main()