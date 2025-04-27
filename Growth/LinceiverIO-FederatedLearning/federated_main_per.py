#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import copy
import time
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

from options import args_parser
from utils import get_dataset, exp_details
from test_with_sharing_lat import SharedLatentPerceiver, PerceiverHead

def average_backbone(local_backbone_states):
    if len(local_backbone_states) == 0:
        return None

    avg_dict = copy.deepcopy(local_backbone_states[0])
    for k in avg_dict.keys():
        for i in range(1, len(local_backbone_states)):
            avg_dict[k] += local_backbone_states[i][k]
        avg_dict[k] = avg_dict[k] / len(local_backbone_states)
    return avg_dict

def forward_model(images, shared_backbone, embedding, head, query):
    b = images.size(0)
    images_reshaped = images.view(b, -1, 1)
    data_tokens = embedding(images_reshaped)

    latents = shared_backbone(b)
    queries = query.expand(b, -1, -1)

    logits = head(data_tokens, queries, latents)
    logits = logits.squeeze(1)
    return logits

if __name__ == '__main__':
    start_time = time.time()

    args = args_parser()
    exp_details(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    logger = SummaryWriter('./logs')

    train_dataset, test_dataset, user_groups = get_dataset(args)

    shared_backbone = SharedLatentPerceiver(
        num_latents=64,
        latent_dim=64,
        depth=2,
        latent_heads=4,
        latent_dim_head=16,
        dropout=0.1
    ).to(device)

    clients = []
    for i in range(args.num_users):
        embedding = nn.Linear(1, 64).to(device)
        head = PerceiverHead(
            data_dim=64,
            latent_dim=64,
            queries_dim=64,
            logits_dim=10,
            seq_len=784,
            cross_heads=1,
            cross_dim_head=16,
            k=args.k,
            dropout=0.1,
            decoder_ff=True
        ).to(device)
        query = nn.Parameter(torch.randn(1, 64, device=device) * 0.1)
        optimizer = optim.Adam(
            list(shared_backbone.parameters()) +
            list(embedding.parameters()) +
            list(head.parameters()) +
            [query],
            lr=args.lr,
            weight_decay=1e-4
        )
        clients.append({
            "embedding": embedding,
            "head": head,
            "query": query,
            "optimizer": optimizer
        })

    criterion = nn.CrossEntropyLoss().to(device)

    round_losses = []
    round_accs   = []

    os.makedirs('./save/fmnist', exist_ok=True)
    report_file_path = os.path.join('./save/fmnist', f'fed_linceiverio_report_{args.local_ep}_{args.k}_{args.test_no}.txt')
    report_file = open(report_file_path, 'w')
    command_line = ' '.join(sys.argv)
    report_file.write(command_line + "\n\n")

    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        local_accs = []
        print(f'\n | Global Training Round : {epoch+1} |\n')
        report_file.write(f"Global round : {epoch+1}\n")
        
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        
        for idx in idxs_users:
            client = clients[idx]
            embedding = client["embedding"]
            head = client["head"]
            query = client["query"]
            optimizer = client["optimizer"]

            train_loader = torch.utils.data.DataLoader(
                torch.utils.data.Subset(train_dataset, list(user_groups[idx])),
                batch_size=args.local_bs, shuffle=True
            )

            embedding.train()
            head.train()
            shared_backbone.train()

            local_loss, local_correct, local_samples = 0.0, 0, 0

            for _ in range(args.local_ep):
                for images, labels in train_loader:
                    images, labels = images.to(device), labels.to(device)
                    optimizer.zero_grad()

                    logits = forward_model(images, shared_backbone, embedding, head, query)
                    loss = criterion(logits, labels)
                    loss.backward()
                    optimizer.step()

                    bs = images.size(0)
                    local_loss += loss.item() * bs
                    preds = logits.argmax(dim=1)
                    local_correct += (preds == labels).sum().item()
                    local_samples += bs

            if local_samples > 0:
                avg_loss = local_loss / local_samples
                avg_acc  = local_correct / local_samples
            else:
                avg_loss = 0.0
                avg_acc  = 0.0

            local_losses.append(avg_loss)
            local_accs.append(avg_acc)
            print(f"  [Client {idx}] local_loss={avg_loss:.4f}, local_acc={avg_acc*100:.2f}%")
            report_file.write(f"[Client {idx}] local_loss: {avg_loss:.4f}, local_acc: {avg_acc*100:.2f}%\n")

            backbone_sd = copy.deepcopy(shared_backbone.state_dict())
            local_weights.append(backbone_sd)

        if len(local_weights) > 0:
            avg_sd = average_backbone(local_weights)
            shared_backbone.load_state_dict(avg_sd)

        round_loss = float(np.mean(local_losses)) if local_losses else 0.0
        round_losses.append(round_loss)
        round_acc = float(np.mean(local_accs)) if local_accs else 0.0
        round_accs.append(round_acc)
        print(f"  [Round {epoch+1}] average local_loss={round_loss:.4f}")
        report_file.write(f"[Average] local_loss: {round_loss:.4f}, local_acc: {round_acc*100:.2f}%\n\n")

    total_run_time = time.time() - start_time
    print("\nDone Training.")
    print(f"Total Run Time: {total_run_time:.2f} sec")
    report_file.write(f"Total Run Time: {total_run_time:.2f} sec\n")
    report_file.close()