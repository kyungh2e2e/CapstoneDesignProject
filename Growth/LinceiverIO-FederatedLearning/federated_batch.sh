#!/bin/bash

ks=(32 64 128 256 512 784)

for k in "${ks[@]}"
do
  echo "Running experiment with k=$k, num_users=10, test_no=0"
  python federated_main_per.py \
    --model=perceiver-io-linstyle \
    --dataset=fmnist \
    --iid=1 \
    --epochs=10 \
    --optimizer=adam \
    --lr=5e-4 \
    --frac=1.0 \
    --num_users=10 \
    --local_bs=64 \
    --local_ep=5 \
    --test_no=1 \
    --k=$k
done
