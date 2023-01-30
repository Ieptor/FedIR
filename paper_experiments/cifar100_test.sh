#!/usr/bin/env bash

pushd ../models

declare -a alphas=("1000")

function run_fedavg() {
  echo "############################################## Running FedAvg ##############################################"
  alpha="$1"
  python main.py -dataset cifar100 --num-rounds 1000 --eval-every 100 --batch-size 64 --num-epochs 3 --clients-per-round 10 -model resnet -lr 0.01 --weight-decay 0.004 -device cuda:0 -algorithm fedopt --server-lr 1 --server-opt sgd --server-momentum 0.9 --num-workers 0 --where-loading init -alpha ${alpha}
}

echo "####################### EXPERIMENTS ON CIFAR100 #######################"
for alpha in "${alphas[@]}"; do
  echo "Alpha ${alpha}"
  run_fedavg "${alpha}"
  echo "Done"
done