#!/usr/bin/env bash

# Train Motifnet using different orderings

export CUDA_VISIBLE_DEVICES=$1

#checkpoints/vgdet/vg-24.tar
python models/train_rels2.py -m sgdet -model motifnet -order size -nl_obj 0 -nl_edge 0 -b 2 -clip 5 \
    -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/vgrel-motifnet-sgdet.tar \
    -save_dir checkpoints/motifnet-sgdet-addi-center-ht -nepoch 50 -use_bias

