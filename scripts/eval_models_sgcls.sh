#!/usr/bin/env bash

# This is a script that will evaluate all models for SGCLS
export CUDA_VISIBLE_DEVICES=$1

python models/eval_rels.py -m sgcls -model motifnet -nl_obj 0 -nl_edge 0 -b 6 \
-clip 5 -p 100 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/motifnet-predcls-addi-center-ht/vgrel-13.tar \
-nepoch 50 -use_bias -test 
