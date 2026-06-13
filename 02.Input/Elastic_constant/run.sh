#!/bin/sh

#export OMP_NUM_THREADS=20
cvd=`get_avail_gpu.py`
export CUDA_VISIBLE_DEVICES=$cvd
#export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:100

~/src_mace_atomic/lammps-stable/build/lmp -in in.elastic
