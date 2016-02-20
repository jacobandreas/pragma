#!/bin/bash

export APOLLO_ROOT=/home/jda/3p/apollocaffe
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$APOLLO_ROOT/build/lib
export PYTHONPATH=$PYTHONPATH:$APOLLO_ROOT/python:$APOLLO_ROOT/python/caffe/proto

python -u main.py $@
#kernprof -l main.py
