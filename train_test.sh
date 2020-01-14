#!/bin/bash

datasetname="hazelnut"
subname="_256_3"

#python3 main.py --dataset_name $datasetname --sub_dirname $subname --phase train
python3 main.py --dataset_name $datasetname --sub_dirname $subname --phase test