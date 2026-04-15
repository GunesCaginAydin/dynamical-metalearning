#!/bin/sh

cd sys_identification
echo "Current Working Directory Is:\n"
pwd
# Testing follows from train.sh with the same parser arguments. For most cases no argument needs to be passed
# since testing script directly utilizes models that are created before.

# default sim2sim testing
#python test.py -cos -std --data-name 'MG11' --test-name 'T1' --total-sim-iterations 1000

