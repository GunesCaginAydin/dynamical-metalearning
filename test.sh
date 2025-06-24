# Testing follows from train.sh with the same parser arguments. For most cases no argument needs to be passed
# since testing script directly utilizes models that are created before. However, for the specific cases of
# context tests and fewshot tests, additional arguments may be passed as below.
#
# context tests
### reduced horizon tests
python test.py -cos -std --data-name 'MG1' --test-name 'T1' --total-sim-iterations 500
### different inference context tests with initparams = {'ctx'=...,}
python test.py -cos -std --data-name 'MG1' --test-name 'T1' --total-sim-iterations 500
# fewshot tests
### |

# default sim2sim testing
python test.py -cos -std --data-name 'MG1' --test-name 'T1' --total-sim-iterations 1000

# default sim2real testing
python test.py -ctrl -cos -std --data-name 'MGC/pid_rng_VSFSFC' --test-name 'TC/pid_VSFSFC' --total-sim-iterations 1000
