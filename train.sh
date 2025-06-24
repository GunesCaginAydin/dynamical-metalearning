# to train / finetune from a dataset, below scripts may be used / modified according to the needs, possible
# parameters of training are listed below for sim2sim, sim2real and finetuning. Training options are presented
# in detail on utils.py under parser class.
#
# sim2sim -init=scratch
#   * -in (input dimension), -out (output dimension), -trb/-vrb (training/validation batch size), -lr (learning rate), -ctrl (implement controller), -cgh (gain horizon)
#   * -cos, -cons, -cwu, -stp, -exp (learning rate decays)
#   * -std, -lin, -cdf (normalizers)
#   * -ctx (context window), -ws (warmstart for diffuser), -rft (reward function training for diffuser)
#   * -transformer, -diffuser, -rechorUnet, -rechorTrf (training architectures)
#
# sim2real -ctrl
#   * -in (input dimension), -out (output dimension), -trb/-vrb (training/validation batch size), -lr (learning rate), -ctrl (implement controller)
#   * -transformer, -diffuser, -rechorUnet, -rechorTrf (training architectures)
#
# finetuning -init=finetune
#   * -in (input dimension), -out (output dimension), -trb/-vrb (training/validation batch size), -lr (learning rate), -ctrl (implement controller)
#   * -transformer, -diffuser, -rechorUnet, -rechorTrf (training architectures)

###############################################################################################################################################################################################
# sim2sim training
### transformer
python train.py -in 7 -out 14 -cos -std --data-name 'MG1' -lr '6e-4' -trb 32 -vlb 32 -evitr 100 -ctx 20 transformer -ttrf 1 -nl 12 -nh 12 -ne 384
### diffuser
python train.py -in 7 -out 14 -cos -std --data-name 'MG1' -lr '6e-4' -trb 32 -vlb 32 -evitr 100 -ctx 20 diffuser -tdif 1 -nh 256 -nc 3 -ts 100 -ucond -ycond -lw 3
### receding horizon Unet
python train.py -in 7 -out 14 -cwu -std --data-name 'MG1' -lr '1e-4' -trb 32 -vlb 32 -evitr 100 -ctx 20 rechorUnet -trch 2 -nh 256 -nc 3 -ts 100 -lc -pt 'epsilon'
### receding horizon Transformer
python train.py -in 7 -out 14 -cwu -std --data-name 'MG1' -lr '1e-4' -trb 32 -vlb 32 -evitr 100 -ctx 20 rechorTrf -trch 2 -nl 12 -nh 12 -ne 384 -ts 100 -csl -tc -oc -pt 'epsilon' 
###############################################################################################################################################################################################
# sim2real training
### torque input
python train.py -ctrl -in 7 -out 14 -cos -std --data-name 'MGC/jointpid_MSCH' -lr '6e-4' -trb 32 -vlb 32 -evitr 100 -ctx 20 transformer -ttrf 1 -nl 12 -nh 12 -ne 384
### tracking trajectory input
python train.py -ctrl -in 7 -out 14 -cos -std --data-name 'MGC/jointpid_MSCH' -lr '6e-4' -trb 32 -vlb 32 -evitr 100 -ctx 20 transformer -ttrf 2 -nl 12 -nh 12 -ne 384
### controller gain input
python train.py -ctrl -cgh 1000 -in 7 -out 14 -cwu -std --data-name 'MGC/jointpid_MSCH' -lr '1e-4' -trb 32 -vlb 32 -evitr 100 -ctx 20 rechorTrf -trch 6 -nl 12 -nh 12 -ne 384 -ts 100 -csl -tc -oc -pt 'epsilon' 
###############################################################################################################################################################################################
# finetuning
### finetune on KUKA domain, with modelname in initialize_model()
python train.py -init 'finetune' -in 7 -out 14 -cos -std --data-name 'MG1' -lr '6e-4' -trb 32 -vlb 32 -evitr 100 -ctx 20 transformer -ttrf 1 -nl 12 -nh 12 -ne 384
