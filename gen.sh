#!/bin/sh

export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
echo "Beginning Generation Procedure"
echo "Current Working Directory Is:\n"
pwd

for f in "0.1"; do
    for i in "MS"; do
        for r in "1"; do
            python genfranka.py -ne 16 -ni 10000 -f $f -c -ti $i -td 'train' -nd 'MG1' -hdo '4D' -ri -rv -rm '10' -rcom '10' -rinr '10' -rstf '10' -rdam '10' -rcf '10' -s '42' -mf -is -fq 
        done
    done
done

#for f in "0.1"; do
#    for i in "VS"; do
#        for r in "1" "2" "3" "4"; do
#            python genfranka.py -ne 4 -ni 1000 -f $f -osc -tosc $i -td 'train' -nd 'trial' -hdo '4D' -ri -rv -rm '10' -rcom '10' -rinr '10' -rstf '10' -rdam '99' -rcf '10' -s '42'    
#        done
#    done
#done

echo "Ending Generation Procedure"
echo "Total Number of Data Generated of the Specified Morphology is: X"  

#python -m cProfile -s 'cumtime' -o 'genprofile_sat.txt' genfranka2.py -ne 1024 -ni 1000 -f '0.1' -c -ti 'MS' -td 'train' -di -dp -ds -is -fq
#python genfranka.py -ne 128 -ni 1000 -f '0.1' -c -ti 'MS' -td 'train' -dg -df -di -ri -rv -rcg -v  
#python genfranka.py -ne 128 -ni 1000 -f '0.1' -c -ti 'MS' -td 'train' -di -ri -rv -rcf '5' -rad '5' -rm '5' -rcom '5' -rinr '5' -rstf '5' -rdam '5' -v -dp

#python genfranka.py -ne 128 -ni 1000 -f '0.1' -c -ti 'MS' -td 'train' -v -dg
#python genfranka.py -ne 128 -ni 1000 -f '0.1' -c -ti 'MS' -td 'train' -v 

#python genfranka.py -ne 128 -ni 1000 -f '0.1' -c -ti 'MS' -td 'train' -v -df
#python genfranka.py -ne 128 -ni 1000 -f '0.1' -c -ti 'MS' -td 'train' -v 

#python genfranka.py -ne 128 -ni 1000 -f '1.0' -c -ti 'MS' -td 'train' -v
#python genfranka.py -ne 128 -ni 1000 -f '0.1' -c -ti 'MS' -td 'train' -v 

#base
##python genfranka.py -ne 128 -ni 1000 -f '1.0' -c -ti 'MS' -td 'train' -v
#python genfranka.py -ne 128 -ni 1000 -f '0.1' -c -ti 'MS' -td 'train' -v 

#extended freqs range in inputs
#python genfranka.py -ne 128 -ni 1000 -f '1.0' -c -ti 'MS' -td 'train' -v
#python genfranka.py -ne 128 -ni 1000 -f '0.1' -c -ti 'MS' -td 'train' -v 

#more randomization
#python genfranka.py -ne 128 -ni 1000 -f '1.0' -c -ti 'MS' -td 'train' -v
#python genfranka.py -ne 128 -ni 1000 -f '0.1' -c -ti 'MS' -td 'train' -v 

#4 tasks
#python genfranka.py -ne 128 -ni 1000 -f '1.0' -c -ti 'MS' -td 'train' -v
#python genfranka.py -ne 128 -ni 1000 -f '0.1' -c -ti 'MS' -td 'train' -v 

#damping
#python genfranka.py -ne 128 -ni 1000 -f '1.0' -c -ti 'MS' -td 'train' -v
#python genfranka.py -ne 128 -ni 1000 -f '0.1' -c -ti 'MS' -td 'train' -v 

#stiffness
#python genfranka.py -ne 128 -ni 1000 -f '1.0' -c -ti 'MS' -td 'train' -v
#python genfranka.py -ne 128 -ni 1000 -f '0.1' -c -ti 'MS' -td 'train' -v 

#coulomb
#python genfranka.py -ne 128 -ni 1000 -f '1.0' -c -ti 'MS' -td 'train' -v
#python genfranka.py -ne 128 -ni 1000 -f '0.1' -c -ti 'MS' -td 'train' -v 

#saturation included
#python genfranka.py -ne 128 -ni 1000 -f '1.0' -c -ti 'MS' -td 'train' -v
#python genfranka.py -ne 128 -ni 1000 -f '0.1' -c -ti 'MS' -td 'train' -v 

#quarternion adjusted
#python genfranka.py -ne 128 -ni 1000 -f '1.0' -c -ti 'MS' -td 'train' -v
#python genfranka.py -ne 128 -ni 1000 -f '0.1' -c -ti 'MS' -td 'train' -v 

#6D orientation
#python genfranka.py -ne 128 -ni 1000 -f '1.0' -c -ti 'MS' -td 'train' -v
#python genfranka.py -ne 128 -ni 1000 -f '0.1' -c -ti 'MS' -td 'train' -v 

#measure torques and hold compensation values
#python genfranka.py -ne 128 -ni 1000 -f '1.0' -c -ti 'MS' -td 'train' -v
#python genfranka.py -ne 128 -ni 1000 -f '0.1' -c -ti 'MS' -td 'train' -v 


#************************************************************************#

#for frequency in 0.1; do
#    for bounds in 10; do
#        for envs in 4; do
#            for type in 'MS'; do                                                                                                                            
#                #python genfranka.py -ri 1 -rm 10 -rstf 5 -rdam 5 -rcom 5 -rinr 5 -ne 4 -di -ni 1000 -f "${frequency}" -c -ti "${type}" -sd -td 'train' -hm --sim_device 'cuda:0'
#                python genfranka.py -ne 64 -ni 1000 -f "${frequency}" -c -ti "${type}" -td 'train'  -di -ri -rv -rcg -dp
#                #python generation_franka.py --num-envs 4 --max-iteration 1000 --frequency "${frequency}" --control-imposed True --type-of-input "${type}" --save-tensors True --type-of-dataset 'train' --headless-mode False
#            done
#        done 
#    done
#done
#seqf = ("0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9" "1.0" "1.1" "1.2" "1.3" "1.4" "1.5" "1.6" "1.7" "1.8" "1.9" "2.0")
#seqi = {'MS' 'CH'}
#seqrep = ("1" "2")

#for f in "0.1"; do
#    for i in "MS"; do
#        for r in "1"; do
#            python genfranka2.py -ne 16 -ni 1000 -f $f -c -ti $i -td 'train' -ds -nd 'MG1' -mf -mgf -s '42' -fq -is -v
#        done
#    done
#done    