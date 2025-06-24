#!/bin/sh

export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
echo "Beginning Generation Procedure"
echo "Current Working Directory Is:\n"
pwd

#FRICTION ***
#6D NOT IMPLEMENTED ***  

# difficult tasks: MS, CH
# easier tasks: VS, FC, FS 
# --- other tasks will be implemented later ---
# pid, joint pid tracking errors persist to some degree


#python genfranka.py -ne 8 -ni 1000 -f 0.15 -nctrl -tjt "MS" -td 'train' -nd 'MGC' -hdo '4D' -tr 'franka' -v -dg -df

#python genfranka.py -ne 8 -ni 1000 -f 0.15 -nctrl -tjt "MS" -td 'train' -nd 'MGC' -hdo '4D' -tr 'franka' -v -dg -df

#python genfranka.py -ne 8 -ni 1000 -f 0.15 -nctrl -tjt "MS" -td 'train' -nd 'MGC' -hdo '4D' -tr 'kuka' -v -dg -df

#python genfranka.py -ne 8 -ni 1000 -f 0.15 -nctrl -tjt "MS" -td 'train' -nd 'MGC' -hdo '4D' -tr 'franka' -v -df

#python genfranka.py -ne 8 -ni 1000 -f 0.15 -nctrl -tjt "MS" -td 'train' -nd 'MGC' -hdo '4D' -tr 'franka' -v -dg

#python genfranka.py -ne 8 -ni 1000 -f 0.30 -nctrl -tjt "MS" -td 'train' -nd 'MGC' -hdo '4D' -tr 'franka' -v -dg -df

#python genfranka.py -ne 8 -ni 1000 -f 0.15 -nctrl -tjt "CH" -td 'train' -nd 'MGC' -hdo '4D' -tr 'franka' -v -dg -df

#python genfranka.py -ne 8 -ni 1000 -f 0.15 -nctrl -tjt "IMP" -td 'train' -nd 'MGC' -hdo '4D' -tr 'franka' -v -dg -df

#python genfranka.py -ne 8 -ni 1000 -f 0.15 -nctrl -tjt "TRAPZ" -td 'train' -nd 'MGC' -hdo '4D' -tr 'franka' -v -dg -df


######
#for f in "0.15"; do
#    for i in "VS" "FS" "FC"; do
#        for r in $(seq 60); do
#            python genfranka.py -ne 2048 -ni 1000 -f 0.15 -ctrl -tctrl "osc" -ttraj $i -rcg -dfrp -nd "MGC/rng_base" -hdo "4D" -tr "franka" -dg -df -ri -rv -rm '10' -rcom '10' -rinr '10' -rstf '10' -rdam '10' -td 'train' -dp
#        done
#    done
#done
######

######
#for f in "0.15"; do
#    for i in "VS" "FS" "FC"; do
#        for r in $(seq 60); do
#            python genfranka.py -ne 2048 -ni 1000 -f $f -ctrl -tctrl "pid" -ttraj $i -rcg -dfrp -nd "MGC/pid_rng_VSFSFC" -hdo "4D" -tr "franka" -dg -df -ri -rv -rm '10' -rcom '10' -rinr '10' -rstf '10' -rdam '10' -td 'train' -dp
#        done
#    done
#done

#for f in "0.05" "0.10" "0.15"; do
#    for i in "MS"; do
#        for r in $(seq 30); do
#            python genfranka.py -ne 2048 -ni 1000 -f  $f -ctrl -tctrl "pid" -ttraj $i -rcg -dfrp -nd "MGC/pid_rng_MSCH" -hdo "4D" -tr "franka" -dg -df -ri -rv -rm '10' -rcom '10' -rinr '10' -rstf '10' -rdam '10' -td 'train' -dp
#        done
#    done
#done

#for f in "0.10" "0.15" "0.20"; do
#    for i in "CH"; do
#        for r in $(seq 30); do
#            python genfranka.py -ne 2048 -ni 1000 -f $f -ctrl -tctrl "pid" -ttraj $i -rcg -dfrp -nd "MGC/pid_rng_MSCH" -hdo "4D" -tr "franka" -dg -df -ri -rv -rm '10' -rcom '10' -rinr '10' -rstf '10' -rdam '10' -td 'train' -dp
#        done
#    done
#done
######

######
for f in "0.05" "0.10" "0.15"; do
    for i in "MS"; do
        for r in $(seq 30); do
            python genfranka.py -ne 2048 -ni 1000 -f $f -ctrl -tctrl "joint_pid" -ttraj $i -rcg -dfrp -nd "MGC/jointpid_rng_MSCH" -hdo "4D" -tr "franka" -dg -df -ri -rv -rm '10' -rcom '10' -rinr '10' -rstf '10' -rdam '10' -td 'train' -dp
        done
    done
done

for f in "0.10" "0.15" "0.20"; do
    for i in "CH"; do
        for r in $(seq 30); do
            python genfranka.py -ne 2048 -ni 1000 -f $f -ctrl -tctrl "joint_pid" -ttraj $i -rcg -dfrp -nd "MGC/jointpid_rng_MSCH" -hdo "4D" -tr "franka" -dg -df -ri -rv -rm '10' -rcom '10' -rinr '10' -rstf '10' -rdam '10' -td 'train' -dp
        done
    done
done
######
################################################33

#for f in "0.15"; do
#    for i in "VS" "FS" "FC"; do
#        for r in $(seq 1); do
#            python genfranka.py -ne 2048 -ni 1000 -f 0.15 -td "test" -ctrl -tctrl "osc" -ttraj $i -rcg -dfrp -nd "TC/osc" -hdo "4D" -tr "franka" -dg -df -ri -rv -rm '10' -rcom '10' -rinr '10' -rstf '10' -rdam '10' -dp
#        done
#    done
#done

#for f in "0.15"; do
#    for i in "VS" "FS" "FC"; do
#        for r in $(seq 1); do
#            python genfranka.py -ne 2048 -ni 1000 -f $f -td "test" -ctrl -tctrl "pid" -ttraj $i -rcg -dfrp -nd "TC/pid_VSFSFC" -hdo "4D" -tr "franka" -dg -df -ri -rv -rm '10' -rcom '10' -rinr '10' -rstf '10' -rdam '10' -dp
#        done
#    done
#done

#for f in "0.01" "0.05" "0.10" "0.15" "0.20"; do
#    for i in "MS"; do
#        for r in $(seq 1); do
#            python genfranka.py -ne 2048 -ni 1000 -f  $f -td "test" -ctrl -tctrl "pid" -ttraj $i -rcg -dfrp -nd "TC/pid_MSCH" -hdo "4D" -tr "franka" -dg -df -ri -rv -rm '10' -rcom '10' -rinr '10' -rstf '10' -rdam '10' -dp
#        done
#    done
#done

#for f in "0.05" "0.10" "0.15" "0.20" "0.25"; do
#    for i in "CH"; do
#        for r in $(seq 1); do
#            python genfranka.py -ne 2048 -ni 1000 -f $f -td "test" -ctrl -tctrl "pid" -ttraj $i -rcg -dfrp -nd "TC/pid_MSCH" -hdo "4D" -tr "franka" -dg -df -ri -rv -rm '10' -rcom '10' -rinr '10' -rstf '10' -rdam '10' -dp
#        done
#    done
#done

for f in "0.01" "0.05" "0.10" "0.15" "0.20"; do
    for i in "MS"; do
        for r in $(seq 1); do
            python genfranka.py -ne 2048 -ni 1000 -f $f -td "test" -ctrl -tctrl "joint_pid" -ttraj $i -rcg -dfrp -nd "TC/jointpid_MSCH" -hdo "4D" -tr "franka" -dg -df -ri -rv -rm '10' -rcom '10' -rinr '10' -rstf '10' -rdam '10' -dp
        done
    done
done

for f in "0.05" "0.10" "0.15" "0.20" "0.25"; do
    for i in "CH"; do
        for r in $(seq 1); do
            python genfranka.py -ne 2048 -ni 1000 -f $f -td "test" -ctrl -tctrl "joint_pid" -ttraj $i -rcg -dfrp -nd "TC/jointpid_MSCH" -hdo "4D" -tr "franka" -dg -df -ri -rv -rm '10' -rcom '10' -rinr '10' -rstf '10' -rdam '10' -dp
        done
    done
done
