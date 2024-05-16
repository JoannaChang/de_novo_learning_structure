#!/bin/bash
export seed_str='1000000 1000001 1000002 1000003 1000004 1000005 1000006 1000007 1000008 1000009'
export sds=($seed_str)
export nseeds=${#sds[@]}
export threads=2
export incr=$((nseeds / threads))

# MAKE DATASETS (only need to do this once)
# python3 tasks/synth_dataset.py
# seeds=($seed_str)
# for seed in ${seeds[@]}
# do
#     python3 tasks/rotate_dataset.py $seed
# done

# SIMULATIONS: skill learning and VR adaptation at 10 degrees for angular and categorical inputs
for ((i=0;i<$threads;i++)); 
do
    export i=$i
    screen -dmS run$i bash -c \
    'seeds=($seed_str)
    for seed in ${seeds[@]:$((i*incr)):$(((i+1)*incr))}
    do  
        # for config_num in {1..23}
        for config_num in 12 13
        do

            # ANGULAR ENCODING
            export sim_set=uni_rad
            export repertoire_pre=uni_
            export file=${repertoire_pre}10.0_rad

            for repertoire in 10.0 2movs_10_50 3movs_10_50 4movs_10_50 
            do
                #SKILL LEARNING
                python3 simulation/run_pipeline.py init_training $seed ${sim_set} ${repertoire_pre}${repertoire} -gpu 0 -cn ${config_num}
    
                #PERTURBATION: ROT
                pert_param=10.0
                python3 simulation/run_pipeline.py perturbation $seed ${sim_set} ${repertoire_pre}${repertoire} -p rotation -pp ${pert_param} -n 1 -gpu 0 -o SGD -lr 0.005 -file ${file} -cn ${config_num}

                #PERTURBATION: REASSOC
                pert_param=0
                python3 simulation/run_pipeline.py perturbation $seed ${sim_set} ${repertoire_pre}${repertoire} -p reassoc -pp ${pert_param} -n 1 -gpu 0 -o SGD -lr 0.005 -tt 300 -cn ${config_num}
            done

            # ONE-HOT ENCODING
            export sim_set=uni_onehot
            export repertoire_pre=uni_
            export file=${repertoire_pre}10.0_onehot

            for repertoire in 10.0 2movs_10_50 3movs_10_50 4movs_10_50
            do
                #SKILL-LEARNING
                python3 simulation/run_pipeline.py init_training $seed ${sim_set} ${repertoire_pre}${repertoire} -gpu 0 -cn ${config_num}
    
                #PERTURBATION: ROT
                python3 simulation/run_pipeline.py perturbation $seed ${sim_set} ${repertoire_pre}${repertoire} -p rotation -pp 10.0 -n 1 -gpu 0 -o SGD -lr 0.005 -file ${file} -cn ${config_num}

                #PERTURBATION: REASSOC
                pert_param=0
                python3 simulation/run_pipeline.py perturbation $seed ${sim_set} ${repertoire_pre}${repertoire} -p reassoc -pp ${pert_param} -n 1 -gpu 0 -o SGD -lr 0.005 -tt 300 -cn ${config_num}
            done
        done
    done'
done