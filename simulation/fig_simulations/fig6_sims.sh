#!/bin/bash
export seed_str='1000000 1000001 1000002 1000003 1000004 1000005 1000006 1000007 1000008 1000009'
export sds=($seed_str)
export nseeds=${#sds[@]}
export threads=2
export incr=$((nseeds / threads))

# MAKE DATASETS (only need to do this once)
seeds=($seed_str)
for seed in ${seeds[@]}
do
    python3 tasks/sinewaves2d_dataset.py $seed
done

# SIMULATIONS
for ((i=0;i<$threads;i++)); 
do
    export i=$i
    screen -dmS run$i bash -c \
    'seeds=($seed_str)
    for seed in ${seeds[@]:$((i*incr)):$(((i+1)*incr))}
    do  
        # AMP A1
        for encoding in cont_onehot
        do
            export sim_set=ampA1_${encoding}
            export repertoire_pre=ampA1_

            for repertoire in 1.0 2movs_1_7 3movs_1_7 4movs_1_7
            do
                #SKILL LEARNING
                python3 simulation/run_pipeline.py init_training $seed ${sim_set} ${repertoire_pre}${repertoire} -gpu 0
    
                #PERTURBATION: PERTAMP A
                python3 simulation/run_pipeline.py perturbation $seed ${sim_set} ${repertoire_pre}${repertoire} -p pertampA -pp 0 -n 1 -gpu 0 -o SGD -lr 0.005 -file ampA1_1.0_${encoding}_pertampA

                #PERTURBATION: PERTAMP B
                python3 simulation/run_pipeline.py perturbation $seed ${sim_set} ${repertoire_pre}${repertoire} -p pertampB -pp 0 -n 1 -gpu 0 -o SGD -lr 0.005 -file ampB1_1.0_${encoding}_pertampB

                #PERTURBATION: REASSOC
                python3 simulation/run_pipeline.py perturbation $seed ${sim_set} ${repertoire_pre}${repertoire} -p reassoc -pp 0 -n 2 -gpu 0 -o SGD -lr 0.005 -tt 400
            done
        done


        # AMP B1
        for encoding in cont_onehot
        do
            export sim_set=ampB1_${encoding}
            export repertoire_pre=ampB1_

            for repertoire in 1.0 2movs_1_7 3movs_1_7 4movs_1_7
            do
                #SKILL LEARNING
                python3 simulation/run_pipeline.py init_training $seed ${sim_set} ${repertoire_pre}${repertoire} -gpu 0
    
                #PERTURBATION: PERTAMP A
                python3 simulation/run_pipeline.py perturbation $seed ${sim_set} ${repertoire_pre}${repertoire} -p pertampA -pp 0 -n 1 -gpu 0 -o SGD -lr 0.005 -file ampA1_1.0_${encoding}_pertampA

                #PERTURBATION: PERTAMP B
                python3 simulation/run_pipeline.py perturbation $seed ${sim_set} ${repertoire_pre}${repertoire} -p pertampB -pp 0 -n 1 -gpu 0 -o SGD -lr 0.005 -file ampB1_1.0_${encoding}_pertampB

                #PERTURBATION: REASSOC
                python3 simulation/run_pipeline.py perturbation $seed ${sim_set} ${repertoire_pre}${repertoire} -p reassoc -pp 0 -n 2 -gpu 0 -o SGD -lr 0.005 -tt 400
            done
        done
    done'
done

