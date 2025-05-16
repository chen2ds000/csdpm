#!/bin/bash //
train_method="gdpo"
val_method="ppo"
lr=1e-4
innerloop=1
sampleloop=2
step_freq=1
fix=0.5
# for run_times in `seq 1 1`:
# do
#     python main_generate.py -m general.test_method=$method general.seed=$RANDOM
# done
python main_generate.py -m dataset="math1" +experiment=math1_ppo.yaml general.fix=0.5 train.lr=0.001 general.train_method="gdpo" general.val_method=null general.step_freq=1 general.innerloop=1 general.sampleloop=1  general.seed=$RANDOM