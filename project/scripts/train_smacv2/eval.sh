#!/bin/bash
env="StarCraft2v2"
map="10gen_protoss"
algo="ippo"
units="20v20"

exp="protoss"
seed_max=1

export SC2PATH='/workspace/3rdparty/StarCraftII'
export MAP_DIR="$SC2PATH/Maps/"
echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python GTDE/scripts/eval/eval.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --map_name ${map} --seed ${seed} --units ${units} --n_eval_rollout_threads 1 --n_rollout_threads 1 --num_mini_batch 1 --episode_length 400 \
    --num_env_steps 10000000 --ppo_epoch 5 --use_value_active_masks --use_eval --eval_episodes 32
done

