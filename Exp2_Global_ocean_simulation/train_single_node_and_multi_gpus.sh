

wandb_group='triton'
yaml_config='config/Model.yaml'
config='Triton' 
batch_size=8
run_num=$(date "+%Y%m%d-%H%M%S")
multi_steps_finetune=1
finetune_max_epochs=0

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 train.py --yaml_config=$yaml_config --config=$config --run_num=$run_num --batch_size=$batch_size --multi_steps_finetune=$multi_steps_finetune --finetune_max_epochs=$finetune_max_epochs >> ./logs/${config}_${wandb_group}_rank0_${SLURM_JOB_ID}_${run_num}.log 2>&1 &

