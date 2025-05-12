

prediction_length=121 # 31

exp_dir='./exp'
config='Triton' # Triton Fourcastnet
run_num='20250217-221700'
finetune_dir='3_steps_finetune'
year=2020
ics_type='default' 

CUDA_VISIBLE_DEVICES=1 python inference.py --exp_dir=${exp_dir} --config=${config} --run_num=${run_num} --finetune_dir=$finetune_dir --prediction_length=${prediction_length} --ics_type=${ics_type} --year=${year} 



