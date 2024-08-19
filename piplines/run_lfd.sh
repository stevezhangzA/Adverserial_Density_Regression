wandb disabled
CUDA_VISIBLE_DEVICES=$1 python imitation_learning.py --log_saving_dir $2 \
													 --project_name $3 \
	                                             	 --lambd $4 \
													 --topk $5 \
													 --seed $6 \
													 --env $7 \
    												 --expert_data $8 \


#sh piplines/run_lfd.sh 0 lfd hopper_m 1 20 0 hopper-medium-v2 hopper-expert-v2 
#sh piplines/run_lfd.sh 1 lfd walker_m 1 20 0 walker2d-medium-v2 walker2d-expert-v2 
#sh piplines/run_lfd.sh 2 lfd cheetach_m 1 20 0 halfcheetah-medium-v2 halfcheetah-expert-v2 
#sh piplines/run_lfd.sh 3 lfd ant_m 1 20 0 ant-medium-v2 ant-expert-v2 


#sh piplines/run_lfd.sh 4 lfd hopper_mr 1 20 0 hopper-medium-replay-v2 hopper-expert-v2 
#sh piplines/run_lfd.sh 5 lfd walker_mr 1 20 0 walker2d-medium-replay-v2 walker2d-expert-v2 
#sh piplines/run_lfd.sh 6 lfd cheetah_mr 1 20 0 halfcheetah-medium-replay-v2 halfcheetah-expert-v2
#sh piplines/run_lfd.sh 7 lfd ant_mr 1 20 0 ant-medium-replay-v2 ant-expert-v2

#sh piplines/run_lfd.sh 0 lfd hopper_me 1 20 0 hopper-medium-expert-v2 hopper-expert-v2 
#sh piplines/run_lfd.sh 1 lfd walker_me 1 20 0 walker2d-medium-expert-v2 walker2d-expert-v2 
#sh piplines/run_lfd.sh 2 lfd cheetah_me 1 20 0 halfcheetah-medium-expert-v2 halfcheetah-expert-v2 
#sh piplines/run_lfd.sh 3 lfd ant_me 1 20 0 ant-medium-expert-v2 ant-expert-v2 

