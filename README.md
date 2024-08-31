# Adverserial Densisty Weighted Regression Behavior Cloning [Paper Link](https://arxiv.org/abs/2405.20351)

Description: The Official Implementation of Adverserial Densisty Weighted Regression Behavior Cloning

## Configurations

requirement  | version | 
--------  | ----- |
python | 3.8 |
system (platform) | ubuntu/Linux |
PyTorch | 2.0.0 |
D4RL | 1.1 |
Gym| -|
 
### running code & obtain the mean value as well as confidence interval

```c
# for obtaining our loggings
sh piplines/run_lfd_vq_reward_spares.sh
# for obtaining the mean and confidence intervals 
python build_table.py --folder_name 'lfd_1_vq'
```


Besides, if you chose to utilize our code base please cite below:

```
@{zhang2024adrbcadversarialdensityweighted,
      title={ADR-BC: Adversarial Density Weighted Regression Behavior Cloning}, 
      author={Ziqi Zhang and Zifeng Zhuang and Donglin Wang and Jingzehua Xu and Miao Liu and Shuai Zhang},
      year={2024},
      eprint={2405.20351},
      journal={arXiv preprint arXiv:2405.20351},
      primaryClass={cs.LG},
}
```


# Thanks

Our codebase is modified from CORL [Paper Link](https://openreview.net/forum?id=SyAS49bBcv)
