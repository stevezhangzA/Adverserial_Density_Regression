# Description: The Official Implementation of Imitating from auxiliary imperfect demonstrations via Adversarial Density Weighted Regression

## [Paper Link](https://arxiv.org/abs/2405.20351)

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
python build_table.py
```

### Raw experimental data

We have released our loggings at 'lfd_1_vq'

Besides, if you chose to utilize our code base please cite below:

### Citations

```
@misc{zhang2025imitatingauxiliaryimperfectdemonstrations,
      title={Imitating from auxiliary imperfect demonstrations via Adversarial Density Weighted Regression}, 
      author={Ziqi Zhang and Zifeng Zhuang and Jingzehua Xu and Yiyuan Yang and Yubo Huang and Donglin Wang and Shuai Zhang},
      year={2025},
      eprint={2405.20351},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2405.20351}, 
}
```


# Thanks

Our codebase is modified from CORL [Paper Link](https://openreview.net/forum?id=SyAS49bBcv)


