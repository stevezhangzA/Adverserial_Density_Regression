
# source: https://github.com/thuml/SPOT/tree/58c591dc48fbd9ff632b7494eab4caf778e86f4a
# https://arxiv.org/pdf/2202.06239.pdf
import copy
import os
import random
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import d4rl
import gym
import numpy as np
import pyrallis
import torch
import torch.distributions as td
import torch.nn as nn
import torch.nn.functional as F
import wandb

import pickle as pkl

TensorBatch = List[torch.Tensor]
ENVS_WITH_GOAL = ("antmaze", "pen", "door", "hammer", "relocate")


@dataclass
class TrainConfig:
    # Experiment
    device: str = "cuda"
    env: str = "hopper-medium-replay-v2"  # OpenAI gym environment name
    expert_data: str = 'hopper-expert-v2'
    topk: int = 10
    seed: int = 0  # Sets Gym, PyTorch and Numpy seeds
    eval_seed: int = 0  # Eval environment seed
    eval_freq: int = int(1e3)  # How often (time steps) we evaluate
    n_episodes: int = 10  # How many episodes run during evaluation
    offline_iterations: int = int(1e6)  # Number of offline updates
    online_iterations: int = 0  # int(1e6)  # Number of online updates
    checkpoints_path: Optional[str] = None  # Save path
    load_model: str = ""  # Model load file name, "" doesn't load
    # TD3
    actor_lr: float = 1e-4  # Actor learning ratev
    critic_lr: float = 3e-4  # Actor learning rate
    buffer_size: int = 20_000_000  # Replay buffer size
    batch_size: int = 64  # Batch size for all networks
    discount: float = 0.99  # Discount factor
    expl_noise: float = 0.1  # Std of Gaussian exploration noise
    tau: float = 0.005  # Target network update rate
    policy_noise: float = 0.2  # Noise added to target actor during critic update
    noise_clip: float = 0.5  # Range to clip target actor noise
    policy_freq: int = 2  # Frequency of delayed actor updates
    # SPOT VAE
    vae_lr: float = 1e-3  # VAE learning rate
    vae_hidden_dim: int = 750  # VAE hidden layers dimension
    vae_latent_dim: Optional[int] = None  # VAE latent space, 2 * action_dim if None
    beta: float = 0.5  # KL loss weight
    vae_iterations: int = 100_000  # Number of VAE training updates
    # SPOT
    actor_init_w: Optional[float] = None  # Actor head init parameter
    critic_init_w: Optional[float] = None  # Critic head init parameter
    lambd: float = 1.0  # Support constraint weight
    num_samples: int = 1  # Number of samples for density estimation
    iwae: bool = False  # Use IWAE loss
    lambd_cool: bool = False  # Cooling lambda during fine-tune
    lambd_end: float = 0.2  # Minimal value of lambda
    normalize: bool = False  # Normalize states
    normalize_reward: bool = True  # Normalize reward
    online_discount: float = 0.995  # Discount for online tuning
    # Wandb logging
    project: str = "CORL"
    group: str = "SPOT-D4RL"
    name: str = "SPOT"
    # new_added params
    split_eps = 100
    dual_regresss_weight = 1
    log_saving_dir: str = ''
    project_name:str=''
    weighted_estimation:bool=False
    num_embeddings:int=4096
    use_vqvae:bool=False
    reward_sparse:bool=False
    adverserial_weight:float=1.0
    def __post_init__(self):
        self.name = f"{self.name}-{self.env}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)
