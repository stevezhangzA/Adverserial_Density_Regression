import pickle as pkl
import numpy as np


from matplotlib import pyplot as plt
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

import os
import random

import copy
import uuid

TensorBatch = List[torch.Tensor]
ENVS_WITH_GOAL = ("antmaze", "pen", "door", "hammer", "relocate")

def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name"],
        id=str(uuid.uuid4()),
    )

    wandb.run.save()
def set_env_seed(env: Optional[gym.Env], seed: int):
    env.seed(seed)
    env.action_space.seed(seed)

def set_seed(
        seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False
):
    if env is not None:
        set_env_seed(env, seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std


def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (states - mean) / std


def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
    reward_scale: float = 1.0,
) -> gym.Env:
    # PEP 8: E731 do not assign a lambda expression, use a def
    def normalize_state(state):
        return (
            state - state_mean
        ) / state_std  # epsilon should be already added in std.

    def scale_reward(reward):
        # Please be careful, here reward is multiplied by scale!
        return reward_scale * reward

    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env

# stiched_good,stiched_bad=load_d4rl_dataset(all_trails,cluster_eps)
def split_d4rl_datasets(trails,
                        cluster_eps):
    def split_trails(trails,
                        cluster_eps):
        all_trails=[]
        new_collecter={}
        for key in trails:
            new_collecter[key]=[]
        #terminations=trails['terminals']
        for seq_id, done in enumerate(trails['teminals']):
            if done:
                for key in trails:
                    new_collecter[key].append(trails[key][seq_id])
                all_trails.append(new_collecter)
                new_collecter={}
                for key in trails:
                    new_collecter[key]=[]
            else:
                for key in trails:
                    new_collecter[key].append(trails[key][seq_id])
        return_list=[sum(trail['rewards']) for trail in all_trails]
        #index=list(range(return_list))
        sorted_index=np.argsort(return_list)
        
        sorted_trails=[all_trails[idx] for idx in sorted_index]
        sorted_return=[return_list[idx] for idx in sorted_index]

        max_return=sorted_return[sorted_index[-1]]

        good_trail=[]
        bad_trail=[]

        for return_, trail in zip(sorted_return, sorted_trails):
            if abs(max_return-return_)<=cluster_eps:
                good_trail.append(trail)
            else:
                bad_trail.append(trail)
        stiched_good={key:[] for key in trails}
        stiched_bad=copy.deepcopy(stiched_good)
        for good_splice in good_trail:
            for k in good_splice:
                stiched_good[k]+=good_splice[k]
        for bad_splice in bad_trail:
            for k in bad_splice:
                stiched_bad[k]+=bad_splice[k]
        return stiched_good,stiched_bad

    stiched_good,stiched_bad=split_trails(trails,
                                          cluster_eps)
    
    for k in stiched_good:
        stiched_good[k]=np.array(stiched_good[k])
        stiched_bad[k]=np.array(stiched_bad[k])
        print(stiched_good[k].shape)
        print(stiched_bad[k].shape)
    return stiched_good,stiched_bad

def filter_expert(dataset,
                  topk):
    def initialized_empty_container(dataset):
        empty_container = {}
        for k in dataset:
            empty_container[k] = []
        return empty_container

    empty_container = initialized_empty_container(dataset)
    new_empty_container = copy.deepcopy(empty_container)
    all_trails = []
    for id_, done in enumerate(dataset['terminals']):
        print(id_,done)
        if done:
            for k in new_empty_container:
                new_empty_container[k].append(dataset[k][id_])
            all_trails.append(new_empty_container)
            new_empty_container = copy.deepcopy(empty_container)
        else:
            for k in new_empty_container:
                new_empty_container[k].append(dataset[k][id_])
            if id_==len(dataset['terminals'])-1:
                all_trails.append(new_empty_container)
    return_list = [sum(trail['rewards']) for trail in all_trails]
    #print(dataset)
    #print(all_trails)
    sorted_id = np.argsort(return_list)
    selected_trails = []
    for real_id, id_ in enumerate(sorted_id[-topk:]):
        #if id_ < topk:
        #if len(all_trails[real_id]['rewards'])==0:
        #    continue
        #print('*'*10)
        #for k in all_trails[real_id]:
        #    print(len(all_trails[real_id][k]))
        selected_trails.append(all_trails[real_id])
    # kd = collections.OrderedDict(sorted(dd.items(), key=lambda t: t[0]))
    for trail in selected_trails:
        for k in empty_container:
            empty_container[k].extend(trail[k])
    for k in empty_container:
        empty_container[k] = np.array(empty_container[k])
    return empty_container

class ReplayBuffer:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        device: str = "cpu",
    ):  
        device='cpu'
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0

        self._states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._actions = torch.zeros(
            (buffer_size, action_dim), dtype=torch.float32, device=device
        )
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._next_states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._device = device

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    # Loads data in d4rl format, i.e. from Dict[str, np.array].
    def load_d4rl_dataset(self, data: Dict[str, np.ndarray]):
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = data["observations"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError(
                "Replay buffer is smaller than the dataset you are trying to load!"
            )
        #print(self._states.size(),data["observations"].shape)
        for k in data:
            print(data[k].shape)
        self._states[:n_transitions] = self._to_tensor(data["observations"])
        self._actions[:n_transitions] = self._to_tensor(data["actions"])
        self._rewards[:n_transitions] = self._to_tensor(data["rewards"][..., None])
        self._next_states[:n_transitions] = self._to_tensor(data["next_observations"])
        self._dones[:n_transitions] = self._to_tensor(data["terminals"][..., None])
        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)
        print(f"Dataset size: {n_transitions}")

    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, self._size, size=batch_size)
        states = self._states[indices].to('cuda')
        actions = self._actions[indices].to('cuda')
        rewards = self._rewards[indices].to('cuda')
        next_states = self._next_states[indices].to('cuda')
        dones = self._dones[indices].to('cuda')
        return [states, actions, rewards, next_states, dones]

    def add_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        # Use this method to add new data into the replay buffer during fine-tuning.
        self._states[self._pointer] = self._to_tensor(state)
        self._actions[self._pointer] = self._to_tensor(action)
        self._rewards[self._pointer] = self._to_tensor(reward)
        self._next_states[self._pointer] = self._to_tensor(next_state)
        self._dones[self._pointer] = self._to_tensor(done)

        self._pointer = (self._pointer + 1) % self._buffer_size
        self._size = min(self._size + 1, self._buffer_size)

def return_reward_range(dataset: Dict, max_episode_steps: int) -> Tuple[float, float]:
    returns, lengths = [], []
    ep_ret, ep_len = 0.0, 0
    for r, d in zip(dataset["rewards"], dataset["terminals"]):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0.0, 0
    lengths.append(ep_len)  # but still keep track of number of steps
    assert sum(lengths) == len(dataset["rewards"])
    return min(returns), max(returns)


def modify_reward(dataset: Dict, env_name: str, max_episode_steps: int = 1000) -> Dict:
    if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
        min_ret, max_ret = return_reward_range(dataset, max_episode_steps)
        dataset["rewards"] /= max_ret - min_ret
        dataset["rewards"] *= max_episode_steps
        return {
            "max_ret": max_ret,
            "min_ret": min_ret,
            "max_episode_steps": max_episode_steps,
        }
    elif "antmaze" in env_name:
        dataset["rewards"] -= 1.0
    return {}


def modify_reward_online(reward: float, env_name: str, **kwargs) -> float:
    if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
        reward /= kwargs["max_ret"] - kwargs["min_ret"]
        reward *= kwargs["max_episode_steps"]
    elif "antmaze" in env_name:
        reward -= 1.0
    return reward
