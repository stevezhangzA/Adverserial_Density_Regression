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
from utils import soft_update

import torch.nn.functional as F
TensorBatch = List[torch.Tensor]
ENVS_WITH_GOAL = ("antmaze", "pen", "door", "hammer", "relocate")

def is_goal_reached(reward: float, info: Dict) -> bool:
    if "goal_achieved" in info:
        return info["goal_achieved"]
    return reward > 0  # Assuming that reaching target is a positive reward

@torch.no_grad()
def eval_actor(
        env: gym.Env, actor: nn.Module, device: str, n_episodes: int, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    env.seed(seed)
    actor.eval()
    episode_rewards = []
    successes = []
    for _ in range(n_episodes):
        state, done = env.reset(), False
        episode_reward = 0.0
        goal_achieved = False
        while not done:
            action = actor.act(state, device)
            state, reward, done, env_infos = env.step(action)
            episode_reward += reward
            if not goal_achieved:
                goal_achieved = is_goal_reached(reward, env_infos)
        # Valid only for environments with goal
        successes.append(float(goal_achieved))
        episode_rewards.append(episode_reward)
    actor.train()
    return np.asarray(episode_rewards), np.mean(successes)


@torch.no_grad()
def eval_kitchen_mix(
        env: gym.Env, actor: nn.Module, device: str, n_episodes: int, seed: int,total_stages:int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    This method is specialized for kitchen environment:

    """
    env.seed(seed)
    actor.eval()
    episode_rewards = []
    successes = []
    for _ in range(n_episodes):
        state, done = env.reset(), False
        episode_reward = []
        goal_achieved = False
        while not done:
            action = actor.act(state, device)
            state, reward, done, env_infos = env.step(action)
            episode_reward.append(reward)
            if not goal_achieved:
                goal_achieved = is_goal_reached(reward, env_infos)
        # Valid only for environments with goal
        successes.append(float(goal_achieved))
        stages_tage=list(set(episode_reward))
        if goal_achieved:
            score=len(stages_tage)
        else:
            score=len(stages_tage)-1
        episode_rewards.append(score)
    actor.train()
    return np.asarray(episode_rewards), np.mean(successes)

class adverseial_density:
    def __init__(self,
                max_action: float,
                actor:nn.Module,
                actor_optimizer: torch.optim.Optimizer,
                vae_good: nn.Module,
                vae_optimizer_good: torch.optim.Optimizer,
                vae_bad: nn.Module,
                vae_optimizer_bad: torch.optim.Optimizer,
                discount: float = 0.99,
                tau: float = 0.005,
                policy_noise: float = 0.2,
                noise_clip: float = 0.5,
                policy_freq: int = 2,
                beta: float = 0.5,
                lambd: float = 1.0,
                num_samples: int = 1,
                iwae: bool = False,
                lambd_cool: bool = False,
                lambd_end: float = 0.2,
                max_online_steps: int = 1_000_000,
                device: str = "cpu",
                weighted_estimation=False):
        """
        L=D(expert)-D(non_expert)
        """
        self.vae_good = vae_good
        self.actor=actor
        self.actor_optimizer = actor_optimizer
        self.vae_optimizer_good = vae_optimizer_good
        self.vae_bad = vae_bad
        self.vae_optimizer_bad = vae_optimizer_bad

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.beta = beta
        self.lambd = lambd
        self.num_samples = num_samples
        self.iwae = iwae
        self.lambd_cool = lambd_cool
        self.lambd_end = lambd_end
        self.max_online_steps = max_online_steps
        self.is_online = False
        self.online_it = 0
        self.total_it = 0
        self.device = device
        self.weighted_estimation=weighted_estimation

    def elbo_loss(
            self,
            vae,
            state: torch.Tensor,
            action: torch.Tensor,
            beta: float,
            num_samples: int = 1,
    ) -> torch.Tensor:
        """
        Note: elbo_loss one is proportional to elbo_estimator
        i.e. there exist a>0 and b, elbo_loss = a * (-elbo_estimator) + b
        """
        mean, std = vae.encode(state, action)
        mean_s = mean.repeat(num_samples, 1, 1).permute(1, 0, 2)  # [B x S x D]
        std_s = std.repeat(num_samples, 1, 1).permute(1, 0, 2)  # [B x S x D]
        z = mean_s + std_s * torch.randn_like(std_s)
        state = state.repeat(num_samples, 1, 1).permute(1, 0, 2)  # [B x S x C]
        action = action.repeat(num_samples, 1, 1).permute(1, 0, 2)  # [B x S x C]
        u = vae.decode(state, z)
        recon_loss = ((u - action) ** 2).mean(dim=(1, 2))
        KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean(-1)
        vae_loss = recon_loss + beta * KL_loss
        return vae_loss

    def dual_estimation_objective_optim(self,
                                        batch_suboptimal,
                                        batch_optimal,
                                        weight,
                                        num_samples: int = 10, ):
        suboptim_state, suboptim_action, _, _, _ = batch_suboptimal
        optim_state, optim_action, _, _, _ = batch_optimal
        sub_estimate = self.vae_good.importance_sampling_estimator(suboptim_state, suboptim_action, self.beta,
                                                                  num_samples)
        optim_estimate = self.vae_good.importance_sampling_estimator(optim_state, optim_action, self.beta, num_samples)
        if not self.weighted_estimation:
            loss = weight * (torch.nn.functional.sigmoid(torch.exp(sub_estimate)).mean() - torch.nn.functional.sigmoid(torch.exp(optim_estimate)).mean())
        else:
            with torch.no_grad():
                sub_optimal_weight=optim_estimate = self.vae_good.importance_sampling_estimator(suboptim_state, suboptim_action, self.beta, num_samples)
            loss = ((torch.ones_like(sub_optimal_weight)-sub_optimal_weight)*torch.nn.functional.sigmoid(torch.exp(sub_estimate))).mean() - torch.nn.functional.sigmoid(torch.exp(optim_estimate)).mean()            
        self.vae_optimizer_good.zero_grad()
        loss.backward()
        self.vae_optimizer_good.step()

    def dual_estimation_objective_suboptim(self,
                                           vae,
                                           batch_suboptimal,
                                           batch_optimal,
                                           weight,
                                           beta: float,
                                           num_samples: int = 10, ):
        suboptim_state, suboptim_action, _, _, _ = batch_suboptimal
        optim_state, optim_action, _, _, _ = batch_optimal
        sub_estimate = vae.importance_sampling_estimator(suboptim_state, suboptim_action, beta, num_samples)
        optim_estimate = vae.importance_sampling_estimator(optim_state, optim_action, beta, num_samples)
        # if not self.weighted_estimation:
        loss = torch.nn.sigmod()(torch.exp(sub_estimate)).mean() - torch.nn.sigmod()(torch.exp(optim_estimate)).mean()
        #else:
        #    sub_optimal_weight=torch.exp(sub_estimate).detach()
        #    loss = (1-sub_optimal_weight)*torch.nn.sigmod()(torch.exp(sub_estimate)).mean() - torch.nn.Sigmod()(torch.exp(optim_estimate)).mean()
        # print(loss)
        self.vae_optimizer_bad.zero_grad()
        weight * loss.backward()
        self.vae_optimizer_bad.step()

    def iwae_loss(self, vae, 
                  state: torch.Tensor,
                  action: torch.Tensor,
                  beta: float,
                  num_samples: int = 10) -> torch.Tensor:
        ll = vae.importance_sampling_estimator(state, action, beta, num_samples)
        return -ll

    def vae_bad_train(self, batch: TensorBatch) -> Dict[str, float]:
        log_dict = {}
        self.total_it += 1
        state, action, _, _, _ = batch
        # Variational Auto-Encoder Training
        recon, mean, std = self.vae_bad(state, action)

        
        recon_loss = F.mse_loss(recon, action)
        KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
        vae_loss = recon_loss + self.beta * KL_loss
        self.vae_optimizer_bad.zero_grad()
        vae_loss.backward()
        self.vae_optimizer_bad.step()
        log_dict["VAE/reconstruction_loss"] = recon_loss.item()
        log_dict["VAE/KL_loss"] = KL_loss.item()
        log_dict["VAE/vae_loss"] = vae_loss.item()
        return log_dict

    def vae_good_train(self, batch: TensorBatch) -> Dict[str, float]:
        log_dict = {}
        self.total_it += 1
        state, action, _, _, _ = batch
        # Variational Auto-Encoder Training
        recon, mean, std = self.vae_good(state, action)
        recon_loss = F.mse_loss(recon, action)
        KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
        vae_loss = recon_loss + self.beta * KL_loss
        self.vae_optimizer_good.zero_grad()
        vae_loss.backward()
        self.vae_optimizer_good.step()
        log_dict["VAE/reconstruction_loss"] = recon_loss.item()
        log_dict["VAE/KL_loss"] = KL_loss.item()
        log_dict["VAE/vae_loss"] = vae_loss.item()
        return log_dict


    def vqvae_bad_train(self, batch: TensorBatch) -> Dict[str, float]:
        log_dict = {}
        self.total_it += 1
        state, action, _, _, _ = batch
        # Variational Auto-Encoder Training
        recon, mean, std,mid_z, z = self.vae_bad(state, action)
        vq_loss=nn.MSELoss()(mid_z,z.detach())+nn.MSELoss()(z,mid_z.detach())
        recon_loss = F.mse_loss(recon, action)
        KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
        vae_loss = recon_loss + self.beta * KL_loss+vq_loss
        self.vae_optimizer_bad.zero_grad()
        vae_loss.backward()
        self.vae_optimizer_bad.step()
        log_dict["VAE/reconstruction_loss"] = recon_loss.item()
        log_dict["VAE/KL_loss"] = KL_loss.item()
        log_dict["VAE/vae_loss"] = vae_loss.item()
        return log_dict

    def vqvae_good_train(self, batch: TensorBatch) -> Dict[str, float]:
        log_dict = {}
        self.total_it += 1
        state, action, _, _, _ = batch
        # Variational Auto-Encoder Training
        recon, mean, std,mid_z, z = self.vae_good(state, action)
        vq_loss=nn.MSELoss()(mid_z,z.detach())+nn.MSELoss()(z,mid_z.detach())
        recon_loss = F.mse_loss(recon, action)
        KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
        vae_loss = recon_loss + self.beta * KL_loss+vq_loss
        self.vae_optimizer_good.zero_grad()
        vae_loss.backward()
        self.vae_optimizer_good.step()
        log_dict["VAE/reconstruction_loss"] = recon_loss.item()
        log_dict["VAE/KL_loss"] = KL_loss.item()
        log_dict["VAE/vae_loss"] = vae_loss.item()
        return log_dict

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        """
        this is one epoch offline training:
        """
        log_dict = {}
        self.total_it += 1
        if self.is_online:
            self.online_it += 1
        state, action, reward, next_state, done = batch
        if self.total_it % self.policy_freq == 0:
            pi = self.actor(state)
            with torch.no_grad():
                # estimate the expert liklihood
                neg_log_beta_good = self.vae_good.importance_sampling_estimator(state, action, self.beta,
                                                                                self.num_samples)
                # estimate the non-expert liklihood
                neg_log_beta_bad = self.vae_bad.importance_sampling_estimator(state, action, self.beta,
                                                                              self.num_samples)
                # computing the density weight
                neg_log_beta = neg_log_beta_bad - neg_log_beta_good
                if self.lambd_cool:
                    lambd = self.lambd * max(self.lambd_end, (1.0 - self.online_it / self.max_online_steps))
                else:
                    lambd = self.lambd
            # actor_loss= |\pi(\cdot|s)-a|*log P(suboptimal action|s)-log P(expert action|s)
            # reduction="none"
            actor_loss = torch.nn.MSELoss()(pi, action) * lambd * neg_log_beta.mean() # upper bound
            #actor_loss = (torch.nn.MSELoss(reduction="none")(pi, action) * lambd * neg_log_beta.view(64,1)).sum() # lower bound
            log_dict["actor_loss"] = actor_loss.item()
            log_dict["neg_log_beta_mean"] = neg_log_beta.mean().item()
            #log_dict["neg_log_beta_max"] = neg_log_beta.max().item()
            log_dict["lambd"] = lambd
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
        return log_dict
    
    def max_train(self, batch: TensorBatch) -> Dict[str, float]:
        """
        this is one epoch offline training:
        """
        log_dict = {}
        self.total_it += 1
        if self.is_online:
            self.online_it += 1
        state, action, reward, next_state, done = batch
        if self.total_it % self.policy_freq == 0:
            pi = self.actor(state)

            neg_log_beta_good = self.vae_good.importance_sampling_estimator(state, pi, self.beta,
                                                                                self.num_samples)

            actor_loss=-1*neg_log_beta_good.sum()
            # actor_loss= |\pi(\cdot|s)-a|*log P(suboptimal action|s)-log P(expert action|s)
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
        return log_dict
    # likelihood_computing
    def max_diverge(self, batch: TensorBatch) -> Dict[str, float]:
        """
        this is one epoch offline training:
        """
        log_dict = {}
        self.total_it += 1
        if self.is_online:
            self.online_it += 1
        state, action, reward, next_state, done = batch
        if self.total_it % self.policy_freq == 0:
            pi = self.actor(state)
            # good likelihood
            neg_log_beta_good = self.vae_good.importance_sampling_estimator(state, pi, self.beta,
                                                                                self.num_samples)
            # bad likelihood
            neg_log_beta_bad = self.vae_bad.importance_sampling_estimator(state, pi, self.beta,
                                                                                self.num_samples)

            actor_loss=neg_log_beta_bad.sum()-neg_log_beta_good.sum()
            # actor_loss= |\pi(\cdot|s)-a|*log P(suboptimal action|s)-log P(expert action|s)
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
        return log_dict

    def max_kl_diverge(self, batch: TensorBatch) -> Dict[str, float]:
        """
        this is one epoch offline training:
        """
        log_dict = {}
        self.total_it += 1
        if self.is_online:
            self.online_it += 1
        state, action, reward, next_state, done = batch
        if self.total_it % self.policy_freq == 0:
            # good likelihood
            # likelihood_computing(self,
            #                 state: torch.Tensor,
            #                 action: torch.Tensor,
            #                 beta: float,
            #                 num_samples: int = 500)
            neg_log_beta_good = self.vae_good.likelihood_computing(state, action, self.beta,
                                                                self.num_samples)
            # bad likelihood
            neg_log_beta_bad = self.vae_bad.likelihood_computing(state, action, self.beta,
                                                                self.num_samples)
            pi = self.actor(state)

            actor_loss=F.kl_div(neg_log_beta_bad, pi, reduction='sum')-F.kl_div(neg_log_beta_good, pi, reduction='sum')
            # actor_loss= |\pi(\cdot|s)-a|*log P(suboptimal action|s)-log P(expert action|s)
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
        return log_dict

    def train_density_regression(self, batch: TensorBatch) -> Dict[str, float]:
        """
        this is one epoch offline training:
        """
        log_dict = {}
        self.total_it += 1
        if self.is_online:
            self.online_it += 1
        state, action, reward, next_state, done = batch
        if self.total_it % self.policy_freq == 0:
            pi = self.actor(state)

            neg_log_beta_good = self.vae_good.importance_sampling_estimator(state, pi, self.beta,
                                                                            self.num_samples)

            # actor_loss= |\pi(\cdot|s)-a|*log P(suboptimal action|s)-log P(expert action|s)
            actor_loss = torch.nn.MSELoss()(pi, action) +neg_log_beta_good.mean()
            log_dict["actor_loss"] = actor_loss.item()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
        return log_dict
    
    def state_dict(self) -> Dict[str, Any]:
        return {"vae": self.vae.state_dict(),
                "vae_optimizer": self.vae_optimizer.state_dict(),
                "critic_1": self.critic_1.state_dict(),
                "critic_1_optimizer": self.critic_1_optimizer.state_dict(),
                "critic_2": self.critic_2.state_dict(),
                "critic_2_optimizer": self.critic_2_optimizer.state_dict(),
                "actor": self.actor.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "total_it": self.total_it}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.vae.load_state_dict(state_dict["vae"])
        self.vae_optimizer.load_state_dict(state_dict["vae_optimizer"])

        self.critic_1.load_state_dict(state_dict["critic_1"])
        self.critic_1_optimizer.load_state_dict(state_dict["critic_1_optimizer"])
        self.critic_1_target = copy.deepcopy(self.critic_1)

        self.critic_2.load_state_dict(state_dict["critic_2"])
        self.critic_2_optimizer.load_state_dict(state_dict["critic_2_optimizer"])
        self.critic_2_target = copy.deepcopy(self.critic_2)

        self.actor.load_state_dict(state_dict["actor"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.actor_target = copy.deepcopy(self.actor)

        self.total_it = state_dict["total_it"]

class adverseial_density_abilition:
    def __init__(self,
                max_action: float,
                actor:nn.Module,
                actor_optimizer: torch.optim.Optimizer,
                ADE,
                ADE_optimizer,
                vae_exprt,
                vae_optimizer_exprt,
                vae_demo,
                vae_optimizer_demo ,
                vae_mixture,
                vae_optimizer_mixture,
                discount: float = 0.99,
                tau: float = 0.005,
                policy_noise: float = 0.2,
                noise_clip: float = 0.5,
                policy_freq: int = 2,
                beta: float = 0.5,
                lambd: float = 1.0,
                num_samples: int = 1,
                iwae: bool = False,
                lambd_cool: bool = False,
                lambd_end: float = 0.2,
                max_online_steps: int = 1_000_000,
                device: str = "cpu",
                weighted_estimation=False):
        """
        L=D(expert)-D(non_expert)
        """
        self.actor=actor
        self.actor_optimizer = actor_optimizer

        self.ADE=ADE
        self.ADE_optimizer=ADE_optimizer
        self.vae_exprt=vae_exprt
        self.vae_optimizer_exprt=vae_optimizer_exprt
        self.vae_demo=vae_demo
        self.vae_optimizer_demo=vae_optimizer_demo
        self.vae_mixture=vae_mixture
        self.vae_optimizer_mixture=vae_optimizer_mixture

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.beta = beta
        self.lambd = lambd
        self.num_samples = num_samples
        self.iwae = iwae
        self.lambd_cool = lambd_cool
        self.lambd_end = lambd_end
        self.max_online_steps = max_online_steps
        self.is_online = False
        self.online_it = 0
        self.total_it = 0
        self.device = device
        self.weighted_estimation=weighted_estimation

    def elbo_loss(
            self,
            vae,
            state: torch.Tensor,
            action: torch.Tensor,
            beta: float,
            num_samples: int = 1,
    ) -> torch.Tensor:
        """
        Note: elbo_loss one is proportional to elbo_estimator
        i.e. there exist a>0 and b, elbo_loss = a * (-elbo_estimator) + b
        """
        mean, std = vae.encode(state, action)
        mean_s = mean.repeat(num_samples, 1, 1).permute(1, 0, 2)  # [B x S x D]
        std_s = std.repeat(num_samples, 1, 1).permute(1, 0, 2)  # [B x S x D]
        z = mean_s + std_s * torch.randn_like(std_s)
        state = state.repeat(num_samples, 1, 1).permute(1, 0, 2)  # [B x S x C]
        action = action.repeat(num_samples, 1, 1).permute(1, 0, 2)  # [B x S x C]
        u = vae.decode(state, z)
        recon_loss = ((u - action) ** 2).mean(dim=(1, 2))
        KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean(-1)
        vae_loss = recon_loss + beta * KL_loss
        return vae_loss

    def dual_estimation_objective_optim(self,
                                        vae_good,
                                        vae_optimizer_good,
                                        batch_suboptimal,
                                        batch_optimal,
                                        weight,
                                        num_samples: int = 10, ):
        """
        this function elevate the training performance via adverserial method
        """
        suboptim_state, suboptim_action, _, _, _ = batch_suboptimal
        optim_state, optim_action, _, _, _ = batch_optimal
        sub_estimate = vae_good.importance_sampling_estimator(suboptim_state, suboptim_action, self.beta,
                                                                  num_samples)
        optim_estimate = vae_good.importance_sampling_estimator(optim_state, optim_action, self.beta, num_samples)

        loss = weight * (torch.nn.functional.sigmoid(torch.exp(sub_estimate)).mean() - torch.nn.functional.sigmoid(torch.exp(optim_estimate)).mean())
      
        vae_optimizer_good.zero_grad()
        loss.backward()
        vae_optimizer_good.step()

    def vae_train(self,
                  vae,
                  vae_optimizer, 
                  batch: TensorBatch) -> Dict[str, float]:
        """
        this method train vae outside this function.
        """
        log_dict = {}
        self.total_it += 1
        state, action, _, _, _ = batch
        # Variational Auto-Encoder Training
        recon, mean, std = vae(state, action)
        recon_loss = F.mse_loss(recon, action)
        KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
        vae_loss = recon_loss + self.beta * KL_loss
        vae_optimizer.zero_grad()
        vae_loss.backward()
        vae_optimizer.step()
        log_dict["VAE/reconstruction_loss"] = recon_loss.item()
        log_dict["VAE/KL_loss"] = KL_loss.item()
        log_dict["VAE/vae_loss"] = vae_loss.item()
        return log_dict

    def iwae_loss(self, vae, 
                  state: torch.Tensor,
                  action: torch.Tensor,
                  beta: float,
                  num_samples: int = 10) -> torch.Tensor:
        ll = vae.importance_sampling_estimator(state, action, beta, num_samples)
        return -ll
