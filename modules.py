
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



def weights_init(m: nn.Module, init_w: float = 3e-3):
    if isinstance(m, nn.Linear):
        m.weight.data.uniform_(-init_w, init_w)
        m.bias.data.uniform_(-init_w, init_w)



class VAE(nn.Module):
    # Vanilla Variational Auto-Encoder

    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            latent_dim: int,
            max_action: float,
            hidden_dim: int = 750,
    ):
        super(VAE, self).__init__()
        if latent_dim is None:
            latent_dim = 2 * action_dim

        self.encoder_shared = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.log_std = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(state_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

        self.max_action = max_action
        self.latent_dim = latent_dim
        self.device = 'cuda'

    def forward(
            self,
            state: torch.Tensor,
            action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, std = self.encode(state, action)
        z = mean + std * torch.randn_like(std)
        u = self.decode(state, z)
        return u, mean, std

    def importance_sampling_estimator(
            self,
            state: torch.Tensor,
            action: torch.Tensor,
            beta: float,
            num_samples: int = 500,
    ) -> torch.Tensor:
        # * num_samples correspond to num of samples L in the paper
        # * note that for exact value for \hat \log \pi_\beta in the paper
        # we also need **an expection over L samples**
        mean, std = self.encode(state, action)

        mean_enc = mean.repeat(num_samples, 1, 1).permute(1, 0, 2)  # [B x S x D]
        std_enc = std.repeat(num_samples, 1, 1).permute(1, 0, 2)  # [B x S x D]
        z = mean_enc + std_enc * torch.randn_like(std_enc)  # [B x S x D]

        state = state.repeat(num_samples, 1, 1).permute(1, 0, 2)  # [B x S x C]
        action = action.repeat(num_samples, 1, 1).permute(1, 0, 2)  # [B x S x C]
        mean_dec = self.decode(state, z)
        std_dec = np.sqrt(beta / 4)

        # Find q(z|x)
        log_qzx = td.Normal(loc=mean_enc, scale=std_enc).log_prob(z)
        # Find p(z)
        mu_prior = torch.zeros_like(z).to(self.device)
        std_prior = torch.ones_like(z).to(self.device)
        log_pz = td.Normal(loc=mu_prior, scale=std_prior).log_prob(z)
        # Find p(x|z)
        std_dec = torch.ones_like(mean_dec).to(self.device) * std_dec
        log_pxz = td.Normal(loc=mean_dec, scale=std_dec).log_prob(action)

        w = log_pxz.sum(-1) + log_pz.sum(-1) - log_qzx.sum(-1)
        ll = w.logsumexp(dim=-1) - np.log(num_samples)
        return ll

    def likelihood_computing(self,
                             state: torch.Tensor,
                             action: torch.Tensor,
                             beta: float,
                             num_samples: int = 500) -> torch.Tensor:
        # * num_samples correspond to num of samples L in the paper
        # * note that for exact value for \hat \log \pi_\beta in the paper
        # we also need **an expection over L samples**
        logprob=self.importance_sampling_estimator(state,action,beta,num_samples)

        return torch.exp(logprob)

    def encode(
            self,
            state: torch.Tensor,
            action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder_shared(torch.cat([state, action], -1))

        mean = self.mean(z)
        # Clamped for numerical stability
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        return mean, std

    def decode(
            self,
            state: torch.Tensor,
            z: torch.Tensor = None,
    ) -> torch.Tensor:
        # When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
        if z is None:
            z = (
                torch.randn((state.shape[0], self.latent_dim))
                    .to(self.device)
                    .clamp(-0.5, 0.5)
            )
        x = torch.cat([state, z], -1)
        return self.max_action * self.decoder(x)
    

class VQVAE(nn.Module):
    # Vanilla Variational Auto-Encoder
    def __init__(self,state_dim: int,
                      action_dim: int,
                      latent_dim: int,
                      max_action: float,
                      hidden_dim: int = 750,
                      num_embeddings:int=256):
        super(VQVAE, self).__init__()
        
        if latent_dim is None:
            latent_dim = 2 * action_dim
        # encoder is responsible for the encoding the mid status of input's representation 
        self.encoder_shared = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )
        # then we sample the embeddings from the generated distribution.
        self.mean = nn.Linear(latent_dim, latent_dim)
        self.log_std = nn.Linear(latent_dim, latent_dim)
        # decoder is to utilize the condition to relocate the state actions.
        self.decoder = nn.Sequential(
            nn.Linear(state_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )
        # initialize hyperparameters
        self.max_action = max_action
        self.latent_dim = latent_dim
        self.device = 'cuda'
        # initialize the tabular embeddings.
        self.embeddings=nn.Embedding(num_embeddings,latent_dim)

    def forward(
            self,
            state: torch.Tensor,
            action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, std,mid_z, z = self.encode(state, action)
        z = mean + std * torch.randn_like(std)
        u = self.decode(state, z)
        return u, mean, std

    def get_code_indices(self,embedded_latents):
        # similarity=torch.nn.cosime
        # print(embedded_latents.size(),self.embeddings.weight.size())
        distances = (torch.sum(embedded_latents ** 2, dim=-1, keepdim=True) +
                     torch.sum(self.embeddings.weight ** 2, dim=-1) -
                     2 * torch.matmul(embedded_latents, self.embeddings.weight.t()))
        #encoding_index=torch.argmin(distances,dim=-1)
        return torch.argmin(distances, dim=1)
    
    def quantize(self,encoding_indices):
        return self.embeddings(encoding_indices)


    def importance_sampling_estimator(
            self,
            state: torch.Tensor,
            action: torch.Tensor,
            beta: float,
            num_samples: int = 500,
    ) -> torch.Tensor:
        # * num_samples correspond to num of samples L in the paper
        # * note that for exact value for \hat \log \pi_\beta in the paper
        # we also need **an expection over L samples**
        mean, std, mid_z, z = self.encode(state, action)

        mean_enc = mean.repeat(num_samples, 1, 1).permute(1, 0, 2)  # [B x S x D]
        std_enc = std.repeat(num_samples, 1, 1).permute(1, 0, 2)  # [B x S x D]
        z = mean_enc + std_enc * torch.randn_like(std_enc)  # [B x S x D]

        state = state.repeat(num_samples, 1, 1).permute(1, 0, 2)  # [B x S x C]
        action = action.repeat(num_samples, 1, 1).permute(1, 0, 2)  # [B x S x C]
        mean_dec = self.decode(state, z)
        std_dec = np.sqrt(beta / 4)

        # Find q(z|x)
        log_qzx = td.Normal(loc=mean_enc, scale=std_enc).log_prob(z)
        # Find p(z)
        mu_prior = torch.zeros_like(z).to(self.device)
        std_prior = torch.ones_like(z).to(self.device)
        log_pz = td.Normal(loc=mu_prior, scale=std_prior).log_prob(z)
        # Find p(x|z)
        std_dec = torch.ones_like(mean_dec).to(self.device) * std_dec
        log_pxz = td.Normal(loc=mean_dec, scale=std_dec).log_prob(action)

        w = log_pxz.sum(-1) + log_pz.sum(-1) - log_qzx.sum(-1)
        ll = w.logsumexp(dim=-1) - np.log(num_samples)
        return ll
    
    def encode(
            self,
            state: torch.Tensor,
            action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mid_z = self.encoder_shared(torch.cat([state, action], -1))
        id_=self.get_code_indices(mid_z)
        z=self.quantize(id_)
        mean = self.mean(z)
        # Clamped for numerical stability
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        return mean, std, mid_z, z

    def decode(
            self,
            state: torch.Tensor,
            z: torch.Tensor = None,
    ) -> torch.Tensor:
        # When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
        if z is None:
            z = (
                torch.randn((state.shape[0], self.latent_dim))
                    .to(self.device)
                    .clamp(-0.5, 0.5)
            )
        x = torch.cat([state, z], -1)
        return self.max_action * self.decoder(x)


class Actor(nn.Module):
    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            max_action: float,
            init_w: Optional[float] = None,
    ):
        super(Actor, self).__init__()

        head = nn.Linear(256, action_dim)
        if init_w is not None:
            weights_init(head, init_w)

        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            head,
            nn.Tanh(),
        )

        self.max_action = max_action

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.max_action * self.net(state)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        return self(state).cpu().data.numpy().flatten()


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, init_w: Optional[float] = None):
        super(Critic, self).__init__()

        head = nn.Linear(256, 1)
        if init_w is not None:
            weights_init(head, init_w)

        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            head,
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        sa = torch.cat([state, action], 1)
        return self.net(sa)
