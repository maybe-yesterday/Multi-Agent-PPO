from typing import Dict, Any, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class ActorCritic(nn.Module):
    """A shared-feature Actor-Critic network."""
    def __init__(self, obs_size: int, action_dim: int):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_size, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_size, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, action_dim), std=0.01),
        )

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(x)

# --- PPO Agent Logic ---

class PPOAgent:
    """
    A PPO sub-agent that manages its own network, buffers, and training process.
    """
    def __init__(self, config: Any, initial_obs: Dict, action_dim: int, agent_id: int, device: torch.device):
        self.config = config
        self.agent_id = agent_id
        self.device = device
        self.action_dim = action_dim

        flat_obs = self._flatten_obs(initial_obs)
        # ensure obs_size is a tuple (N,)
        image_shape = initial_obs["image"].shape
        flattened_image_size = np.prod(image_shape)
        self.obs_size = (flattened_image_size + 1,)

        self.network = ActorCritic(int(np.prod(self.obs_size)), action_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=config.lr, eps=1e-5)

        # ALGO Logic: Storage setup
        self.obs = torch.zeros((config.num_steps, ) + self.obs_size).to(device)
        self.actions = torch.zeros(config.num_steps).to(device)
        self.logprobs = torch.zeros(config.num_steps).to(device)
        self.rewards = torch.zeros(config.num_steps).to(device)
        self.dones = torch.zeros(config.num_steps).to(device)
        self.values = torch.zeros(config.num_steps).to(device)

        self.next_obs = torch.Tensor(flat_obs).to(device)
        self.next_done = torch.tensor(0.0).to(device)

    def _flatten_obs(self, obs: Dict) -> np.ndarray:
        flat_image = np.array(obs["image"]).flatten()
        flat_direction = np.array(obs["direction"]).flatten()
        return np.concatenate([flat_image, flat_direction]).astype(np.float32)

    def get_action(self, step: int) -> int:
        self.obs[step] = self.next_obs
        self.dones[step] = self.next_done

        with torch.no_grad():
            action, logprob, _, value = self.get_action_and_value(self.next_obs.unsqueeze(0))

        self.values[step] = value.squeeze(0)
        self.actions[step] = action.squeeze(0)
        self.logprobs[step] = logprob.squeeze(0)

        return action.item()
    
    def get_action_and_value(
        self, x: torch.Tensor, action: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.network.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.network.critic(x)


    def save(self, model_path: str = 'models/'):
        save_path=model_path + '_agent_' + str(self.agent_id)
        checkpoint = {
            "network_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        torch.save(checkpoint, save_path)
        print(f"Model for agent {self.agent_id} saved to {save_path}")

    def load(self, model_path: str = 'models/'):
        save_path=model_path + '_agent_' + str(self.agent_id)
        checkpoint = torch.load(save_path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"Model loaded from {save_path}")
