from copy import deepcopy
import gym
from itertools import count
import math
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import os
from PIL import Image
import torch
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import wandb
import time
import random
import numpy as np

from utils import plot_single_frame, make_video, extract_mode_from_path
from ppo_agent import PPOAgent

class MultiAgent():
    """This is a meta agent that creates and controls several sub agents. If model_others is True,
    Enables sharing of buffer experience data between agents to allow them to learn models of the 
    other agents. """

    def __init__(self, config, env, device, training=True, with_expert=None, debug=False):
        config.batch_size = int(config.num_envs * config.num_steps)
        config.minibatch_size = int(config.batch_size // config.num_minibatches)
        config.num_iterations = config.total_timesteps // config.batch_size
        self.model_others = False
        run_name = f"{config.env_id}__{config.exp_name}__{config.seed}__{int(time.time())}"
        self.config = config
        
        wandb.init(
            project=config.wandb_project_name,
            entity=config.wandb_entity,
            sync_tensorboard=True,
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
        self.writer = SummaryWriter(f"runs/{run_name}")
        self.writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(config).items()])),
        )

        seed = config.seed
        # TRY NOT TO MODIFY: seeding
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = config.torch_deterministic

        self.device = torch.device("cuda" if torch.cuda.is_available() and config.cuda else "cpu")
        
        # env setup
        self.combined_next_obs = env.reset()

        self.agents = []
        self.n_agents = env.n_agents
        action_dim = env.action_space.high[0] + 1
        for agent_id in range(self.n_agents):
            next_obs = {"image": self.combined_next_obs["image"][agent_id],
                            "direction": self.combined_next_obs["direction"][agent_id]}
            self.agents.append(PPOAgent(config, next_obs, action_dim, agent_id, torch.device('cpu')))
            
        self.total_steps = 0
        self.debug = debug
        # TRY NOT TO MODIFY: start the game
        for agent in self.agents:
            agent.next_done = 0.0


    def run_one_episode(self, env, episode, log=True, train=True, save_model=True, visualize=False):
        done = False
        rewards = []
        t = 0

        # Annealing the rate if instructed to do so.
        if self.config.anneal_lr:
            frac = 1.0 - (episode - 1.0) / self.config.num_iterations
            lrnow = frac * self.config.lr
            for agent_id in range(self.n_agents):
                self.agents[agent_id].optimizer.param_groups[0]["lr"] = lrnow

        if visualize:
            viz_data = self.init_visualization_data(env, self.combined_next_obs)

        while not done:
            self.total_steps += 1
            t += 1
            for agent_id, agent in enumerate(self.agents):
                agent.obs[t] = torch.tensor(agent._flatten_obs({
                    "image": self.combined_next_obs["image"][agent_id],
                    "direction": self.combined_next_obs["direction"][agent_id]
                })).to(self.device)
                agent.dones[t] = agent.next_done

            # ALGO LOGIC: action logic
            actions = []
            all_next_obs = []
            if self.config.mappo:
                for agent in self.agents:
                    agent.obs[t] = agent.next_obs
                    agent.dones[t] = torch.tensor(agent.next_done).to(self.device)

                    with torch.no_grad():
                        logits = agent.network.actor(agent.next_obs.unsqueeze(0))
                        probs = Categorical(logits=logits)
                        action = probs.sample()
                        value = self.agents[0].network.critic(agent.next_obs)
                        agent.values[t] = value.squeeze(0)
                        all_next_obs.append(agent.next_obs.unsqueeze(0))
                    
                    agent.actions[t] = action.squeeze(0)
                    actions.append(action.squeeze(0))
                    agent.logprobs[t] = probs.log_prob(action).squeeze(0)
            else:
                for agent_id, agent in enumerate(self.agents):
                    # ALGO LOGIC: action logic
                    next_obs = torch.tensor(agent._flatten_obs({
                    "image": self.combined_next_obs["image"][agent_id],
                    "direction": self.combined_next_obs["direction"][agent_id]
                    }))
                    with torch.no_grad():
                        action, logprob, _, value = agent.get_action_and_value(next_obs)
                        agent.values[t] = value.flatten()
                        action = action.squeeze(0)
                    agent.actions[t] = action
                    agent.logprobs[t] = logprob.squeeze(0)
                    actions.append(action.item())

            # TRY NOT TO MODIFY: execute the game and log data.
            next_state, reward, done, info = env.step(actions)
            if done:
                next_state = env.reset()
                # Not enough data, continue epoch
                if t < self.config.batch_size:
                    done = False
            done = done or t >= self.config.batch_size - 1

            rewards.append(reward)
            self.combined_next_obs = next_state

            # For each agent, update its rollout buffers.
            for agent_id, agent in enumerate(self.agents):
                next_obs_idx = {"image": next_state["image"][agent_id],
                                "direction": next_state["direction"][agent_id]}
                agent.rewards[t] = torch.tensor(reward[agent_id]).to(self.device)
                agent.next_obs = torch.tensor(agent._flatten_obs(next_obs_idx)).to(self.device)
                agent.next_done = torch.tensor(1.0 if done else 0.0).to(self.device)

            if visualize:
                viz_data = self.add_visualization_data(viz_data, env, self.combined_next_obs, actions, next_state)

        for agent in self.agents:
            self.train_agent(agent)

        # Logging and checkpointing
        if log: self.log_one_episode(episode, t, rewards)
        self.print_terminal_output(episode, np.sum(rewards))
        self.save_model_checkpoints(episode)

        if visualize:
            viz_data['rewards'] = np.array(rewards)
            return viz_data
        
    def train_agent(self, agent):
            # bootstrap value if not done
            with torch.no_grad():
                agent.next_value = agent.network.get_value(agent.next_obs).reshape(1, -1)
                agent.advantages = torch.zeros_like(agent.rewards).to(self.device)
                agent.lastgaelam = 0
                for t in reversed(range(self.config.num_steps)):
                    if t == self.config.num_steps - 1:
                        agent.nextnonterminal = 1.0 - agent.next_done
                        agent.nextvalues = agent.next_value
                    else:
                        agent.nextnonterminal = 1.0 - agent.dones[t + 1]
                        agent.nextvalues = agent.values[t + 1]
                    agent.delta = agent.rewards[t] + self.config.gamma * agent.nextvalues * agent.nextnonterminal - agent.values[t]
                    agent.advantages[t] = agent.lastgaelam = agent.delta + self.config.gamma * self.config.gae_lambda * \
                                            agent.nextnonterminal * agent.lastgaelam
                agent.returns = agent.advantages + agent.values

            # flatten the batch
            agent.b_obs = agent.obs.reshape((-1,) + agent.obs_size)
            agent.b_logprobs = agent.logprobs.reshape(-1)
            agent.b_actions = agent.actions.reshape(-1)
            agent.b_advantages = agent.advantages.reshape(-1)
            agent.b_returns = agent.returns.reshape(-1)
            agent.b_values = agent.values.reshape(-1)

            # Optimizing the policy and value network
            agent.b_inds = np.arange(self.config.batch_size)
            agent.clipfracs = []
            for epoch in range(self.config.update_epochs):
                np.random.shuffle(agent.b_inds)
                for start in range(0, self.config.batch_size, self.config.minibatch_size):
                    end = start + self.config.minibatch_size
                    mb_inds = agent.b_inds[start:end]

                    _, agent.newlogprob, agent.entropy, agent.newvalue = agent.get_action_and_value(agent.b_obs[mb_inds], agent.b_actions.long()[mb_inds])
                    logratio = agent.newlogprob - agent.b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        agent.clipfracs += [((ratio - 1.0).abs() > self.config.clip_coef).float().mean().item()]

                    agent.mb_advantages = agent.b_advantages[mb_inds]
                    if self.config.norm_adv:
                        agent.mb_advantages = (agent.mb_advantages - agent.mb_advantages.mean()) / (agent.mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -agent.mb_advantages * ratio
                    pg_loss2 = -agent.mb_advantages * torch.clamp(ratio, 1 - self.config.clip_coef, 1 + self.config.clip_coef)
                    agent.pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    agent.newvalue = agent.newvalue.view(-1)
                    if self.config.clip_vloss:
                        v_loss_unclipped = (agent.newvalue - agent.b_returns[mb_inds]) ** 2
                        v_clipped = agent.b_values[mb_inds] + torch.clamp(
                            agent.newvalue - agent.b_values[mb_inds],
                            -self.config.clip_coef,
                            self.config.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - agent.b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        agent.v_loss = 0.5 * v_loss_max.mean()
                    else:
                        agent.v_loss = 0.5 * ((agent.newvalue - agent.b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = agent.entropy.mean()
                    agent.loss = agent.pg_loss - self.config.ent_coef * entropy_loss + agent.v_loss * self.config.vf_coef

                    agent.optimizer.zero_grad()
                    agent.loss.backward()
                    nn.utils.clip_grad_norm_(agent.network.parameters(), self.config.max_grad_norm)
                    agent.optimizer.step()

            y_pred, y_true = agent.b_values.cpu().numpy(), agent.b_returns.cpu().numpy()
            var_y = np.var(y_true)
            agent.explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y


    def log_one_episode(self, episode, t, rewards):
        # Log total reward and per-agent rewards.
        rewards = np.sum(np.array(rewards), axis=0)
        self.writer.add_scalar("charts/total_reward", \
                np.sum(rewards), self.total_steps)
        for idx, agent_reward in enumerate(rewards):
            self.writer.add_scalar(f"charts/agent_{idx}_return", \
                                    agent_reward, self.total_steps)

    def get_action_predictions(self, step):
        actions = []
        for idx in range(self.n_agents):
            actions.append(self.agents[idx].get_action(step))
        return actions

    def save_model_checkpoints(self, episode):
        if episode % self.config.save_model_episode == 0:
            for i in range(self.n_agents):
                self.agents[i].save()

    def print_terminal_output(self, episode, total_reward):
        if episode % self.config.print_every == 0:
            print('Total steps: {} \t Episode: {} \t Total reward: {}'.format(
                self.total_steps, episode, total_reward))

    def init_visualization_data(self, env, state):
        viz_data = {
            'agents_partial_images': [],
            'actions': [],
            'full_images': [],
            'predicted_actions': None
            }
        viz_data['full_images'].append(env.render('rgb_array'))

        if self.model_others:
            predicted_actions = []
            predicted_actions.append(self.get_action_predictions(state))
            viz_data['predicted_actions'] = predicted_actions

        return viz_data

    def get_agent_state(self, state, idx):
        return {"image": state["image"][idx], "direction": state["direction"][idx]}

    def add_visualization_data(self, viz_data, env, state, actions, next_state):
        viz_data['actions'].append(actions)
        viz_data['agents_partial_images'].append(
            [env.get_obs_render(
                self.get_agent_state(state, i)['image']) for i in range(self.n_agents)])
        viz_data['full_images'].append(env.render('rgb_array'))
        if self.model_others:
            viz_data['predicted_actions'].append(self.get_action_predictions(next_state))
        return viz_data
        
    def update_models(self):
        # Don't update model until you've taken enough steps in env
        if self.total_steps > self.config.initial_memory: 
            if self.total_steps % self.config.update_every == 0: # How often to update model
                for agent in self.agents:
                    self.train_agent(agent)

    def train(self, env):
        for episode in range(self.config.n_episodes):
            if episode % self.config.visualize_every == 0 and not (self.debug and episode == 0):
                viz_data = self.run_one_episode(env, episode, visualize=True)
                self.visualize(env, self.config.mode + '_training_step' + str(episode), 
                               viz_data=viz_data)
            else:
                self.run_one_episode(env, episode)

        env.close()
        return

    def visualize(self, env, mode, video_dir='videos', viz_data=None):
        if not viz_data:
            viz_data = self.run_one_episode(
                env, episode=0, log=False, train=False, save_model=False, visualize=True)
            env.close()

        video_path = os.path.join(*[video_dir, self.config.experiment_name, self.config.model_name])

        # Set up directory.
        if not os.path.exists(video_path):
            os.makedirs(video_path)

        # Get names of actions
        action_dict = {}
        for act in env.Actions:
            action_dict[act.value] = act.name

        traj_len = len(viz_data['rewards'])
        for t in range(traj_len):
            self.visualize_one_frame(t, viz_data, action_dict, video_path, self.config.model_name)
            print('Frame {}/{}'.format(t, traj_len))

        make_video(video_path, mode + '_trajectory_video')

    def visualize_one_frame(self, t, viz_data, action_dict, video_path, model_name):
        plot_single_frame(t, 
                          viz_data['full_images'][t], 
                          viz_data['agents_partial_images'][t], 
                          viz_data['actions'][t], 
                          viz_data['rewards'], 
                          action_dict, 
                          video_path, 
                          self.config.model_name)

    def load_models(self, model_path=None):
        for i in range(self.n_agents):
            if model_path is not None:
                self.agents[i].load(save_path=model_path + '_agent_' + str(i))
            else:
                # Use agents' default model path
                self.agents[i].load()


