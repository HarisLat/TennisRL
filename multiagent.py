import random
from replay_buffer import ReplayBuffer
import numpy as np
import torch

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128  # minibatch size

from ddpg_agent import Agent


class MultiAgent:
    def __init__(self, state_size, n_agents, action_size, random_seed):

        self.state_size = state_size
        self.n_agents = n_agents
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
        self.agents = [
            Agent(state_size, n_agents, action_size, random_seed)
            for _ in range(n_agents)
        ]

    def reset(self):
        [agent.reset() for agent in self.agents]

    def act(self, states):
        actions = [
            self.agents[agent].act(np.array([states[agent]]))
            for agent in range(self.n_agents)
        ]
        return actions

    def step(self, states, actions, rewards, next_states, dones):
        """Returns actions for given state as per current policy."""

        [
            self.agents[agent].step(
                states[agent],
                actions[agent],
                rewards[agent],
                next_states[agent],
                dones[agent],
            )
            for agent in range(self.n_agents)
        ]

    def save(self):
        for agent in range(self.n_agents):
            torch.save(
                self.agents[agent].actor_local.state_dict(),
                "checkpoint_actor_{}.pth".format(agent),
            )
            torch.save(
                self.agents[agent].critic_local.state_dict(),
                "checkpoint_critic_{}.pth".format(agent),
            )
