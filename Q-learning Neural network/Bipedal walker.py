# Install required modules
!pip install gymnasium
!pip install "gymnasium[atari, accept-rom-license]"
!apt-get install -y swig
!pip install gymnasium[box2d]

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import gymnasium as gym

class Actor(nn.Module):
    def __init__(self, state_size, action_size, seed=42):
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 128)  # Increased size
        self.fc2 = nn.Linear(128, 128)  # Increased size
        self.fc3 = nn.Linear(128, action_size)
        self.action_size = action_size

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))  # Action values are scaled between -1 and 1

class Critic(nn.Module):
    def __init__(self, state_size, action_size, seed=42):
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size + action_size, 128)  # Increased size
        self.fc2 = nn.Linear(128, 128)  # Increased size
        self.fc3 = nn.Linear(128, 1)

    def forward(self, state, action):
        x = F.relu(self.fc1(torch.cat([state, action], dim=1)))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ReplayMemory:
    def __init__(self, capacity):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def push(self, event):
        self.memory.append(event)

    def sample(self, batch_size):
        experiences = random.sample(self.memory, k=batch_size)
        states = torch.from_numpy(np.vstack([e[0] for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e[3] for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
        return states, actions, rewards, next_states, dones

class Agent:
    def __init__(self, state_size, action_size):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.state_size = state_size
        self.action_size = action_size

        self.actor_local = Actor(state_size, action_size).to(self.device)
        self.actor_target = Actor(state_size, action_size).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=1e-4)  # Adjusted learning rate

        self.critic_local = Critic(state_size, action_size).to(self.device)
        self.critic_target = Critic(state_size, action_size).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=1e-4)  # Adjusted learning rate

        self.memory = ReplayMemory(int(1e5))
        self.t_step = 0
        self.interpolation_parameter = 1e-2  # Adjusted interpolation parameter

    def step(self, state, action, reward, next_state, done):
        self.memory.push((state, action, reward, next_state, done))
        self.t_step = (self.t_step + 1) % 4
        if self.t_step == 0:
            if len(self.memory.memory) > minibatch_size:
                experiences = self.memory.sample(minibatch_size)
                self.learn(experiences, discount_factor)

    def act(self, state, noise=0.):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy().squeeze()
        self.actor_local.train()
        action = np.clip(action + noise, env.action_space.low, env.action_space.high)
        return action

    def learn(self, experiences, discount_factor):
        states, actions, rewards, next_states, dones = experiences

        # Update Critic
        next_actions = self.actor_target(next_states)
        next_q_targets = self.critic_target(next_states, next_actions).detach()
        q_targets = rewards + discount_factor * next_q_targets * (1 - dones)
        q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(q_expected, q_targets)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1.0)
        self.critic_optimizer.step()

        # Update Actor
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.actor_local, self.actor_target, self.interpolation_parameter)
        self.soft_update(self.critic_local, self.critic_target, self.interpolation_parameter)

    def soft_update(self, local_model, target_model, interpolation_parameter):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(
                interpolation_parameter * local_param.data + (1.0 - interpolation_parameter) * target_param.data
            )

# Hyperparameters
learning_rate_actor = 1e-4
learning_rate_critic = 1e-4  # Adjusted learning rate
minibatch_size = 128  # Increased batch size
discount_factor = 0.99  # Slightly increased
interpolation_parameter = 1e-2  # Adjusted interpolation parameter

# Initialize environment and agent
env = gym.make('BipedalWalker-v3')
state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]  # For continuous action space
agent = Agent(state_size, action_size)

# Training loop
number_episodes = 2000
maximum_number_timesteps_per_episode = 1000
epsilon_starting_value = 1.0
epsilon_ending_value = 0.01
epsilon_decay_value = 0.995
epsilon = epsilon_starting_value
scores_on_100_episodes = deque(maxlen=100)

for episode in range(1, number_episodes + 1):
    state, _ = env.reset()
    score = 0
    for t in range(maximum_number_timesteps_per_episode):
        action = agent.act(state, epsilon)
        next_state, reward, done, _, _ = env.step(action)
        agent.step(state, action, reward, next_state, done)
        state = next_state
        score += reward
        if done:
            break
    scores_on_100_episodes.append(score)
    epsilon = max(epsilon_ending_value, epsilon_decay_value * epsilon)
    if episode % 100 == 0:
        print(f'\rEpisode {episode}\tAverage Score: {np.mean(scores_on_100_episodes):.2f}')
    if np.mean(scores_on_100_episodes) >= 200.0:
        print(f'\nEnvironment solved in {episode - 100} episodes!\tAverage Score: {np.mean(scores_on_100_episodes):.2f}')
        torch.save(agent.actor_local.state_dict(), 'checkpoint.pth')
        break

# Record and display video
import glob
import io
import base64
import imageio
from IPython.display import HTML, display
from gym.wrappers.monitoring.video_recorder import VideoRecorder

def show_video_of_model(agent, env_name):
    env = gym.make(env_name, render_mode='rgb_array')
    state, _ = env.reset()
    done = False
    frames = []
    while not done:
        frame = env.render()
        frames.append(frame)
        action = agent.act(state)
        state, reward, done, _, _ = env.step(action)
    env.close()
    imageio.mimsave('video.mp4', frames, fps=30)

show_video_of_model(agent, 'BipedalWalker-v3')

def show_video():
    mp4list = glob.glob('*.mp4')
    if len(mp4list) > 0:
        mp4 = mp4list[0]
        video = io.open(mp4, 'r+b').read()
        encoded = base64.b64encode(video)
        display(HTML(data='''<video alt="test" autoplay
                loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii'))))
    else:
        print("Could not find video")

show_video()
