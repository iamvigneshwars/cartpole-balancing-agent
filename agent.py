import gym 
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time


# DQN Model
class DQN(nn.Module):
    def __init__(self, num_state_features, num_actions):
        super().__init__()
         
        self.fc1 = nn.Linear(in_features=num_state_features, out_features=100)   
        self.fc2 = nn.Linear(in_features=100, out_features=50)
        self.out = nn.Linear(in_features=50, out_features=num_actions)            

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x

# Replay Memory
class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
    
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# Epsilon Strategy
class EpsilonGreedyStrategy():
    def __init__(self, start, end , decay):
        self.start = start
        self.end = end
        self.decay = decay

    def get_exploration_rate(self, current_step):
        return self.end + (self.start - self.end) * math.exp(-1 * current_step * self.decay)

# Agent
class Agent():
    def __init__(self, strategy, num_actions, device):
        self.current_step = 0
        self.strategy = strategy
        self.num_actions = num_actions
        self.device = device

    def select_action(self, state, policy_net):
        rate = strategy.get_exploration_rate(self.current_step)
        self.current_step += 1

        # Random action (Explore)
        if rate > random.random():
            action = random.randrange(self.num_actions)
            return torch.tensor([action]).to(self.device)
        else:
            # Predict the best action 
            with torch.no_grad():
                return policy_net(state).unsqueeze(dim=0).argmax(dim=1).to(self.device)

class GameManager():
    def __init__(self, device, environment):
        self.device = device
        self.env = gym.make(environment).unwrapped
        self.env.reset()
        self.done = False
        self.current_state = None
        
    def reset(self):
        self.current_state = self.env.reset()
        
    def close(self):
        self.env.close()
    
    def render(self, mode = 'human'):
        return self.env.render(mode)
    
    def num_states(self):
        return self.env.observation_space.shape[0]
    
    def num_actions(self):
        return self.env.action_space.n
    
    def take_action(self, action):        
        self.current_state, reward, self.done, _ = self.env.step(action.item())
        return torch.tensor([reward], device=self.device).long()
    
    def get_state(self):
        if self.done:
            return torch.zeros_like(torch.tensor(self.current_state), device = device).float()
        
        else:
            return torch.tensor(self.current_state, device = device).float()


def get_moving_average(period, values):
    values = torch.tensor(values, dtype=torch.float)
    if len(values) >= period:
        moving_avg = values.unfold(dimension=0, size=period, step=1) \
            .mean(dim=1).flatten(start_dim=0)
        moving_avg = torch.cat((torch.zeros(period-1), moving_avg))
        return moving_avg.numpy()
    else:
        moving_avg = torch.zeros(len(values))
        return moving_avg.numpy()
    
def plot(values, moving_avg_period):
    plt.figure(2)
    plt.clf()        
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(values)
    moving_avg=get_moving_average(moving_avg_period, values)
    
    plt.plot(get_moving_average(moving_avg_period, values))
    plt.pause(0.001)
    print("Episode", len(values), "\n", moving_avg_period, "episode moving avg:", moving_avg[-1])
 
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'mask'))


batch_size = 1024
gamma = 0.8
eps_start = 1
eps_end = 0.01
eps_decay = 0.001
target_update = 10
memory_size = 10000
lr = 0.001
num_episodes = 250

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gm= GameManager(device, 'CartPole-v1')
strategy = EpsilonGreedyStrategy(eps_start ,eps_end, eps_decay)
agent = Agent(strategy, gm.num_actions(), device)
memory = ReplayMemory(memory_size)

policy_net = DQN(gm.num_states(), gm.num_actions()).to(device)
target_net = DQN(gm.num_states(), gm.num_actions()).to(device)
target_net.eval()
optimizer = optim.Adam(params = policy_net.parameters(), lr=lr)

episode_durations = []

for episode in range(num_episodes):
    gm.reset()
    state = gm.get_state()

    for timestep in count():
        gm.render()
        action = agent.select_action(state, policy_net)
        reward = gm.take_action(action)
        next_state = gm.get_state()
        mask = torch.tensor([0 if gm.done else 1], device = device)
        # mask = 0 if gm.done else 1
        memory.push(state, action, next_state, reward, mask)
        state = next_state

        if len(memory) >= batch_size:
            transitions = memory.sample(batch_size)
            batch = Transition(*zip(*transitions))

            states = torch.stack(batch.state)
            actions = torch.cat(batch.action)
            rewards = torch.cat(batch.reward)
            next_states = torch.stack(batch.next_state)
            masks = torch.cat(batch.mask)

            # Calculate current qvalue
            current_qvalues = policy_net(states).gather(dim = 1, index = actions.unsqueeze(-1))
            next_qvalues = target_net(next_states).max(dim = 1)[0].detach()
            target_qvalues = (next_qvalues * mask * gamma) + rewards

            loss = F.mse_loss(current_qvalues, target_qvalues)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if gm.done:
            episode_durations.append(timestep)
            plot(episode_durations, 100)
            break

    if episode % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())

    if get_moving_average(100, episode_durations)[-1] >= 195:
        break

gm.close()
    
    
