
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from itertools import count

device = (
    torch.device(0) if torch.cuda.is_available()
    else torch.device("cpu")
)

env = gym.make('LunarLander-v2')
num_cells = 256
actor = nn.Sequential(
    nn.LazyLinear(num_cells, device=device),
    nn.ReLU(),
    nn.LazyLinear(num_cells, device=device),
    nn.ReLU(),
    nn.LazyLinear(env.action_space.n, device=device),
)

critic = nn.Sequential(
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(1, device=device),
)

actor_optim = optim.Adam(actor.parameters(), lr=1e-4)
critic_optim = optim.Adam(critic.parameters(), lr=1e-4)

num_episodes = 1000
gamma = 0.99

for episode in range(num_episodes):

    state, info = env.reset()
    epside_reward = 0
    state = torch.tensor(state, device=device)
    for t in count():
        action_values = actor(state)
        action_probs = nn.Softmax(dim=0)(action_values)
        action = torch.multinomial(action_probs, num_samples=1).item()
        next_state, reward, done, truncated, _= env.step(action)
        next_state = torch.tensor(next_state, device=device)

        state_value = critic(state)
        next_state_value = critic(next_state)
        advantage = reward + gamma * next_state_value - state_value

        actor_advantage = advantage.detach()
        actor_loss = -torch.log(action_probs[action]) * actor_advantage
        critic_loss = torch.square(advantage)

        epside_reward += reward

        actor_optim.zero_grad()
        critic_optim.zero_grad()

        actor_loss.backward()
        critic_loss.backward()

        actor_optim.step()
        critic_optim.step()

        state = next_state

        if done or truncated:
            break

    if episode % 10 == 0:
        print(f"Episode {episode}, Reward: {epside_reward}")
env.close()