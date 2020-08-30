import gym
import slimevolleygym
import time

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions.categorical import Categorical

def mlp(sizes, activation=nn.Tanh):
    # Build a feedforward neural network.
    layers = []
    for j in range(len(sizes) - 1):
        layers += [nn.Linear(sizes[j], sizes[j+1]), activation()]
    return nn.Sequential(*layers)


env = gym.make('SlimeVolley-v0')
#env = gym.make('CartPole-v0')

obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.n

policy_net = mlp(sizes=[obs_dim, 8, 8, 8, act_dim])
optimizer = Adam(policy_net.parameters(), lr=1e-2)

batch_size = 5000

def reward_to_go(rewards):
    n = len(rewards)
    acc = [0] * n
    for i in reversed(range(n)):
        acc[i] = rewards[i] + (acc[i+1] if i+1 < n else 0)
    return acc

def reward_ever(rewards):
    Return = sum(rewards)
    return [Return] * len(rewards)

def train_one_epoch(render=True):
    observation = env.reset()

    episode_obs = []
    episode_act = []
    episode_rew = []
    episode_ret = []

    episode_win = []
    while True:
        if render:
            env.render()
            time.sleep(0.01)

        episode_obs.append(observation.copy())
        observation = torch.as_tensor(observation, dtype=torch.float32)

        # policy execution
        logits = policy_net(observation)
        act = Categorical(logits=logits).sample().item()

        #multi_binary_act_input = act
        multi_binary_act_input = [0] * act_dim
        multi_binary_act_input[act] = 1;

        # take action
        observation, reward, done, _ = env.step(multi_binary_act_input)

        episode_act.append(act)
        episode_rew.append(reward)

        if done:
            #episode_ret += reward_ever(episode_rew)
            episode_ret += reward_to_go(episode_rew)

            episode_win.append(sum(episode_rew))
            episode_rew = []

            observation = env.reset()
            if len(episode_obs) > batch_size:
                    break

    train_obs = torch.as_tensor(episode_obs, dtype=torch.float32)
    train_act = torch.as_tensor(episode_act, dtype=torch.int32)
    train_ret = torch.as_tensor(episode_ret, dtype=torch.float32)

    optimizer.zero_grad()

    logits = policy_net(train_obs)
    logp = Categorical(logits=logits).log_prob(train_act)
    loss = -(logp * train_ret).mean()

    loss.backward()
    optimizer.step()

    return loss.item(), train_ret.mean(), sum(episode_win) / len(episode_win)

render = False
for epoch in range(5000):
    Loss, Return, Wins = train_one_epoch(render=render)

    #render = True if epoch > 80 else False
    render = True if Wins >= -2.5 else False

    print(f'epoch: {epoch} \t loss: {Loss:.2f} \t return: {Return:.2f} \t Wins: {Wins:.2f}')

print('Closing...')
env.close()
