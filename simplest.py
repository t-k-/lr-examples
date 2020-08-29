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

policy_net = mlp(sizes=[obs_dim, 32, act_dim])
optimizer = Adam(policy_net.parameters(), lr=1e-2)

batch_size = 5000

def train_one_epoch(render=True):
    observation = env.reset()

    episode_obs = []
    episode_act = []
    episode_rew = []
    episode_ret = []
    episode_len = []
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
            Return = sum(episode_rew)
            Length = len(episode_rew)
            episode_ret += [Return] * Length
            episode_len.append(Length)

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

    return loss.item(), sum(episode_len) / len(episode_len), train_ret.mean()

render = False
for epoch in range(5000):
    loss, frames, Return = train_one_epoch(render=render)

    #render = True if epoch > 40 else False
    render = True if Return >= -4.0 else False

    print(f'epoch: {epoch} \t loss: {loss:.2f} \t return: {Return:.2f} \t frames: {frames:.2f}')

print('Closing...')
env.close()
