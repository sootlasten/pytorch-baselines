import time
import gym
import numpy as np

from gym import wrappers

from skimage.color import rgb2gray
from skimage.transform import resize

import torch
import torch.nn.functional as F 
from torch import nn, optim
from torch.autograd import Variable
from torch.distributions import Categorical



def preprocess(frame):
    frame = frame[35:195, :]  # remove uninformative parts of the frame
    frame = rgb2gray(frame) * 255
    frame = resize(frame, (84, 84))
    return frame


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.fc1 = nn.Linear(in_features=64*7*7, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=6)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.view(-1, 64*7*7)
        x = self.relu(self.fc1(x))
        scores = self.fc2(x)
        return F.softmax(scores, dim=1)


def train():
    gamma = 0.99

    running_reward = None
    reward_sum = 0

    env = gym.make('Pong-v0')
    obs = env.reset()

    policy = Policy()
    policy.cuda()
    optimizer = optim.RMSprop(policy.parameters(), lr=2.5e-4)
    
    rewards = []
    logprobs = []
    allprobs = []
    
    nb_episode = 0
    frames = Variable(torch.zeros((1, 4, 84, 84)))  # used to hold 4 consecutive frames
    start_time = time.time()
    while True:  # main training loop
        frames = frames.data.cpu().numpy()
        obs = preprocess(obs)
        frames = np.roll(frames, 1, axis=0)
        frames[0, 0] = obs
        frames = torch.from_numpy(frames)
        frames = Variable(frames)
        frames = frames.cuda()
        
        action_probs = policy(frames)
        torch.cuda.synchronize
        action_dist = Categorical(action_probs)
        action = action_dist.sample()
    
        obs, reward, done, _ = env.step(action.data[0])
        reward_sum += reward
    
        rewards.append(reward)
        logprobs.append(action_dist.log_prob(action))
        allprobs.append(action_probs)
        
        if done:
            end_time = time.time()
            print("Game finished in {:.2f} s".format(end_time - start_time))

            # discount rewards
            nb_rewards = len(rewards)
            discounted_rewards = np.zeros(nb_rewards)
            running_add = 0
            for i in np.arange(nb_rewards-1, 0, -1):
                running_add = rewards[i] + gamma*running_add
                discounted_rewards[i] = running_add
        
            # standardize rewards
            norm_rewards = (discounted_rewards - np.mean(discounted_rewards)) / np.std(discounted_rewards)
            
            loss = []
            for logp, r in zip(logprobs, norm_rewards):
                loss.append(-logp*r)
            allprobs = torch.cat(allprobs)
            entropy = -torch.sum(allprobs*torch.log(allprobs), dim=1)
            loss = torch.cat(loss).sum() # + 0.01*entropy.sum()
        
            # param update 
            backstart = time.time()
            optimizer.zero_grad()
            loss.backward()
            torch.cuda.synchronize
            backend = time.time()
            print("Backward pass took: {:.2f} s".format(backend - backstart))
            optimizer.step()
            
            # 0-out buffers
            rewards = []
            logprobs = []
            allprobs = []

            # log
            # running_reward = reward if not running_reward else 0.01*reward_sum + 0.99*running_reward
            # print(nb_episode, reward_sum)
            nb_episode += 1
            reward_sum = 0

            # start new episode
            obs = env.reset()
            start_time = time.time()

        
        # if not nb_episode % 10:
        #     torch.save(policy.state_dict(), '/home/stensootla/projects/pytorch-baselines/reinforce/pong.pt')


def test():
    policy = Policy()
    policy.load_state_dict(torch.load('pong.pt'))
    
    # gym.envs.register(
    #     id='MountainCarLong-v0',
    #     entry_point='gym.envs.classic_control:MountainCarEnv',
    #     max_episode_steps=200,
    # )

    env = gym.make('Pong-v0')
    # env = wrappers.Monitor(env, '/tmp/pong')

    obs = env.reset()
    done = False
    nb_frames = 0
    frames = Variable(torch.zeros((1, 4, 84, 84)))  # used to hold 4 consecutive frames
    while not done:
        env.render()
        # if not nb_frames % 10:
        #     print(nb_frames)

        frames = frames.data.cpu().numpy()
        obs = preprocess(obs)
        frames = np.roll(frames, 1, axis=0)
        frames[0, 0] = obs
        frames = torch.from_numpy(frames)
        frames = Variable(frames)

        time.sleep(0.008)
        # env.render()

        action_probs = policy(frames)
        action_dist = Categorical(action_probs)
        action = action_dist.sample()
        
        obs, reward, done, _ = env.step(action.data[0])
        
        nb_frames += 1
        

if __name__ == '__main__':
    train()
    #test()

