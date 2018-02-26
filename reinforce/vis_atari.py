import time

import numpy as np

import gym
import torch

from torch.autograd import Variable
from torch.distributions import Categorical

from policies import CnnPolicy

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import preprocess_pong


def test():
    env = gym.make('Pong-v0')
    policy = CnnPolicy(env.action_space.n)

    policy.load_state_dict(torch.load('policy.ckpt'))
    obs = env.reset()
    done = False
    nb_frames = 0
    frames = Variable(torch.zeros((1, 4, 80, 80)))  # used to hold 4 consecutive frames
    while not done:
        env.render()
        frames = frames.data.cpu().numpy()
        obs = preprocess_pong(obs)
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
    test()

