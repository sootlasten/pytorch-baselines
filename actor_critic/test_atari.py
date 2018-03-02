import time
import argparse

import numpy as np
import gym
import torch

from torch.autograd import Variable
from torch.distributions import Categorical

from policies import CnnPolicy

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import preprocess_pong


def test(args):
    env = gym.make(args.env)
    obs = env.reset()

    policy = CnnPolicy(env.action_space.n)
    policy.load_state_dict(torch.load(args.ckpt_path)['state_dict'])
    if not args.no_cuda: policy.cuda()

    frames = Variable(torch.zeros((1, 4, 80, 80)))  # used to hold 4 consecutive frames
    if not args.no_cuda: frames = frames.cuda()
    prepro = preprocess_pong if 'Pong' in args.env else preprocess_atari

    while True:
        env.render()

        obs = prepro(obs)
        obs = np.expand_dims(np.expand_dims(obs, 0), 0)
        obs = Variable(torch.from_numpy(obs))
        if not args.no_cuda: obs = obs.cuda()
        
        # add current observation to structure that holds cosecutive frames
        frames = frames[:, :-1, :, :]
        frames = torch.cat((obs, frames), 1)

        action_probs, _ = policy(frames)
        action_dist = Categorical(action_probs)
        action = action_dist.sample()
        
        obs, reward, done, _ = env.step(action.data[0])
        if done: return
            
        time.sleep(0.01)  # so the game wouldn't move too fast


def parse():
    parser = argparse.ArgumentParser(description='test actor-critic on atari env')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA testing')
    parser.add_argument('--env', type=str, default='PongDeterministic-v0', metavar='envname',
                        help='the environment to train the model on (default: PongDeterministic-v0)')
    parser.add_argument('--ckpt-path', type=str, default=os.path.join(os.getcwd(), 'policy.ckpt'),
                        metavar='path', help='full path of the model checkpoint file')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse()
    test(args)

