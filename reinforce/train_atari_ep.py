import time
import argparse
import numpy as np

# find parent directory and import utilities from there
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import preprocess_pong

import gym
import torch

from torch import optim
from torch.autograd import Variable
from torch.distributions import Categorical

from policies import CnnPolicy


def _discount_rewards(rewards):
    disc_rewards = np.zeros(len(rewards))
    running_add = 0
    for i in np.arange(len(rewards)-1, -1, -1):
        running_add = rewards[i] + args.gamma*running_add
        disc_rewards[i] = running_add
    return disc_rewards


def train(args):
    """Trains an agent in a ATARI episodic environment. Rewards are accumulated 
       thorughout the entire episode, and updates are made after the end of 
       the episode."""
    env = gym.make(args.env)
    obs = env.reset()

    policy = CnnPolicy(env.action_space.n)
    if not args.no_cuda: policy.cuda()
    optimizer = optim.RMSprop(policy.parameters(), lr=args.eta)
    
    frames = Variable(torch.zeros((1, 4, 80, 80)))
    if not args.no_cuda: frames = frames.cuda()

    rewards, logprobs, aprobs = [], [], []
    reward_sum = 0
    epi = 0; ep_start = time.time()

    for ts in range(args.nb_timesteps):
        obs = preprocess_pong(obs)  # TODO! Make this more general
        obs = np.expand_dims(np.expand_dims(obs, 0), 0)
        obs = Variable(torch.from_numpy(obs))
        if not args.no_cuda: obs = obs.cuda()
        
        # add current observation to structure that holds cosecutive frames
        frames = frames[:, :-1, :, :]
        frames = torch.cat((obs, frames), 1)

        action_probs = policy(frames)
        action_dist = Categorical(action_probs)
        action = action_dist.sample()
    
        obs, reward, done, _ = env.step(action.data[0])
        reward_sum += reward
    
        rewards.append(reward)
        logprobs.append(action_dist.log_prob(action))
        aprobs.append(action_probs)
        
        if done:
            disc_rewards = _discount_rewards(rewards)
            norm_rewards = (disc_rewards - np.mean(disc_rewards)) / np.std(disc_rewards)
            
            aprobs = torch.cat(aprobs).clamp(1e-8)
            entropies = -torch.sum(aprobs*torch.log(aprobs), dim=1)
            loss = (logprobs*-norm_rewards).sum() + args.beta*entropies.sum()
        
            # param update 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # zero-out buffers and restart game
            frames = Variable(torch.zeros((1, 4, 80, 80)))
            if not args.no_cuda: frames = frames.cuda()
            rewards, logprobs, aprobs = [], [], []
            obs = env.reset()
            
            total_time = time.time() - ep_start
            print("Episode {} took {:.2f} s ({}). Reward: {:.2f}".format(epi, total_time, ts, reward_sum))

            epi += 1; reward_sum = 0; ep_start = time.time()
    
        if not epi % args.save_freq:
            torch.save(policy.state_dict(), args.ckpt_path)


def parse():
    parser = argparse.ArgumentParser(description='train reinforce on an episodic atari env')
    parser.add_argument('--eta', type=float, default=2.5e-4, metavar='L',
                        help='learning rate for RMSProp (default: 2.5e-4)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor (default: 0.99)')
    parser.add_argument('--beta', type=float, default=0.01, metavar='G',
                        help='controls the strength of the entropy regularization' +
                        'term (default: 0.01)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=543, metavar='N',
                        help='random seed (default: 543)')
    parser.add_argument('--env', type=str, default='Pong-v0', metavar='envname',
                        help='the environment to train the model on (default: Pong-v0)')
    parser.add_argument('--nb-timesteps', type=int, default=int(2e8), metavar='ts',
                        help='number of timesteps the agent is trained')
    parser.add_argument('--ckpt-path', type=str, default=os.path.join(os.getcwd(), 'policy.ckpt'),
                        metavar='path', help='full path of the model checkpoint file')
    parser.add_argument('--save-freq', type=int, default=200, metavar='F',
                        help='save policy params after every that many episodes')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse()
    
    torch.manual_seed(args.seed)
    if not args.no_cuda:
        torch.cuda.manual_seed(args.seed)

    train(args)

