import time
import argparse
import numpy as np
import pickle

import gym
import torch
from torch import optim
from torch.autograd import Variable
from torch.distributions import Categorical

from policies import CnnPolicy

# find parent directory and import utilities from there
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import preprocess_pong, preprocess_atari


def _discount_rewards(rewards):
    disc_rewards = np.zeros(len(rewards))
    running_add = 0
    for i in np.arange(len(rewards)-1, -1, -1):
        running_add = rewards[i] + args.gamma*running_add
        disc_rewards[i] = running_add
    return disc_rewards


def train(args):
    env = gym.make(args.env)
    obs = env.reset()

    policy = CnnPolicy(env.action_space.n)
    if not args.no_cuda: policy.cuda()
    optimizer = optim.Adam(policy.parameters(), lr=args.eta)
    
    frames = Variable(torch.zeros((1, 4, 80, 80)))
    if not args.no_cuda: frames = frames.cuda()
    prepro = preprocess_pong if 'Pong' in args.env else preprocess_atari

    rewards, logprobs, aprobs = [], [], []
    reward_sum = 0
    epi = 0; ep_start = time.time()

    running_reward = args.init_runreward
    running_rewards = []
    saved_ckpt_epi = epi
    saved_reward_epi = epi
    start_ts = 1

    if args.resume_ckpt:
        checkpoint = torch.load(args.resume_ckpt)
        policy.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_ts = checkpoint['step']
        epi = checkpoint['episode']
        running_reward = checkpoint['running_reward']
        
    for ts in range(start_ts, args.nb_steps+1):
        obs = prepro(obs)
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
    
        if not ts % args.update_freq:
            disc_rewards = _discount_rewards(rewards)
            if np.any(disc_rewards):
                disc_rewards = (disc_rewards - np.mean(disc_rewards)) / np.std(disc_rewards)
            
            aprobs = torch.cat(aprobs).clamp(1e-8)
            entropies = -torch.sum(aprobs*torch.log(aprobs), dim=1)
        
            t1 = torch.cat(logprobs)*-Variable(torch.Tensor(disc_rewards)).cuda()
            loss = t1.sum() + args.beta*entropies.sum()
        
            # param update 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # zero-out buffers
            frames = Variable(torch.zeros((1, 4, 80, 80)))
            if not args.no_cuda: frames = frames.cuda()
            rewards, logprobs, aprobs = [], [], []
            
        if done:
            total_time = time.time() - ep_start
            running_reward = reward_sum if not running_reward else running_reward*0.99 + reward_sum*0.01
            print("Episode {} took {:.2f} s. Steps: {}. Reward: {:.2f}. Running: {:.2f}".format(epi, total_time, ts, reward_sum, running_reward))

            epi += 1; reward_sum = 0; ep_start = time.time()
            obs = env.reset()
    
        if not epi % args.save_ckpt_freq and saved_ckpt_epi < epi:
            model_state = {'state_dict': policy.state_dict(), 
                           'optimizer': optimizer.state_dict(),
                           'step': ts,
                           'episode': epi,
                           'running_reward': running_reward}
            torch.save(model_state, args.ckpt_path)
            saved_ckpt_epi = epi
        
        if not epi % args.save_reward_freq and saved_reward_epi < epi:
            running_rewards.append(running_reward)
            with open(args.rewards_path, 'wb') as f:
                pickle.dump(running_rewards, f)
            saved_reward_epi = epi


def parse():
    parser = argparse.ArgumentParser(description='train reinforce on an episodic atari env')
    parser.add_argument('--resume-ckpt', type=str, default=None, metavar='path',
                        help='path the the checkpoint with which to resume training')
    parser.add_argument('--eta', type=float, default=2.5e-5, metavar='L',
                        help='learning rate for Adam (default: 2e-4)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor (default: 0.99)')
    parser.add_argument('--beta', type=float, default=0.01, metavar='B',
                        help='controls the strength of the entropy regularization' +
                        'term (default: 0.01)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=543, metavar='N',
                        help='random seed (default: 543)')
    parser.add_argument('--env', type=str, default='PongDeterministic-v0', metavar='envname',
                        help='the environment to train the model on (default: PongDeterministic-v0)')
    parser.add_argument('--nb-steps', type=int, default=int(2e8), metavar='T',
                        help='number of timesteps the agent is trained'),
    parser.add_argument('--update-freq', type=int, default=int(1e3), metavar='U',
                        help='update the net params after every that number of steps')
    parser.add_argument('--save-reward-freq', type=int, default=1, metavar='RF',
                        help='save running rewards after every that many episodes')
    parser.add_argument('--save-ckpt-freq', type=int, default=100, metavar='CF',
                        help='save model checkpoint after every that many episodes')
    parser.add_argument('--ckpt-path', type=str, default=os.path.join(os.getcwd(), 'policy.ckpt'),
                        metavar='path', help='full path of the model checkpoint file')
    parser.add_argument('--rewards-path', type=str, default=os.path.join(os.getcwd(), 'rewards.pickle'),
                        metavar='path', help='full path of the pickled rewards gotten from evaluation')
    parser.add_argument('--init-runreward', type=int, default=None, metavar='R',
                        help='initial running reward for loggig')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse()
    
    torch.manual_seed(args.seed)
    if not args.no_cuda:
        torch.cuda.manual_seed(args.seed)

    train(args)

