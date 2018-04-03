import time
import argparse
import numpy as np
import pickle

import gym
import torch
from torch import optim
from torch.autograd import Variable
from torch.distributions import Categorical
import torch.nn.functional as F

from policies import CnnPolicy

# find parent directory and import utilities from there
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import preprocess_pong, preprocess_atari


def _build_frames(prepro_func, obs, frames, no_cuda):
    obs = prepro_func(obs)
    obs = np.expand_dims(np.expand_dims(obs, 0), 0)
    obs = Variable(torch.from_numpy(obs))
    if not no_cuda: obs = obs.cuda()
    
    # add current observation to structure that holds cosecutive frames
    frames = frames[:, :-1, :, :]
    frames = torch.cat((obs, frames), 1)
    return frames


def _discount_rewards(rewards):
    disc_rewards = np.zeros(len(rewards))
    running_add = 0
    for i in np.arange(len(rewards)-1, -1, -1):
        running_add = rewards[i] + args.gamma*running_add
        disc_rewards[i] = running_add
    return disc_rewards


def kl_div(p, q):
    return torch.mean(torch.sum(p * (torch.log(p) - torch.log(q)), 1))


def train(args):
    env = gym.make(args.env)
    obs = env.reset()

    policy = CnnPolicy(env.action_space.n)
    if not args.no_cuda: policy.cuda()
    
    frames = Variable(torch.zeros((1, 4, 80, 80)))
    if not args.no_cuda: frames = frames.cuda()
    prepro = preprocess_pong if 'Pong' in args.env else preprocess_atari
 
    # arrays for holding history and statistics
    frames_hist, rewards, action_hist = [], [], []
    
    # stuff for monitoring and logging progress
    reward_sum = 0
    epi = 0; ep_start = time.time()
    running_reward = args.init_runreward
    running_rewards = []
    saved_reward_epi = epi
    saved_ckpt_epi = epi
    start_ts = 1
    
    if args.resume_ckpt:
        checkpoint = torch.load(args.resume_ckpt)
        policy.load_state_dict(checkpoint['state_dict'])
        start_ts = checkpoint['step']
        epi = checkpoint['episode']
        running_reward = checkpoint['running_reward']

    for ts in range(start_ts, args.nb_steps+1):
        frames = _build_frames(prepro, obs, frames, args.no_cuda)
        
        action_probs = policy(frames)
        action_dist = Categorical(action_probs)
        action = action_dist.sample()
    
        obs, reward, done, _ = env.step(action.data[0])
        reward_sum += reward
    
        frames_hist.append(frames)
        action_hist.append(action)
        rewards.append(reward)
            
        if not ts % args.update_freq:
            disc_rewards = _discount_rewards(rewards)
            disc_rewards = (disc_rewards - np.mean(disc_rewards)) / (np.std(disc_rewards) + 1e-10)
            advs = Variable(torch.Tensor(disc_rewards)).cuda().detach()
                    
            frames_hist = torch.cat(frames_hist)
            action_hist = torch.cat(action_hist)
            pi_orig = policy(frames_hist)
            aprobs_orig = torch.gather(pi_orig, 1, action_hist.view(-1, 1))
                    
            orig_theta = policy.gather_flat_params().detach()

            # CG. Initial direction.
            # Gradient of linear term
            is_obj = torch.sum((aprobs_orig/aprobs_orig.detach())*advs)
            is_obj.backward(retain_graph=True)
            g = policy.gather_flat_grad().detach()
            
            # gradient from quadratic term
            kl = kl_div(pi_orig.detach(), pi_orig)
            policy.zero_grad()
            kl.backward(retain_graph=True, create_graph=True)
            g_kl = policy.gather_flat_grad()
            
            z = torch.dot(g_kl, orig_theta)
            policy.zero_grad()
            z.backward(retain_graph=True)
            Hx = policy.gather_flat_grad()
            d = g - Hx
            
            # compute dHd
            z = torch.dot(g_kl, d.detach())
            policy.zero_grad()
            z.backward(retain_graph=True)
            Hd = policy.gather_flat_grad()
            dHd = torch.dot(d, Hd)
            
            for cg_iter in range(args.nb_cgsteps):
                # Take a step in the obtained direction by first performing line search to
                # find a reasonable step size
                theta_old = policy.gather_flat_params()
                beta = torch.sqrt((2*args.stepsize) / dHd)
                i = 1
                while True:
                    theta = (theta_old + beta*d).detach()
                    policy.replace_params(theta)
                    pi = policy(frames_hist)
                    kl_indicator = 0 if kl_div(pi_orig, pi) <= args.stepsize else float("inf")

                    aprobs = torch.gather(pi, 1, action_hist.view(-1, 1))
                    cur_is_obj = torch.sum((aprobs/aprobs_orig)*advs)

                    if cur_is_obj - kl_indicator >= is_obj:
                        is_obj = cur_is_obj
                        break
                    else:
                        beta /= 2
                        i += 1
                
                print("CG iter {}/{}. Line search terminated after {} steps."
                    .format(cg_iter, args.nb_cgsteps, i))

                # Find new conjugate direction
                # Get steepest descent direction
                z = torch.dot(g_kl, theta)
                policy.zero_grad()
                z.backward(retain_graph=True)
                Hx = policy.gather_flat_grad()
                grad_f = g - Hx
                
                # Get gamma
                z = torch.dot(g_kl, d.detach())
                policy.zero_grad()
                z.backward(retain_graph=True)
                Hd = policy.gather_flat_grad()
                dHd = torch.dot(d, Hd)
                gamma = torch.dot(grad_f, Hd) / dHd

                # new direction
                d = gamma*d - grad_f

            # zero-out buffers
            frames = Variable(torch.zeros((1, 4, 80, 80)))
            if not args.no_cuda: frames = frames.cuda()
            frames_hist, action_hist, rewards = [], [], []
            
        if done:
            total_time = time.time() - ep_start
            running_reward = reward_sum if not running_reward else running_reward*0.99 + reward_sum*0.01
            print("Episode {} took {:.2f} s. Steps: {}. Reward: {:.2f}. Running: {:.2f}".format(epi, total_time, ts, reward_sum, running_reward))

            epi += 1; reward_sum = 0; ep_start = time.time()
            obs = env.reset()
    
        # if not epi % args.save_ckpt_freq and saved_ckpt_epi < epi:
        #     model_state = {'state_dict': policy.state_dict(), 
        #                    'optimizer': optimizer.state_dict(),
        #                    'step': ts,
        #                    'episode': epi,
        #                    'running_reward': running_reward}
        #     torch.save(model_state, args.ckpt_path)
        #     saved_ckpt_epi = epi
        # 
        # if not epi % args.save_reward_freq and saved_reward_epi < epi:
        #     running_rewards.append(running_reward)
        #     with open(args.rewards_path, 'wb') as f:
        #         pickle.dump(running_rewards, f)
        #     saved_reward_epi = epi


def parse():
    parser = argparse.ArgumentParser(description='train actor-critic on atari env')
    parser.add_argument('--resume-ckpt', type=str, default=None, metavar='path',
                        help='path the the checkpoint with which to resume training')
    parser.add_argument('--beta', type=float, default=0.01, metavar='B',
                        help='controls the strength of the entropy regularization' +
                        'term (default: 0.01)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor (default: 0.99)')
    parser.add_argument('--nb-cgsteps', type=int, default=10, metavar='CG',
                        help='number of conjugate gradient steps to take')
    parser.add_argument('--stepsize', type=float, default=1e-2, metavar='KL',
                        help='the desired KL divergence')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=543, metavar='N',
                        help='random seed (default: 543)')
    parser.add_argument('--env', type=str, default='PongDeterministic-v0', metavar='envname',
                        help='the environment to train the model on (default: PongDeterministic-v0)')
    parser.add_argument('--nb-steps', type=int, default=int(2e8), metavar='T',
                        help='number of timesteps the agent is trained'),
    parser.add_argument('--update-freq', type=int, default=2000, metavar='U',
                        help='update the net params after every that number of steps')
    parser.add_argument('--save-reward-freq', type=int, default=1, metavar='RF',
                        help='save running rewards after every that many episodes')
    parser.add_argument('--save-ckpt-freq', type=int, default=100, metavar='CF',
                        help='save model checkpoint after every that many episodes')
    parser.add_argument('--ckpt-path', type=str, default=os.path.join(os.getcwd(), 'policy.ckpt'),
                        metavar='path', help='full path of the model checkpoint file')
    parser.add_argument('--rewards-path', type=str, default=os.path.join(os.getcwd(), 'rewards.pickle'),
                        metavar='path', help='full path of the pickled rewards gotten from evaluation')
    parser.add_argument('--init-runreward', type=int, default=-21, metavar='R',
                        help='initial running reward for loggig')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse()
    
    torch.manual_seed(args.seed)
    if not args.no_cuda:
        torch.cuda.manual_seed(args.seed)

    train(args)

