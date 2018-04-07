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


class TRPO():
    
    def __kl_div(self, p, q):
        return torch.mean(torch.sum(p * (torch.log(p) - torch.log(q)), 1))


    def __hessian_vec_prod(self, v):
        # Note: torch.autograd.grad doesn't accumulate gradients 
        # into the .grad buffers, but instead returns gradients 
        # as Variable tuples. Hence, no zero_grad() is needed.
        kl = self.__kl_div(self.pi_old.detach(), self.pi_old)
        g_kl = torch.autograd.grad(kl, self.policy.parameters(), create_graph=True)
        g_kl = torch.cat([p.view(-1) for p in g_kl])
        grad_vec_prod = torch.dot(g_kl, v)
        Hv = torch.autograd.grad(grad_vec_prod, self.policy.parameters(), create_graph=True)
        return torch.cat([p.contiguous().view(-1) for p in Hv]) + args.cg_damp*v


    # def __conjugate_gradient(self, b):
    #     """Optimizer conjugate-gradient, Nocedal & Wright algorithm 5.2"""
    #     x = torch.zeros_like(b)
    #     r = b - self.__hessian_vec_prod(x)
    #     p = -r
    #     
    #     for cg_iter in range(args.nb_cgsteps):
    #         rr = torch.dot(r, r)
    #         Ap = self.__hessian_vec_prod(p.detach())
    #         alpha = rr / torch.dot(p, Ap)
    #         x += alpha*p
    #         r -= alpha*Ap
    #         beta = torch.dot(r, r) / rr
    #         p = -r + beta*p
    # 
    #     return x
    

    def __conjugate_gradient(self, b):
        """Optimizer conjugate-gradient, Nocedal & Wright algorithm 5.2"""
        """Code adapted from https://github.com/openai/baselines/blob/master/baselines/common/cg.py"""
        p = b.clone()
        r = b.clone()
        x = torch.zeros_like(b)
        rdotr = torch.dot(r, r)
        
        for cg_iter in range(args.nb_cgsteps):
            z = self.__hessian_vec_prod(p.detach())
            v = rdotr / torch.dot(p, z)
            x += v*p
            r -= v*z
            newrdotr = torch.dot(r, r)
            mu = newrdotr/rdotr
            p = r + mu*p
            rdotr = newrdotr
        return x
        

    def __line_search(self, s, prev_is_obj):
        theta_old = self.policy.gather_flat_params()

        sAs = torch.dot(s, self.__hessian_vec_prod(s.detach()))
        beta = torch.sqrt((2*args.stepsize) / sAs)

        nb_iters = 1
        while True:
            theta = (theta_old + beta*s)
            self.policy.replace_params(theta)
            pi = self.policy(self.frames_hist)
            kl_indicator = 0 if self.__kl_div(self.pi_old, pi) <= args.stepsize else float("inf")
    
            aprobs = torch.gather(pi, 1, self.action_hist.view(-1, 1))
            entropies = -torch.sum(aprobs*torch.log(aprobs), dim=1)
            is_obj = torch.sum((aprobs/self.aprobs_old)*self.advs) + args.ent_coeff*entropies.sum()
    
            if is_obj - kl_indicator >= prev_is_obj: break
            beta /= 2
            nb_iters += 1
            
            if nb_iters > 100:
                print("WARNING! Line search didn't terminate in 100 steps.")
                return
                
        print("Line search terminated after {} steps.".format(nb_iters))


    def train(self, args):
        env = gym.make(args.env)
        obs = env.reset()
    
        self.policy = CnnPolicy(env.action_space.n)
        if not args.no_cuda: self.policy.cuda()
        
        frames = Variable(torch.zeros((1, 4, 80, 80)))
        if not args.no_cuda: frames = frames.cuda()
        prepro = preprocess_pong if 'Pong' in args.env else preprocess_atari
     
        # arrays for holding history and statistics
        frames_hist, action_hist, rewards = [], [], []
        
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
            self.policy.load_state_dict(checkpoint['state_dict'])
            start_ts = checkpoint['step']
            epi = checkpoint['episode']
            running_reward = checkpoint['running_reward']
    
        for ts in range(start_ts, args.nb_steps+1):
            frames = _build_frames(prepro, obs, frames, args.no_cuda)
            
            action_probs = self.policy(frames)
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
                self.advs = Variable(torch.Tensor(disc_rewards)).cuda().detach()
                        
                self.frames_hist = torch.cat(frames_hist)
                self.action_hist = torch.cat(action_hist)
                self.pi_old = self.policy(self.frames_hist)
                self.aprobs_old = torch.gather(self.pi_old, 1, self.action_hist.view(-1, 1))
                        
                # Gradient of linear term
                entropies = -torch.sum(self.aprobs_old*torch.log(self.aprobs_old), dim=1)
                is_obj = torch.sum((self.aprobs_old/self.aprobs_old.detach())*self.advs) + args.ent_coeff*entropies.sum()
                is_obj.backward(retain_graph=True)
                g = self.policy.gather_flat_grad()

                p = self.__conjugate_gradient(g)
                self.__line_search(p, is_obj)

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
        
            if not epi % args.save_ckpt_freq and saved_ckpt_epi < epi:
                model_state = {'state_dict': policy.state_dict(), 
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
    parser.add_argument('--ent-coeff', type=float, default=0.01, metavar='E',
                        help='controls the strength of the entropy regularization' +
                        'term (default: 0.01)')
    parser.add_argument('--cg_damp', type=float, default=0.001, metavar='D',
                        help='Hessian damping (?) coefficient for conjugate gradient (default: 0.001)')
    parser.add_argument('--stepsize', type=float, default=1e-3, metavar='KL',
                        help='the desired KL divergence')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=666, metavar='N',
                        help='random seed (default: 666)')
    parser.add_argument('--env', type=str, default='PongDeterministic-v0', metavar='envname',
                        help='the environment to train the model on (default: PongDeterministic-v0)')
    parser.add_argument('--nb-steps', type=int, default=int(2e8), metavar='T',
                        help='number of timesteps the agent is trained'),
    parser.add_argument('--update-freq', type=int, default=1000, metavar='U',
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

    trpo = TRPO()
    trpo.train(args)

