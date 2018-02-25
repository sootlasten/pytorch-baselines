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


def train(eta, gamma, beta, envname, no_cuda, seed):
    env = gym.make(envname)
    obs = env.reset()  # initial observation in the new episode

    policy = CnnPolicy(env.action_space.n)
    if not no_cuda: policy.cuda()
    optimizer = optim.RMSprop(policy.parameters(), lr=eta)
    
    frames = Variable(torch.zeros((1, 4, 80, 80)))  # used to hold 4 consecutive frames
    if not no_cuda: frames = frames.cuda()

    rewards, logprobs, aprobs = [], [], []  # aprobs are all softmax outputs for the entropy term
    reward_sum = 0
    epi = 0; ep_start = time.time()

    while True:  # main training loop
        obs = preprocess_pong(obs)  # TODO! Make this more general
        obs = np.expand_dims(np.expand_dims(obs, 0), 0)
        obs = Variable(torch.from_numpy(obs))
        if not no_cuda: obs = obs.cuda()
        
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
            # compute discounted returns
            running_add = 0
            for i in np.arange(len(rewards)-1, 0, -1):
                running_add = rewards[i] + gamma*running_add
                rewards[i] = running_add
        
            # normalize rewards
            norm_rewards = (rewards - np.mean(rewards)) / np.std(rewards)
            
            aprobs = torch.cat(aprobs).clamp(1e-8)  # clamp do avoid log going to nan
            entropies = -torch.sum(aprobs*torch.log(aprobs), dim=1)
            loss = (logprobs*-norm_rewards).sum() + beta*entropies.sum()
        
            # param update 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # zero-out buffers and restart game
            frames = Variable(torch.zeros((1, 4, 80, 80)))
            if not no_cuda: frames = frames.cuda()
            rewards, logprobs, aprobs = [], [], []
            obs = env.reset()
            
            # log
            total_time = time.time() - ep_start
            print("Episode {} took {:.2f} s. Reward: {:.2f}".format(epi, total_time, reward_sum))

            epi += 1; reward_sum = 0; ep_start = time.time()
    
        if not epi % 200:
            torch.save(policy.state_dict(), '/home/stensootla/projects/pytorch-baselines/reinforce/pong.pt')


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

 
def parse():
    parser = argparse.ArgumentParser(description='REINFORCE')
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

    # parser.add_argument('--render', action='store_true',
    #                     help='render the environment')
    #parser.add_argument('--log-interval', type=int, default=10, metavar='N',
    #                    help='interval between training status logs (default: 10)')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse()
    train(args.eta, args.gamma, args.beta, args.env, args.no_cuda, args.seed)

