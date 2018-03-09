import pickle
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.transform import resize

import time


def preprocess_atari(frame):
    """Preprocess the input DQN-style."""
    frame = rgb2gray(frame) * 255
    frame = resize(frame, (80, 80))
    return frame.astype(np.float32)


def preprocess_pong(frame):
  frame = frame[35:195] # crop
  frame = frame[::2,::2,0] # downsample by factor of 2
  frame[frame == 144] = 0 # erase background (background type 1)
  frame[frame == 109] = 0 # erase background (background type 2)
  frame[frame != 0] = 1 # everything else (paddles, ball) just set to 1
  return frame.astype(np.float32)
    

def vis_rewards(filepath):
    with open(filepath, 'rb') as f:
        rewards = pickle.load(f)
    plt.ylabel('Reward')
    plt.plot(rewards)
    plt.show()


if __name__ == '__main__':
    vis_rewards('/home/stensootla/projects/pytorch-baselines/actor_critic/experiments/rewards.pickle')
    pass
    
    
