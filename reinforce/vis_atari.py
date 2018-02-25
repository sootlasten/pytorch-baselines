

def test():
    policy = Policy()
    policy.load_state_dict(torch.load('pong.pt'))
    
    env = gym.make('Pong-v0')

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

