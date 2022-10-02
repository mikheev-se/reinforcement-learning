import gym
from sys import argv

env = gym.make(argv[1], render_mode='human')
env.reset()

for t in range(1234567890):
    act = env.action_space.sample()
    observations, reward, done, i = env.step(act)
    print(f't: {t} | action: {act} | observations: {observations}')
    if done:
        break

env.close()
