import gym


env_sim = gym.make("CarRacing-v2", render_mode="human")

s = env_sim.reset()
totalReward = 0

for _ in range(300):
    env_sim.render()
    a = env_sim.action_space.sample() # случайная стратегия
    s, r, done, trunc, _ = env_sim.step(a)
    totalReward += r
    if r != 0:
        print('New reward = {}'.format(r))
    if done:
        break

env_sim.close()

print('Total reward = {}'.format(totalReward))