import gym 
import time
env = gym.make("CartPole-v1")
observation = env.reset()
for _ in range(1000):
    env.render()
    action = env.action_space.sample()
    obsertvation, reward, done, info = env.step(action)
    time.sleep(0.01)

    if done:
        observation = env.reset()

env.close
