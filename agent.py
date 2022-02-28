import gym
import time
env = gym.make('CartPole-v1')
env.reset()

# done = False

# for _ in range(1000):
#     env.render()
#     time.sleep(0.01)
#     state, reward, done, info = env.step(env.action_space.sample())
#     if done == True:
#         env.reset()

#     # if KeyboardInterrupt:
#     #     env.close()
#     #     break
# env.close()
        
