# cartpole-balancing-agent

An AI agent balances the pole on a cart from falling. The Agent uses deep Q learning to approximate the q values for state-action pairs. The DQN model consists of two networks the policy network, and the target network. The policy network is trained by sampling a batch of experiences from the memory (experience replay). The target network is used to approximate the q-values for the next states. The weights in the target network are updated after N episodes which maintain the stability of the network.


## Run my project

To train the agent, run the following command.

```bash
python again.py train 200
```
200 is the number of episodes you want the agent to train.

To play the game with the trained agent, run the followig command.

```bash
python play 5
```

5 is the number episodes you want the agent to play.
