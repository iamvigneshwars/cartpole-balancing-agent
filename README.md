# cartpole-balancing-agent

An AI agent balances the pole on a cart from falling. The Agent uses deep Q learning to approximate the q values for state-action pairs. The DQN model consists of two networks the policy network, and the target network. The policy network is trained by sampling a batch of experiences from the memory (experience replay). The target network is used to approximate the q-values for the next states. The weights in the target network are updated after N episodes which maintain the stability of the network.
