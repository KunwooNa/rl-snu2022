import torch 
import torch.nn as nn 
from torch.optim import Optimizer
import numpy as np 
from collections import deque, namedtuple
import random


###############################  [Code Reference] ##################################
##  Code for Bootstrapped DQN and memoryDataset is modified based on the code from : 
##  https://github.com/JoungheeKim/bootsrapped-dqn

class HeadNet(nn.Module) : 
    def __init__(self, input_size = 60, n_actions = 4) : 
        super().__init__()
        self.fc = nn.Linear(input_size, n_actions) 
    
    def forward(self, x) : 
        x = self.fc(x) 
        return x 
    

class EnsembleNet(nn.Module) : 
    def __init__(self, n_ensemble, input_size = 60, n_actions = 4) : 
        super().__init__()
        self.net_list = nn.ModuleList([HeadNet(input_size, n_actions) for k in range(n_ensemble)])

    def _heads(self,x) : 
        return [net(x) for net in self.net_list] 
    
    def forward(self, x, k = None) : 
        if k : 
            return [self.net_list[k](x)]
        else : 
            return self._heads(x)


class RNDNet(nn.Module):
    def __init__(self, input_size = 60, output_size = 32) : 
        super().__init__()
        self.fc = nn.Linear(input_size, output_size)
    
    def forward(self, x) : 
        x = self.fc(x) 
        return x 



class memoryDataset(object):
    def __init__(self, maxlen, n_ensemble=1, bernoulli_prob=0.9):
        """
        maxlen: max replay memory size
        n_ensemble: the number of different networks (K)
        bernoulli_prob: probability of including given sample to training a network
        """
        self.memory = deque(maxlen=maxlen)
        self.n_ensemble = n_ensemble
        self.bernoulli_prob = bernoulli_prob

        ## if ensemble is 0 then no need to apply mask
        if n_ensemble==1:
            self.bernoulli_prob = 1

        self.subset = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done', 'mask'))


    def push(self, state, action, reward, next_state, done):
        action = np.array([action])
        reward = np.array([reward])
        done = np.array([done])
        mask = np.random.binomial(1, self.bernoulli_prob, self.n_ensemble)

        self.memory.append(self.subset(state, action, reward, next_state, done, mask))

    def __len__(self):
        return len(self.memory)

    def sample(self, batch_size, device):
        
        batch = random.sample(self.memory, min(len(self.memory), batch_size))
        batch = self.subset(*zip(*batch))

        state = torch.tensor(np.stack(batch.state), dtype=torch.float).to(device)
        action = torch.tensor(np.stack(batch.action), dtype=torch.long).to(device)
        reward = torch.tensor(np.stack(batch.reward), dtype=torch.float).to(device)
        next_state = torch.tensor(np.stack(batch.next_state), dtype=torch.float).to(device)

        done = torch.tensor(np.stack(batch.done), dtype=torch.long).to(device)
        mask = torch.tensor(np.stack(batch.mask), dtype=torch.float).to(device)

        """batch includes six tensors with size [batch_size, *]"""
        batch = self.subset(state, action, reward, next_state, done, mask)

        return batch


class agent():
    def __init__(self, nState=10, nAction=2):
        self.start_training_timestep = 100
        self.training_period = 1
        self.memory_size = 100
        self.bernoulli_prob = 0.5
        self.target_network_update_freq = 50
        self.batch_size = 32
        self.max_gradient_norm = 1
        self.Q_lr = 1e-3
        self.discount = 1
        self.n_ensemble = 10
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.totalstep = 0
        self.n_actions = nAction
        self.input_feature_size = nState      # self.env.n
        self.k = 0


        self.Qpolicy = EnsembleNet(
            n_ensemble=self.n_ensemble, 
            input_size=self.input_feature_size, 
            n_actions=self.n_actions
        )
        self.Qtarget = EnsembleNet(
            n_ensemble=self.n_ensemble, 
            input_size=self.input_feature_size, 
            n_actions=self.n_actions
        )

        self.Qpolicy.to(self.device)
        self.Qtarget.to(self.device)

        self.Qtarget_optimizer = torch.optim.Adam(params=self.Qpolicy.parameters(), lr=self.Q_lr)

        self.memory = memoryDataset(maxlen=self.memory_size, n_ensemble=self.n_ensemble,
                                    bernoulli_prob=self.bernoulli_prob)



    def load_weights(self):
        self.Qpolicy.load_state_dict(torch.load("./model_chainMDP.pt"))

    def action(self, state):
     
        # state_index = state.argmax()  <-- Code for vanilla-UCB 
        state_feature_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
        action_values = self.Qpolicy(state_feature_tensor)[self.k]
        a = action_values.argmax(1).cpu().item()

        return a

    
    def _train_step(self, batch):
        self.Qtarget_optimizer.zero_grad()
        q_target = self.Qtarget(batch.next_state)
        q_pred = self.Qpolicy(batch.state)

        loss = 0
        for k in range(self.n_ensemble):
            q_pred_ = torch.gather(q_pred[k], 1, batch.action)
            q_target_ = batch.reward + self.discount * torch.max(q_target[k], axis=1)[0].unsqueeze(-1)

            total_used = torch.sum(batch.mask[:,k])
            if total_used > 0:
                loss += torch.nn.functional.smooth_l1_loss(batch.mask[:, k] * q_pred_, batch.mask[:, k] * q_target_, reduction='sum')
        loss /= batch.mask.sum()
        loss.backward()
        nn.utils.clip_grad_norm_(self.Qpolicy.parameters(), max_norm=self.max_gradient_norm)
        self.Qtarget_optimizer.step()


    def update(self, state_feature, action, reward, next_state_feature, done):
        self.totalstep += 1
        self.memory.push(state_feature, action, reward, next_state_feature, done)

        if self.totalstep >= self.start_training_timestep and self.totalstep % self.training_period == 0:
            batch = self.memory.sample(self.batch_size, self.device)
            self._train_step(batch)

        if self.totalstep % self.target_network_update_freq == 0:
            self.Qtarget.load_state_dict(self.Qpolicy.state_dict())



if __name__ == "__main__":
    from interaction_chainMDP import calculate_sample_efficiency
    import sys, os
    sys.path.insert(
        0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir)))
    )
    from chain_mdp import ChainMDP

    env = ChainMDP(10)
    n_games = 1000
    agent = agent()

    calculate_sample_efficiency(n_games, env, agent)
        
    torch.save(
        agent.Qpolicy.state_dict(), 
        "./model_chainMDP.pt"
    )
