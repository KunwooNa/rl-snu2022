import torch 
import torch.nn as nn 
from torch.optim import Optimizer
import torch.nn.functional as F
import numpy as np 
from collections import deque, namedtuple
import random
import math 



###################  [Code Reference] ####################
##  Code for NoisyNet is modified based on the code from : 
##  https://github.com/Curt-Park/rainbow-is-all-you-need

class NoiseLinear(nn.Module) : 
    def __init__(self, input_dim, n_actions, std_init = 0.5) : 
        super(NoiseLinear, self).__init__()
        self.input_dim = input_dim 
        self.n_actions = n_actions
        self.std_init = std_init 
        self.weight_mu = nn.Parameter(torch.Tensor(n_actions, input_dim)) 
        self.weight_sigma = nn.Parameter(torch.Tensor(n_actions, input_dim))
        self.bias_mu = nn.Parameter(torch.Tensor(n_actions))
        self.bias_sigma = nn.Parameter(torch.Tensor(n_actions))
        self.register_buffer(
            "weight_epsilon", torch.Tensor(n_actions, input_dim)
        )
        self.register_buffer(
            "bias_epsilon", torch.Tensor(n_actions)
        )
        
        self.reset_parameters()
        self.reset_noise()

    
    def reset_parameters(self) : 
        '''
        Factorized Gaussian Noise
        '''
        mu_range = 1 / math.sqrt(self.input_dim)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(
            self.std_init / math.sqrt(self.input_dim)
        )
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(
            self.std_init / math.sqrt(self.n_actions)
        )
       

    def reset_noise(self) : 
        epsilon_in = self.scale_noise(self.input_dim)
        epsilon_out = self.scale_noise(self.n_actions)

        # outer product
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)


    @staticmethod
    def scale_noise(size : int) : 
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())
    

    def forward(self, x) : 
        return F.linear(
            x, 
            self.weight_mu + self.weight_sigma * self.weight_epsilon, 
            self.bias_mu + self.bias_sigma * self.bias_epsilon
        )
    



class EnsembleNet(nn.Module) : 
    def __init__(self, n_ensemble, input_size, n_actions) : 
        super().__init__()
        #self.fc1 = nn.Linear(60, 512)
        self.net_list = nn.ModuleList([NoiseLinear(input_size, n_actions) for k in range(n_ensemble)])
        self.n_ensemble = n_ensemble
    
    def _heads(self,x) : 
        return [net(x) for net in self.net_list] 
    
    def forward(self, x, k = None) : 
        if k : 
            return [self.net_list[k](x)]
        else : 
            return self._heads(x)

    
    def reset_noise(self) : 
        for i in range(self.n_ensemble) : 
            self.net_list[i].reset_noise()
    


class RNDNet(nn.Module):
    def __init__(self, input_size = 60, output_size = 32) : 
        super().__init__()
        self.fc = nn.Linear(input_size, output_size)
    
    def forward(self, x) : 
        x = self.fc(x) 
        return x 



#############   [Code reference]   ###############
##   PER code is modified based on the code from :
##   https://nn.labml.ai/rl/dqn/replay_buffer.html

class memoryDataset(object) : 

    def __init__(self, max_length, alpha, n_ensemble = 1, bernoulli_prob = 0.6) : 
        '''
        @param max_length : max memory size of replay buffer. 
        @param alpha : alpha value to compute PER priority. 
        @param n_ensemble : the number of distinct ensemble networks denoted K. 
        @param bernoulli_prob :  probability of including given sample to training a network
        '''
        self.max_length = max_length 
        self.memory = deque(maxlen = max_length)
        self.alpha = alpha              
        self.priority_sum = [0 for _ in range(2 * self.max_length)]      
        self.priority_min = [float('inf') for _ in range(2 * self.max_length)]
        self.max_priority = 1.
        self.n_ensemble = n_ensemble
        self.bernoulli_prob = bernoulli_prob


        ## if ensemble is 0 then there is no need to apply mask
        if n_ensemble==1:
            self.bernoulli_prob = 1

        self.subset = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done', 'mask'))
        self.next_idx = 0 
        self.size = 0



    def __len__(self) : 
        return len(self.memory)
    

    def add(self, state, action, reward, next_state, done) : 
        idx = self.next_idx        
        mask = np.random.binomial(1, self.bernoulli_prob, self.n_ensemble)
        action = np.array([action])
        reward = np.array([reward])
        done = np.array([done])
        self.memory.append(self.subset(state, action, reward, next_state, done, mask))

        self.next_idx = (idx + 1) % self.max_length
        self.size = min(self.max_length, self.size + 1)
        priority_alpha = self.max_priority ** self.alpha
        self._set_priority_min(idx, priority_alpha)         # reset minimum priority    
        self._set_priority_sum(idx, priority_alpha)         # reset sum of priority 


    def _set_priority_min(self, idx, priority_alpha) : 
        idx += self.max_length
        self.priority_min[idx] = priority_alpha
        while idx >= 2 : 
            idx //= 2 
            self.priority_min[idx] = min(self.priority_min[2 * idx], self.priority_min[2 * idx + 1])
        
    

    def _set_priority_sum(self, idx, priority) : 
        idx += self.max_length
        self.priority_sum[idx] = priority
        while idx >= 2 :
            idx //= 2 
            self.priority_sum[idx] = self.priority_sum[2 * idx] + self.priority_sum[2 * idx + 1]
    


    def _sum(self) : 
        return self.priority_sum[1]
    


    def _min(self) :
        return self.priority_min[1]
    

    def find_prefix_sum_idx(self, prefix_sum) :
        idx = 1 
        while idx < self.max_length : 
            if self.priority_sum[idx * 2] > prefix_sum : 
                idx = idx * 2 
            else : 
                prefix_sum -= self.priority_sum[2 * idx] 
                idx = 2 * idx + 1 
            
        return idx - self.max_length
    


    def sample(self, batch_size, beta, device) : 
        
        samples = {
            'weights' : np.zeros(shape = (batch_size, ), dtype = np.float32), 
            'indices' : np.zeros(shape = (batch_size, ), dtype = np.int32)
        }
        
        for i in range(batch_size) :
            p = random.random() * self._sum()               # get sample index 
            idx = self.find_prefix_sum_idx(p)
            samples['indices'][i] = idx 

        prob_min = self._min() / self._sum()                # minimum p value 
        max_weight = (prob_min * self.size) ** (-beta)      # importance sampling weight
        

        for i in range(batch_size) : 
            idx = samples['indices'][i]
            prob = self.priority_sum[idx + self.max_length] / self._sum()
            weight = (prob * self.size) ** (-beta) 
            samples['weights'][i] = weight / max_weight 
        

    
        idxs = samples['indices']
        batch = np.array([self.memory[i] for i in idxs])
        batch = self.subset(*zip(*batch))

        state = torch.tensor(np.stack(batch.state), dtype=torch.float).to(device)
        action = torch.tensor(np.stack(batch.action), dtype=torch.long).to(device)
        reward = torch.tensor(np.stack(batch.reward), dtype=torch.float).to(device)
        next_state = torch.tensor(np.stack(batch.next_state), dtype=torch.float).to(device)

        done = torch.tensor(np.stack(batch.done), dtype=torch.long).to(device)
        mask = torch.tensor(np.stack(batch.mask), dtype=torch.float).to(device)
        

        """batch includes six tensors with size [batch_size, *]"""
        batch = self.subset(state, action, reward, next_state, done, mask)

        """batch weights are needed to correct the bias error, i.e., 
            it works as an importance weight."""
        batch_weights = torch.tensor(samples['weights'], dtype = torch.float).to(device)

        """batch_idxs includes the location of batch data in the memory dataset. 
            This is necessary to cmpute new priority of sampling."""
        

        batch_idxs = samples['indices']

        return batch,  batch_idxs, batch_weights
    


    def update_priorities(self, indices, priorities) : 
        
        for idx, priority in zip(indices, priorities) : 
            self.max_priority = max(self.max_priority, priority)
            priority_alpha = priority ** self.alpha 
            self._set_priority_min(idx, priority_alpha)
            self._set_priority_sum(idx, priority_alpha)
    


    def is_full(self) : 
        return self.max_length == self.size 




class agent():
    def __init__(self, nState=60, nAction=4):
        self.start_training_timestep = 100
        self.training_period = 2
        self.memory_size = 100
        self.bernoulli_prob = 0.5
        self.target_network_update_freq = 50
        self.batch_size = 64
        self.max_gradient_norm = 10
        self.Q_lr = 1e-3
        self.RND_lr = 3e-4
        self.rnd_feature_size = 64
        self.intrinsic_reward_multiplier = 1
        self.discount = 1
        self.n_ensemble = 1
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.totalstep = 0
        self.n_actions = nAction
        self.input_feature_size = nState      # self.env.n
        self.reset_noise_freq = 1

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
        self.RNDtarget = RNDNet(
            input_size=self.input_feature_size, 
            output_size=self.rnd_feature_size
        )
        self.RNDpredictor = RNDNet(
            input_size=self.input_feature_size, 
            output_size=self.rnd_feature_size
        )

        self.Qpolicy.to(self.device)
        self.Qtarget.to(self.device)
        self.RNDtarget.to(self.device)
        self.RNDpredictor.to(self.device)


        self.alpha = 0.5
        self.beta = 0.5


        self.Qtarget_optimizer = torch.optim.Adam(params=self.Qpolicy.parameters(), lr=self.Q_lr)
        self.RNDpredictor_optimizer = torch.optim.Adam(params=self.RNDpredictor.parameters(), lr=self.RND_lr)

        self.memory = memoryDataset(max_length = 2 ** 9, alpha = self.alpha)

        self.episode_cntr = 0 



    def _get_beta(self, by_episode = True) : 
        if by_episode : 
            beta = 1 - math.exp(-self.episode_cntr / 200)
        else : 
            beta = 1 - math.exp(-self.totalstep / 1.3e10)
        return beta


    def load_weights(self):
        self.Qpolicy.load_state_dict(torch.load("./model_lava.pt"))


    def action(self, state):
        for i in range(self.n_ensemble) : 
            self.Qpolicy.reset_noise()
            self.Qtarget.reset_noise()

        if len(state.shape) == 0:
            state = np.zeros(self.input_feature_size)
            state[0] = 1

        state_feature_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
        action_values = self.Qpolicy(state_feature_tensor)[0]
        a = action_values.argmax(1).cpu().item()

        return a

    
    def _get_intrinsic_reward(self, state):
        state = torch.tensor(state).float().unsqueeze(0).to(self.device)
        rnd_target = self.RNDtarget(state)
        rnd_prediction = self.RNDpredictor(state)
        mse_error = torch.nn.functional.mse_loss(rnd_target, rnd_prediction)
        
        return mse_error

    
    def _train_step(self, batch, batch_idx, batch_weights):
        self.Qtarget_optimizer.zero_grad()
        q_target = self.Qtarget(batch.next_state)
        q_pred = self.Qpolicy(batch.state)
        loss = 0
        one_hot_action = torch.zeros(self.batch_size, self.n_actions, dtype = torch.float).to(self.device)
        action_index = torch.ravel(batch.action).reshape(-1, 1)
        one_hot_action.scatter_(1, action_index, 1)
        pred = torch.sum(q_pred[0].mul(one_hot_action), dim = 1)
        td_error = batch.reward.view(-1) + (1 - batch.done.view(-1)) * self.discount * torch.max(q_target[0], axis = 1)[0].view(-1) - pred
        
        weights = torch.tensor(batch_weights, dtype = torch.float).to(self.device)
        
        
        
        for k in range(self.n_ensemble):
            q_pred_ = torch.gather(q_pred[k], 1, batch.action)
            q_target_ = batch.reward + self.discount * torch.max(q_target[k], axis=1)[0].unsqueeze(-1)
            # q_target_ = batch.reward + self.discount * torch.max(q_target[k], axis=1)[0]

            total_used = torch.sum(batch.mask[:,k])
            if total_used > 0:
                loss = torch.nn.functional.smooth_l1_loss(weights * batch.mask[:, k] * q_pred_, weights * batch.mask[:, k] * q_target_, 
                                                          reduction = 'sum')
        
        loss /= batch.mask.sum()
        loss.backward()
        nn.utils.clip_grad_norm_(self.Qpolicy.parameters(), max_norm=self.max_gradient_norm)
        self.Qtarget_optimizer.step()
        td_error = torch.abs(td_error).cpu().detach()
        self.memory.update_priorities(batch_idx, td_error)



    def update(self, state_feature, action, reward, next_state_feature, done):
        if len(state_feature.shape) == 0:
            state_feature = np.zeros(self.input_feature_size)
            state_feature[0] = 1

        intrinsic_reward = self._get_intrinsic_reward(state_feature)

        self.RNDpredictor_optimizer.zero_grad()
        intrinsic_reward.backward()
        self.RNDpredictor_optimizer.step()

        reward += self.intrinsic_reward_multiplier * intrinsic_reward.cpu().item()
        
        self.totalstep += 1
        self.memory.add(state_feature, action, reward, next_state_feature, done)

        if self.totalstep >= self.start_training_timestep and self.totalstep % self.training_period == 0:
            
            batch, batch_idx, batch_weights = self.memory.sample(self.batch_size, self._get_beta(by_episode = False), self.device)
            self._train_step(batch, batch_idx, batch_weights)

        if self.totalstep % self.target_network_update_freq == 0:
            self.Qtarget.load_state_dict(self.Qpolicy.state_dict())

        if done : 
            self.episode_cntr += 1


    

if __name__ == "__main__":
    from interaction_lava import calculate_sample_efficiency
    import sys, os
    sys.path.insert(
        0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir)))
    )
    from lava_grid import ZigZag6x10

    env = ZigZag6x10(max_steps=100, act_fail_prob=0, goal=(5, 9), numpy_state=False)
    n_games = 3000
    agent = agent()

    returns = calculate_sample_efficiency(n_games, env, agent)
        
    torch.save(
        agent.Qpolicy.state_dict(), 
        "./model_lava.pt"
    )

    