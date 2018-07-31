import d2bot.visualizer as visualizer
import d2bot.core.game_env as game_env
import d2bot.simulator as simulator

from .ActorCritic import ActorCritic

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import random
import time

'''
    model from https://github.com/ikostrikov/pytorch-a3c
'''

class A3CEnv(game_env.GameEnv):
    def __init__(self, *args, **kwargs):
        super(A3CEnv,self).__init__(*args, **kwargs)

        random.seed(time.time())

        self.a3c_model = ActorCritic(5, 9, 128)
        self.optimizer = optim.SGD(self.a3c_model.parameters(), lr=0.01)
        
        self.entropies = []
        self.values = []
        self.rewards = []
        self.log_probs = []
        self.raw_log_probs = []
        self.predefined_steps = []

        self.gamma = 0.99
        self.tau = 1.0
        self.entropy_coef = 0.01

        self.batch_size = 256
        self.buffer_size = 1000
        self.i = 0

        self.nll_loss_fn = nn.NLLLoss()
    
    def reset(self):
        self.entropies = []
        self.values = []
        self.rewards = []
        self.log_probs = []
        self.raw_log_probs = []
        self.predefined_steps = []
    
    def train(self):
        #just make it simple
        self.a3c_model.zero_grad()
        #todo gae

        R = torch.zeros(1, 1)
        gae = torch.zeros(1, 1)

        self.values.append(R)

        policy_loss = 0
        value_loss = 0
        teacher_loss = 0

        for i in reversed(range(len(self.rewards))):
            R = self.gamma * R + self.rewards[i]
            adv = R - self.values[i]
            value_loss = value_loss + 0.5 * adv.pow(2)

            #GAE
            delta_t = self.rewards[i] + self.gamma * self.values[i + 1].data - self.values[i].data
            gae = gae * self.gamma * self.tau + delta_t

            policy_loss = policy_loss - self.log_probs[i] * Variable(gae) - self.entropy_coef * self.entropies[i]

            teacher_loss = teacher_loss + self.nll_loss_fn(self.raw_log_probs[i], self.predefined_steps[i])
        
        self.optimizer.zero_grad()
        total_loss = (policy_loss + 0.5 * value_loss + 0.1 * teacher_loss)
        print(total_loss)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm(self.a3c_model.parameters(), 10)
        self.optimizer.step()

    def _generator_run(self, input_):
        self.init_fn(input_)

        self.engine = simulator.Simulator(feature_name='unit_test1',
            actionspace_name='lattice1',
            canvas=self.canvas)
        
        self.reset()

        while self.engine.get_time() < 200:
            self.i = self.i + 1
            
            #print(dire_predefine_step)
            dire_state = self.engine.get_state_tup("Dire", 0)

            dire_predefine_step = self.engine.predefined_step("Dire",0)
            predefine_move = torch.LongTensor([dire_predefine_step[1]])

            is_end = dire_state[2]
            if is_end:
                break
            
            self.predefined_steps.append(predefine_move)
            action_out, value_out = self.a3c_model(dire_state[0])
            
            prob = F.softmax(action_out)
            log_prob = F.log_softmax(action_out)
            self.raw_log_probs.append(log_prob)
            

            entropy = - (log_prob * prob).sum(1, keepdim=True)
            self.entropies.append(entropy)

            action = prob.multinomial(num_samples=1).data
            log_prob = log_prob.gather(1,Variable(action))

            self.engine.set_order("Dire", 0, (1,action))

            self.engine.loop()

            reward = 0
            self.rewards.append(reward)
            self.values.append(value_out)
            self.log_probs.append(log_prob)
            

            yield
        
        self.train()

