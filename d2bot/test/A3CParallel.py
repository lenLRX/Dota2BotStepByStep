import d2bot.visualizer as visualizer
import d2bot.core.game_env as game_env
import d2bot.simulator as simulator
import d2bot.core.parallel as parallel

from d2bot.torch.a3c.ActorCritic import ActorCritic

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from torch.autograd import Variable

import random
import time
import sys
import os
import math

from tensorboardX import SummaryWriter

'''
    model from https://github.com/ikostrikov/pytorch-a3c
'''

lr = 0.01
num_hidden = 32

class ParallelA3CPPOEnv(game_env.GameEnv):
    def __init__(self, *args, **kwargs):
        super(ParallelA3CPPOEnv,self).__init__(*args, **kwargs)

        random.seed(os.getpid())
        torch.random.manual_seed(os.getpid())

        self.game_no = 0

        self.out_classes = 9

        self.a3c_model = ActorCritic(5, self.out_classes, num_hidden)
        self.optimizer = optim.SGD(self.a3c_model.parameters(), lr=lr)
        
        self.reset()

        self.gamma = 0.9
        self.tau = 1.0
        self.entropy_coef = 0.01
        self.epsilon = 0.2

        self.batch_size = 256
        self.buffer_size = 1000
        self.i = 0

        self.update_steps = 0

        self.nll_loss_fn = nn.NLLLoss()

        self.writer = SummaryWriter(comment='_%d'%os.getpid())

        self.rank=-1
    
    def set_rank(self, rank):
        self.rank = rank
    
    def set_model(self, model):
        self.a3c_model = model
        self.optimizer = optim.SGD(self.a3c_model.parameters(), lr=lr)
    
    def get_model(self):
        return self.a3c_model
    
    def reset(self):
        self.states = []
        self.actions = []
        self.entropies = []
        self.values = []
        self.rewards = []
        self.log_probs = []
        self.raw_log_probs = []
        self.raw_probs = []
        self.predefined_steps = []
    
    def update(self):
        print("Game %d updating"%self.game_no)
        self.optimizer.step()
        self.a3c_model.zero_grad()
        self.optimizer.zero_grad()
    
    
    
    def ppo_train_actor(self, old_model):
        self.a3c_model.zero_grad()
        self.optimizer.zero_grad()

        l = 0.0
        R = torch.zeros(1, 1)

        reduced_r = []
        for i in reversed(range(len(self.rewards))):
            R = self.gamma * R + self.rewards[i]
            reduced_r.append(R)
        
        reduced_r = list(reversed(reduced_r))

        idxs = list(range(len(self.rewards)))
        random.shuffle(idxs)
        idxs = idxs[:self.batch_size]

        #TODO: turn `for loop` to tensor operations
        for i in idxs:
            new_prob, v = self.a3c_model(self.states[i])
            new_prob = F.softmax(new_prob)
            old_prob, _ = old_model(self.states[i])
            old_prob = F.softmax(old_prob)
            adv = reduced_r[i] - v.data
            onehot_act = torch.zeros(self.out_classes)
            onehot_act[self.actions[i]] = 1

            ratio = torch.sum(new_prob * onehot_act) / torch.sum(old_prob * onehot_act)
            surr = ratio * adv

            l = l - min(surr, torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon)*adv)
        
        l = l / self.batch_size
        
        l.backward(retain_graph=True)
        self.writer.add_scalar("train/policy_loss", l.item())
        self.optimizer.step()
    
    def ppo_train_critic(self):
        
        self.a3c_model.zero_grad()
        self.optimizer.zero_grad()

        R = torch.zeros(1, 1)
        l = 0.0

        reduced_r = []
        for i in reversed(range(len(self.rewards))):
            R = self.gamma * R + self.rewards[i]
            reduced_r.append(R)
        
        reduced_r = list(reversed(reduced_r))

        idxs = list(range(len(self.rewards)))
        random.shuffle(idxs)
        idxs = idxs[:self.batch_size]

        for i in idxs:
            adv = reduced_r[i] - self.a3c_model(self.states[i])[1]
            l = l + adv ** 2
        
        l = l / self.batch_size
        l.backward(retain_graph=True)

        self.writer.add_scalar("train/value_loss", l.item())
        self.optimizer.step()
    
    def teacher_train(self):
        #TODO: teacher_loss * (1 - Gini coefficient)
        #TODO: entropy loss
        self.a3c_model.zero_grad()
        self.optimizer.zero_grad()
        teacher_loss = 0

        labels = torch.cat(self.predefined_steps)

        #balance loss
        weight = torch.zeros((self.out_classes,))
        for i in range(self.out_classes):
            weight[i] = torch.sum(labels==i)
            if weight[i] > 0:
                weight[i] = 1./ weight[i]
        
        nll = nn.NLLLoss(weight=weight)

        log_probs =  torch.cat(self.raw_log_probs)

        teacher_loss = nll(log_probs, labels) * 0.1#0.1 as coeff
        teacher_loss.backward(retain_graph=True)

        self.writer.add_scalar("train/teacher_loss",teacher_loss.item() )
        print("train/teacher_loss",teacher_loss.item())

        self.optimizer.step()
    
    def train(self):
        #just make it simple
        self.a3c_model.train()

        old_model = self.a3c_model.clone()

        self.teacher_train()

        for _ in range(self.update_steps):
            self.ppo_train_actor(old_model)

        for _ in range(self.update_steps):
            self.ppo_train_critic()
        
        print("reward %f"%sum(self.rewards))

        self.writer.add_scalar("train/rewards",sum(self.rewards))
        self.writer.add_scalar("train/values",sum(self.values).item() / len(self.values))
        self.writer.add_scalar("train/entropy",sum(self.entropies).item() / len(self.entropies))

        acts = torch.cat(self.raw_log_probs)
        pd = torch.tensor(self.predefined_steps)

    def _generator_run(self, input_):
        self.game_no = self.game_no + 1
        self.init_fn(input_)

        self.engine = simulator.Simulator(feature_name='unit_test1',
            actionspace_name='lattice1',
            canvas=self.canvas)
        
        self.reset()

        while self.engine.get_time() < 30:
            self.i = self.i + 1
            
            #print(dire_predefine_step)
            dire_state = self.engine.get_state_tup("Dire", 0)

            dire_predefine_step = self.engine.predefined_step("Dire",0)
            predefine_move = torch.LongTensor([dire_predefine_step[1]])

            is_end = dire_state[2]
            if is_end:
                break
            
            self.predefined_steps.append(predefine_move)
            state_now = dire_state[0]
            self.states.append(state_now)
            action_out, value_out = self.a3c_model(state_now)
            
            prob = F.softmax(action_out)
            self.raw_probs.append(prob)
            log_prob = F.log_softmax(action_out)
            self.raw_log_probs.append(log_prob)

            entropy = - (log_prob * prob).sum(1, keepdim=True)
            self.entropies.append(entropy)

            #action = prob.multinomial(num_samples=1).data

            action = torch.argmax(log_prob, 1).data.view(-1,1)
            self.actions.append(action)
            log_prob = log_prob.gather(1,Variable(action))

            self.engine.set_order("Dire", 0, (1,action))

            self.engine.loop()

            reward = dire_state[1]
            self.rewards.append(reward)
            self.values.append(value_out)
            self.log_probs.append(log_prob)
            

            yield
        print("rank %d os.pid %d"%(self.rank,os.getpid()))
        self.train()
        torch.save(self.a3c_model.state_dict(), "./tmp/model_%d_%d"%(self.game_no, os.getpid()))

def test(model):
    v = visualizer.Visualizer(ParallelA3CPPOEnv)
    v.env.set_model(model)
    v.visualize()
    

def test_without_gui(clazz, model, rank, args):
    #1 trainer per cpu core
    os.environ['OMP_NUM_THREADS'] = '1'
    print("rank %d os.pid=%d"%(rank, os.getpid()))
    env = clazz()
    env.set_model(model)
    env.set_rank(rank)
        

    gen = env.run(True)
    gen.send(None)
    while True:
        try:
            gen.send((None,))
        except StopIteration:
            if rank == 0:
                torch.save(env.a3c_model.state_dict(), "./tmp/model_%d_%d"%(env.game_no, os.getpid()))
            gen = env.run(True)
            gen.send(None)

if __name__ == '__main__':
    model = ActorCritic(5, 9, num_hidden)
    import torch.nn.init as weight_init
    for name, param in model.named_parameters(): 
        weight_init.normal(param); 
    model.share_memory()
    #test(model)

    parallel.start_parallel(ParallelA3CPPOEnv, model, np=os.cpu_count(), func=test_without_gui, args=None)

