import d2bot.visualizer as visualizer
import d2bot.core.game_env as game_env
import d2bot.simulator as simulator
import d2bot.core.parallel as parallel

from net import ActorCritic_CNN

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

num_hidden = 128
lr = 0.01

class StatePkg:
    def __init__(self):
        self.states = []
        self.actions = []
        self.entropies = []
        self.values = []
        self.rewards = []
        self.log_probs = []
        self.raw_log_probs = []
        self.raw_probs = []
        self.predefined_steps = []


class ParallelA3CPPOEnv(game_env.GameEnv):
    def __init__(self, *args, **kwargs):
        super(ParallelA3CPPOEnv,self).__init__(*args, **kwargs)

        random.seed(os.getpid())
        torch.random.manual_seed(os.getpid())

        self.game_no = 0

        self.out_classes = 9

        self.a3c_model = ActorCritic_CNN(1, 40, self.out_classes, num_hidden)
        self.optimizer = optim.SGD(self.a3c_model.parameters(), lr=lr)

        self.state_buffer = []

        self.gamma = 0.5
        self.tau = 1.0
        self.entropy_coef = 0.01
        self.epsilon = 0.2

        self.batch_size = 128
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
        
    
    def update(self):
        print("Game %d updating"%self.game_no)
        self.optimizer.step()
        self.a3c_model.zero_grad()
        self.optimizer.zero_grad()
    
    
    
    def ppo_train_actor(self, old_model, state_pkgs):
        self.a3c_model.zero_grad()
        self.optimizer.zero_grad()

        l = 0.0

        for state_pkg in state_pkgs:
            l = l + self.ppo_train_actor_single(old_model, state_pkg)
        
        self.writer.add_scalar("train/policy_loss", l)
        self.optimizer.step()

    def ppo_train_actor_single(self, old_model, state_pkg):

        l = 0.0
        R = torch.zeros(1, 1)

        reduced_r = []
        for i in reversed(range(len(state_pkg.rewards))):
            R = self.gamma * R + state_pkg.rewards[i]
            reduced_r.append(R)
        
        reduced_r = list(reversed(reduced_r))

        idxs = list(range(len(state_pkg.rewards)))
        random.shuffle(idxs)
        idxs = idxs[:self.batch_size]

        #TODO: turn `for loop` to tensor operations
        for i in idxs:
            new_prob, v = self.a3c_model(state_pkg.states[i])
            new_prob = F.softmax(new_prob)
            old_prob, _ = old_model(state_pkg.states[i])
            old_prob = F.softmax(old_prob)
            adv = reduced_r[i] - v.data
            onehot_act = torch.zeros(self.out_classes)
            onehot_act[state_pkg.actions[i]] = 1

            ratio = torch.sum(new_prob * onehot_act) / torch.sum(old_prob * onehot_act)
            surr = ratio * adv

            l = l - min(surr, torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon)*adv)
        
        l = l / self.batch_size
        
        l.backward(retain_graph=True)
        return l.item()
        
    
    def ppo_train_critic(self, state_pkgs):
        self.a3c_model.zero_grad()
        self.optimizer.zero_grad()

        l = 0.0

        for state_pkg in state_pkgs:
            l = l + self.ppo_train_critic_single(state_pkg)
        
        self.writer.add_scalar("train/value_loss", l)
        print("train/value_loss", l)
        self.optimizer.step()


    def ppo_train_critic_single(self, state_pkg):
        R = torch.zeros(1, 1)
        l = 0.0

        reduced_r = []
        for i in reversed(range(len(state_pkg.rewards))):
            R = self.gamma * R + state_pkg.rewards[i]
            reduced_r.append(R)
        
        reduced_r = list(reversed(reduced_r))

        idxs = list(range(len(state_pkg.rewards)))
        random.shuffle(idxs)
        idxs = idxs[:self.batch_size]

        for i in idxs:
            adv = reduced_r[i] - self.a3c_model(state_pkg.states[i])[1]
            l = l + adv ** 2
        
        l = l / self.batch_size
        l.backward(retain_graph=True)

        return l.item()

        
    
    def teacher_train(self, state_pkgs):
        #TODO: teacher_loss * (1 - Gini coefficient)
        #TODO: entropy loss
        self.a3c_model.zero_grad()
        self.optimizer.zero_grad()
        teacher_loss = 0

        for state_pkg in state_pkgs:
            labels = torch.cat(state_pkg.predefined_steps)

            #balance loss
            weight = torch.zeros((self.out_classes,))
            for i in range(self.out_classes):
                weight[i] = torch.sum(labels==i)
                if weight[i] > 0:
                    weight[i] = 1./ weight[i]
            
            nll = nn.NLLLoss(weight=weight)

            log_probs =  torch.cat(state_pkg.raw_log_probs)

            teacher_loss = teacher_loss + nll(log_probs, labels) * 0.1#0.1 as coeff
            teacher_loss.backward(retain_graph=True)

        self.writer.add_scalar("train/teacher_loss",teacher_loss.item() )
        print("train/teacher_loss",teacher_loss.item())

        self.optimizer.step()
    
    def train(self, state_pkgs):
        #just make it simple
        self.a3c_model.train()

        old_model = self.a3c_model.clone()

        self.teacher_train(state_pkgs)

        for _ in range(self.update_steps):
            self.ppo_train_actor(old_model,state_pkgs)

        for _ in range(self.update_steps):
            self.ppo_train_critic(state_pkgs)
        
        total_reward = 0.0

        for state_pkg in state_pkgs:
            total_reward = total_reward + sum(state_pkg.rewards)
        
        avg_reward = total_reward

        if len(state_pkgs) > 1:
            avg_reward = avg_reward / len(state_pkgs)
        
        print("reward %f pid=%d"%(avg_reward, os.getpid()))

        self.writer.add_scalar("train/rewards", avg_reward)
        #self.writer.add_scalar("train/values",sum(self.values).item() / len(self.values))
        #self.writer.add_scalar("train/entropy",sum(self.entropies).item() / len(self.entropies))

    def _generator_run(self, input_):
        self.game_no = self.game_no + 1
        self.init_fn(input_)

        self.engine = simulator.Simulator(feature_name='Lattice1',
            actionspace_name='lattice1',
            canvas=self.canvas)
        
        state_pkg = StatePkg()

        while self.engine.get_time() < 30:
            self.i = self.i + 1
            
            #print(dire_predefine_step)
            dire_state = self.engine.get_state_tup("Dire", 0)

            dire_predefine_step = self.engine.predefined_step("Dire",0)
            predefine_move = torch.LongTensor([dire_predefine_step[1]])

            is_end = dire_state[2]
            if is_end:
                break
            
            state_pkg.predefined_steps.append(predefine_move)
            state_now = dire_state[0]
            state_pkg.states.append(state_now)
            action_out, value_out = self.a3c_model(state_now)
            
            prob = F.softmax(action_out)
            state_pkg.raw_probs.append(prob)
            log_prob = F.log_softmax(action_out)
            state_pkg.raw_log_probs.append(log_prob)

            entropy = - (log_prob * prob).sum(1, keepdim=True)
            state_pkg.entropies.append(entropy)

            #print(torch.max(prob).data)
            max_prob = torch.max(prob).data

            
            if max_prob > 0.9:
                action = torch.argmax(log_prob, 1).data.view(-1,1)
            else:
                action = prob.multinomial(num_samples=1).data
            

            #action = torch.argmax(log_prob, 1).data.view(-1,1)
            state_pkg.actions.append(action)
            log_prob = log_prob.gather(1,Variable(action))

            self.engine.set_order("Dire", 0, (1,action))

            self.engine.loop()

            reward = dire_state[1]
            state_pkg.rewards.append(reward)
            state_pkg.values.append(value_out)
            state_pkg.log_probs.append(log_prob)
            

            yield
        print("rank %d os.pid %d"%(self.rank,os.getpid()))

        self.state_buffer.append(state_pkg)

        self.train(self.state_buffer)

        self.state_buffer = self.state_buffer[-2:]

        torch.save(self.a3c_model.state_dict(), "./tmp/model_%d_%d"%(self.game_no, os.getpid()))

def test():
    v = visualizer.Visualizer(ParallelA3CPPOEnv)
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
    model = ActorCritic_CNN(1, 40, 9, num_hidden)
    import torch.nn.init as weight_init
    for name, param in model.named_parameters():
        if name.endswith('bias'):
            continue
        print('inited: ' + name)
        weight_init.normal(param)
    test()
    #model.share_memory()

    #parallel.start_parallel(ParallelA3CPPOEnv, model, np=1, func=test_without_gui, args=None)

