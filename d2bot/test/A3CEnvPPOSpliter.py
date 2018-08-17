import d2bot.visualizer as visualizer
import d2bot.core.game_env as game_env
import d2bot.simulator as simulator

from d2bot.torch.a3c.ActorCritic import ActorCritic

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import random
import time
import sys

from tensorboardX import SummaryWriter
writer = SummaryWriter()

'''
    model from https://github.com/ikostrikov/pytorch-a3c
'''


class Spliter(torch.nn.Module):

    def __init__(self, num_inputs, num_hidden):
        super(Spliter, self).__init__()

        self.num_hidden = num_hidden
        self.num_inputs = num_inputs
        self.num_outs = 2

        self.lstm_layer = nn.LSTM(num_inputs, self.num_hidden)

        self.linear = nn.Linear(self.num_hidden, self.num_outs)
        self.train()
    
    def forward(self, inputs):
        t = torch.FloatTensor(inputs)
        t = t.view(len(inputs),1,-1)
        
        lstm_outs,_ = self.lstm_layer(t)
        lstm_out = lstm_outs[-1]
        lstm_out = F.tanh(lstm_out)
        out = self.linear(lstm_out)
        return out

class SpliterWarpper:
    def __init__(self,num_inputs, num_hidden):
        self.nn_spliter = Spliter(num_inputs, num_hidden)
        self.optimizer = optim.SGD(self.nn_spliter.parameters(), lr=0.01)
        self.outs = []
        self.nll_loss_fn = nn.NLLLoss()
    
    def reset(self):
        self.outs.clear()
    
    def step(self, inputs):
        out = F.log_softmax(self.nn_spliter(inputs))
        self.outs.append(out)
    
    def train(self, labels):
        assert(len(labels) == len(self.outs))
        labels = labels.type(torch.LongTensor)
        loss = self.nll_loss_fn(torch.cat(self.outs),labels) / len(labels)
        pred = torch.cat(self.outs)
        score = torch.argmax(pred,1) == labels
        print(loss, 
            float(sum(score).numpy()) / len(labels),
            float(sum(labels).numpy()) / len(labels))
        
        writer.add_scalar("train/spliter_loss",loss.item())
        writer.add_scalar("train/spliter_score",sum(score).item() / len(labels))
        writer.add_scalar("train/acc",sum(score).item() / len(labels))

        loss.backward()
        self.reset()
    
    def update(self):
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.nn_spliter.zero_grad()


class A3CSpliterPPOEnv(game_env.GameEnv):
    def __init__(self, *args, **kwargs):
        super(A3CSpliterPPOEnv,self).__init__(*args, **kwargs)

        random.seed(time.time())

        self.game_no = 0

        self.out_classes = 9

        self.a3c_model = ActorCritic(5, self.out_classes, 64)
        self.optimizer = optim.SGD(self.a3c_model.parameters(), lr=0.1)
        
        self.reset()

        self.gamma = 0.99
        self.tau = 1.0
        self.entropy_coef = 0.01
        self.epsilon = 0.2

        self.batch_size = 256
        self.buffer_size = 1000
        self.i = 0

        self.update_steps = 3

        self.nll_loss_fn = nn.NLLLoss()

        self.sw = SpliterWarpper(5, 128)
        
    
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

        total_r = 0.0

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
        writer.add_scalar("train/policy_loss", l.item() / self.batch_size)
        self.optimizer.step()

    
    def teacher_train(self):
        #TODO: teacher_loss * (1 - Gini coefficient)
        #TODO: entropy loss
        self.a3c_model.zero_grad()
        self.optimizer.zero_grad()
        teacher_loss = 0

        teacher_loss = torch.cat(self.raw_probs)
        teacher_loss = teacher_loss - torch.zeros(len(self.predefined_steps),self.out_classes).scatter_(1, torch.cat(self.predefined_steps).view(-1,1), 1)
        teacher_loss = torch.sum(teacher_loss ** 2)
        teacher_loss = teacher_loss / len(self.raw_probs)
        teacher_loss.backward()

        writer.add_scalar("train/teacher_loss",teacher_loss.item() / len(self.raw_probs))

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

        writer.add_scalar("train/value_loss", l.item() / self.batch_size)
        self.optimizer.step()
    
    def train(self):
        #just make it simple

        old_model = self.a3c_model.clone()

        #TODO:move it to the last
        self.teacher_train()

        for _ in range(self.update_steps):
            self.ppo_train_actor(old_model)

        for _ in range(self.update_steps):
            self.ppo_train_critic()
        
        print("reward %f"%sum(self.rewards))

        if self.game_no % 100 == 0:
            torch.save(self.a3c_model.state_dict(), "./tmp/model_%d"%self.game_no)

        writer.add_scalar("train/rewards",sum(self.rewards))
        writer.add_scalar("train/values",sum(self.values).item() / len(self.values))
        writer.add_scalar("train/entropy",sum(self.entropies).item() / len(self.entropies))

        acts = torch.cat(self.raw_log_probs)
        pd = torch.tensor(self.predefined_steps)
        self.sw.train(torch.argmax(acts,1) == pd)

    def _generator_run(self, input_):
        self.game_no = self.game_no + 1
        self.init_fn(input_)

        self.engine = simulator.Simulator(feature_name='unit_test1',
            actionspace_name='lattice1',
            canvas=self.canvas)
        
        self.reset()
        self.sw.reset()

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
            state_now = dire_state[0]
            self.states.append(state_now)
            action_out, value_out = self.a3c_model(state_now)

            self.sw.step(state_now)
            
            prob = F.softmax(action_out)
            self.raw_probs.append(prob)
            log_prob = F.log_softmax(action_out)
            self.raw_log_probs.append(log_prob)

            entropy = - (log_prob * prob).sum(1, keepdim=True)
            self.entropies.append(entropy)

            action = prob.multinomial(num_samples=1).data
            self.actions.append(action)
            #action = torch.argmax(log_prob, 1).data.view(-1,1)
            log_prob = log_prob.gather(1,Variable(action))

            self.engine.set_order("Dire", 0, (1,action))

            self.engine.loop()

            reward = dire_state[1]
            self.rewards.append(reward)
            self.values.append(value_out)
            self.log_probs.append(log_prob)
            

            yield
        

        self.train()

def test():
    v = visualizer.Visualizer(A3CSpliterPPOEnv)
    v.visualize()

def test_without_gui():
    env = A3CSpliterPPOEnv()
    gen = env.run(True)
    gen.send(None)
    while True:
        try:
            gen.send((None,))
        except StopIteration:
            gen = env.run(True)
            gen.send(None)

if __name__ == '__main__':
    if len(sys.argv) == 2:
        if sys.argv[1] == "visible":
            test()
    test_without_gui()
