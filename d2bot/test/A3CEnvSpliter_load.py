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


class A3CSpliterEnv(game_env.GameEnv):
    def __init__(self, *args, **kwargs):
        super(A3CSpliterEnv,self).__init__(*args, **kwargs)

        random.seed(time.time())

        self.game_no = 0

        self.out_classes = 9

        self.a3c_model = ActorCritic(5, self.out_classes, 64)
        self.a3c_model.load_state_dict(torch.load('F:\\Dota2BotStepByStep\\tmp\\model_3453_6240'))
        self.optimizer = optim.SGD(self.a3c_model.parameters(), lr=0.01)
        
        self.reset()

        self.gamma = 0.99
        self.tau = 1.0
        self.entropy_coef = 0.01

        self.batch_size = 256
        self.buffer_size = 1000
        self.i = 0

        self.nll_loss_fn = nn.NLLLoss()

        self.sw = SpliterWarpper(5, 128)
        
    
    def reset(self):
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
        self.sw.update()
    
    def train(self):
        #just make it simple
        
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

            #teacher_loss = teacher_loss + self.nll_loss_fn(self.raw_log_probs[i], self.predefined_steps[i])
        
        teacher_loss = torch.cat(self.raw_probs)
        teacher_loss = teacher_loss - torch.zeros(len(self.predefined_steps),self.out_classes).scatter_(1, torch.cat(self.predefined_steps).view(-1,1), 1)
        teacher_loss = torch.sum(teacher_loss ** 2)

        total_loss = (policy_loss + 0.5 * value_loss + teacher_loss)
        total_loss = total_loss / len(self.raw_probs)
        #print(total_loss)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm(self.a3c_model.parameters(), 10)

        print("reward %f"%sum(self.rewards))

        writer.add_scalar("train/loss",total_loss.item())
        writer.add_scalar("train/policy_loss",policy_loss.item() / len(self.raw_probs))
        writer.add_scalar("train/policy_loss",value_loss.item() / len(self.raw_probs))
        writer.add_scalar("train/teacher_loss",teacher_loss.item() / len(self.raw_probs))
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

        while self.engine.get_time() < 300:
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

            self.sw.step(dire_state[0])
            
            prob = F.softmax(action_out)
            self.raw_probs.append(prob)
            log_prob = F.log_softmax(action_out)
            self.raw_log_probs.append(log_prob)
            

            entropy = - (log_prob * prob).sum(1, keepdim=True)
            self.entropies.append(entropy)

            action = prob.multinomial(num_samples=1).data
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
        self.update()

def test():
    v = visualizer.Visualizer(A3CSpliterEnv)
    v.visualize()

def test_without_gui():
    env = A3CSpliterEnv()
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
