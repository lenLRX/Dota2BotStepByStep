import d2bot.visualizer as visualizer
import d2bot.core.game_env as game_env
import d2bot.simulator as simulator

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import random
import time

class LSTM_Module(nn.Module):

    def __init__(self):
        super(LSTM_Module,self).__init__()
        self.out_num = 9
        self.middle_layer_size = 1024
        self.lstm = nn.LSTM(5, self.middle_layer_size)#idle move atk
        self.hidden = nn.Linear(self.middle_layer_size,self.out_num)
 
    
    def forward(self, inputs):
        t = torch.FloatTensor(inputs)
        t = t.view(len(inputs),1,-1)
        
        lstm_outs,_ = self.lstm(t)
        lstm_out = lstm_outs[-1]
        lstm_out = F.tanh(lstm_out)
        out = F.log_softmax(self.hidden(lstm_out), dim=1)
        return out

class DefaultGameEnv(game_env.GameEnv):
    def __init__(self, *args, **kwargs):
        super(DefaultGameEnv,self).__init__(*args, **kwargs)

        random.seed(time.time())

        self.lstm_module = LSTM_Module()
        self.optimizer = optim.SGD(self.lstm_module.parameters(), lr=0.05)

        self.losses = []
        self.batch_size = 256
        self.buffer_size = 1000
        self.i = 0

        self.loss_fn = nn.NLLLoss()
    
    def train(self):
        #just make it simple
        self.lstm_module.zero_grad()
        random.shuffle(self.losses)
        if len(self.losses) > self.buffer_size:
            self.losses = self.losses[:self.buffer_size]
        buf = self.losses[:self.batch_size]
        avg_loss = (sum(buf)/self.batch_size)
        print(avg_loss.float())
        avg_loss.backward(retain_graph=True)
        self.optimizer.step()

    def _generator_run(self, input_):
        self.init_fn(input_)

        self.engine = simulator.Simulator(feature_name='unit_test1',
            actionspace_name='lattice1',
            canvas=self.canvas)        

        while self.engine.get_time() < 200:
            self.i = self.i + 1

            dire_predefine_step = self.engine.predefined_step("Dire",0)
            self.engine.loop()
            

            dire_predefine_step = self.engine.predefined_step("Dire",0)
            predefine_move = torch.LongTensor([dire_predefine_step[1]])
            #print(dire_predefine_step)
            dire_state = self.engine.get_state_tup("Dire", 0)
            is_end = dire_state[2]

            out = self.lstm_module(dire_state[0])
            loss = self.loss_fn(out, predefine_move)
            self.losses.append(loss)

            order = torch.max(out,1)[1].numpy()[0]

            if random.random() < 0.5:
                order = random.randint(0, 8)

            self.engine.set_order("Dire", 0, (1,order))

            if self.i % self.batch_size == 0:
                self.train()

            yield

            if is_end:
                break

def test():
    v = visualizer.Visualizer(DefaultGameEnv)
    v.visualize()

def test_without_gui():
    env = DefaultGameEnv()
    gen = env.run(True)
    while True:
        gen.send(None)

if __name__ == '__main__':
    test()
    #test_without_gui()