import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ActorCritic_CNN(torch.nn.Module):

    def __init__(self, chan_num, num_inputs, action_space, num_hidden):
        super(ActorCritic_CNN, self).__init__()

        self.chan_num = chan_num
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.action_space = action_space
        self.in_channels = 2

        self.kernel_size = 3

        self.conv1 = nn.Conv2d(self.in_channels, self.chan_num, self.kernel_size)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(self.chan_num, self.chan_num, self.kernel_size)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(self.chan_num, self.chan_num, self.kernel_size)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(self.chan_num, self.chan_num, self.kernel_size)
        self.relu4 = nn.ReLU()

        #self.cnn_out_size = (math.floor((self.num_inputs - (self.kernel_size - 1) - 1) + 1))
        
        #self.cnn_out_flatten_size = self.cnn_out_size ** 2

        self.num_hidden = 1024

        self.critic_linear = nn.Linear(self.num_hidden, 1)
        self.actor_linear  = nn.Linear(self.num_hidden, self.action_space)
        self.train()
    
    def forward(self, inputs):
        t = torch.FloatTensor(inputs)
        t = t.view(1, self.in_channels, self.num_inputs, self.num_inputs)
        
        _o = t
        _o = self.relu1(self.conv1(_o))
        _o = self.relu2(self.conv2(_o))
        _o = self.relu3(self.conv3(_o))
        _o = self.relu4(self.conv4(_o))
        #print(_o.size())
        cnn_layer_outs = _o.sum(1).view(1,-1)
        actor_out = self.actor_linear(F.tanh(cnn_layer_outs))

        critic_out = self.critic_linear(F.tanh(cnn_layer_outs))
        return actor_out, critic_out
    
    def clone(self):
        m = ActorCritic_CNN(self.chan_num, self.num_inputs, self.action_space, self.num_hidden)
        m.load_state_dict(self.state_dict())
        return m
