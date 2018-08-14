import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCritic(torch.nn.Module):

    def __init__(self, num_inputs, action_space, num_hidden):
        super(ActorCritic, self).__init__()

        self.num_hidden = num_hidden
        self.num_inputs = num_inputs
        self.action_space = action_space

        self.a_lstm_layer = nn.LSTM(num_inputs, self.num_hidden)
        self.c_lstm_layer = nn.LSTM(num_inputs, self.num_hidden)

        self.critic_linear = nn.Linear(self.num_hidden, 1)
        self.actor_linear  = nn.Linear(self.num_hidden, self.action_space)
        self.train()
    
    def forward(self, inputs):
        t = torch.FloatTensor(inputs)
        t = t.view(len(inputs),1,-1)
        
        a_lstm_outs,_ = self.a_lstm_layer(t)
        a_lstm_out = a_lstm_outs[-1]
        a_lstm_out = F.tanh(a_lstm_out)
        actor_out = self.actor_linear(a_lstm_out)

        c_lstm_outs,_ = self.c_lstm_layer(t)
        c_lstm_out = c_lstm_outs[-1]
        c_lstm_out = F.tanh(c_lstm_out)
        critic_out = self.critic_linear(c_lstm_out)
        return actor_out, critic_out
    
    def clone(self):
        m = ActorCritic(self.num_inputs, self.action_space, self.num_hidden)
        m.load_state_dict(self.state_dict())
        return m
