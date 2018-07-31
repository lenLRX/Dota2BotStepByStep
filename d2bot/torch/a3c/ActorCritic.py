import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCritic(torch.nn.Module):

    def __init__(self, num_inputs, action_space, num_hidden):
        super(ActorCritic, self).__init__()

        self.num_hidden = num_hidden
        self.num_inputs = num_inputs
        self.action_space = action_space

        self.lstm_layer = nn.LSTM(num_inputs, self.num_hidden)

        self.critic_linear = nn.Linear(self.num_hidden, 1)
        self.actor_linear  = nn.Linear(self.num_hidden, self.action_space)
        self.train()
    
    def forward(self, inputs):
        t = torch.FloatTensor(inputs)
        t = t.view(len(inputs),1,-1)
        
        lstm_outs,_ = self.lstm_layer(t)
        lstm_out = lstm_outs[-1]
        lstm_out = F.tanh(lstm_out)
        actor_out = self.actor_linear(lstm_out)
        critic_out = self.critic_linear(lstm_out)
        return actor_out, critic_out
