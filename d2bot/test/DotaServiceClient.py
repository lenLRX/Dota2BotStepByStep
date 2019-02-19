from collections import Counter
from time import time
import asyncio
import math
import os
import math
import random
import shutil

import torch.multiprocessing as mp

from grpclib.client import Channel
from google.protobuf.json_format import MessageToDict
from tensorboardX import SummaryWriter

from pympler.tracker import SummaryTracker

from dotaservice.protos.DotaService_grpc import DotaServiceStub
from dotaservice.protos.dota_gcmessages_common_bot_script_pb2 import CMsgBotWorldState
from dotaservice.protos.DotaService_pb2 import Action
from dotaservice.protos.DotaService_pb2 import Empty
from dotaservice.protos.DotaService_pb2 import Config
from dotaservice.protos.DotaService_pb2 import HostMode

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorNCritic(torch.nn.Module):

    def __init__(self, num_inputs, action_space, num_hidden):
        super(ActorNCritic, self).__init__()

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
        m = ActorNCritic(self.num_inputs, self.action_space, self.num_hidden)
        m.load_state_dict(self.state_dict())
        return m

lr = 1e-3
num_hidden = 16

MODE_NORMAL=0
MODE_AUTO=1


config = Config(
        ticks_per_observation=10,
        host_timescale=5,
        host_mode=HostMode.Value('DEDICATED'),
    )

def get_hero_unit(state, id=0):
    for unit in state.units:
        if unit.unit_type == CMsgBotWorldState.UnitType.Value('HERO') and unit.player_id == id:
            return unit
    raise ValueError("hero {} not found in state:\n{}".format(id, state))

def get_moving_vec(idx):
    if (idx == 0):
        return (0,0)
    rad = idx * math.pi / 4 - math.pi
    x = math.cos(rad)
    y = math.sin(rad)
    return x,y


xp_to_reach_level = {
    1: 0,
    2: 230,
    3: 600,
    4: 1080,
    5: 1680,
    6: 2300,
    7: 2940,
    8: 3600,
    9: 4280,
    10: 5080,
    11: 5900,
    12: 6740,
    13: 7640,
    14: 8865,
    15: 10115,
    16: 11390,
    17: 12690,
    18: 14015,
    19: 15415,
    20: 16905,
    21: 18405,
    22: 20155,
    23: 22155,
    24: 24405,
    25: 26905
}

action_none = CMsgBotWorldState.Action()
action_none.actionType = CMsgBotWorldState.Action.Type.Value(
        'DOTA_UNIT_ORDER_NONE')


def get_total_xp(level, xp_needed_to_level):
    if level == 25:
        return xp_to_reach_level[level]
    xp_required_for_next_level = xp_to_reach_level[level + 1] - xp_to_reach_level[level]
    missing_xp_for_next_level = (xp_required_for_next_level - xp_needed_to_level)
    return xp_to_reach_level[level] + missing_xp_for_next_level


def get_reward(prev_state, state):
    """Get the reward."""
    unit_init = get_hero_unit(prev_state)
    unit = get_hero_unit(state)

    reward = {'xp': 0, 'hp': 0, 'death': 0, 'dist': 0, 'lh': 0}


    xp_init = get_total_xp(level=unit_init.level, xp_needed_to_level=unit_init.xp_needed_to_level)
    xp = get_total_xp(level=unit.level, xp_needed_to_level=unit.xp_needed_to_level)

    reward['xp'] = (xp - xp_init) * 0.02  # One creep will result in 0.114 reward

    if unit_init.is_alive and unit.is_alive:
        hp_init = unit_init.health / unit_init.health_max
        hp = unit.health / unit.health_max
        reward['hp'] = (hp - hp_init) * 1.0
    if unit_init.is_alive and not unit.is_alive:
        reward['death'] = - 0.5  # Death should be a big penalty

    # Last-hit reward
    lh = unit.last_hits - unit_init.last_hits
    reward['lh'] = lh * 0.5

    # Help him get to mid, for minor speed boost

    dt = state.dota_time - prev_state.dota_time
    dist_mid_init = math.sqrt(unit_init.location.x**2 + unit_init.location.y**2)
    dist_mid = math.sqrt(unit.location.x**2 + unit.location.y**2)
    reward_dist = (dist_mid_init - dist_mid) /\
    ((unit_init.base_movement_speed + unit.base_movement_speed) * dt) * 0.1


    #print(dt, reward['dist'])

    total_reward = sum(reward.values())

    return total_reward, reward_dist

def calc_reward(state, prev_state):
    return get_reward(prev_state, state)

config = Config(
    ticks_per_observation=10,
    host_timescale=10,
    host_mode=HostMode.Value('DEDICATED'),
)

class DotaServiceEnv(object):

    def __init__(self, rank, config, host, port):
        loop = asyncio.get_event_loop()
        channel = Channel(host, port, loop=loop)
        self.env = DotaServiceStub(channel) # place holder
        self.config = config

        self.gamma = 0.7
        self.gamma_dist = 0.
        self.tau = 1.0
        self.entropy_coef = 0.01
        self.epsilon = 0.2

        self.buffer_size = 1000

        self.out_classes = 9

        self.update_steps = 3

        self.rank = rank
        self.writer = SummaryWriter(comment='_%d'%self.rank)

        self.a3c_model = ActorCritic(8, self.out_classes, num_hidden)
        self.optimizer = optim.SGD(self.a3c_model.parameters(), lr=lr)

        self.MODE = MODE_NORMAL
    
    def reset(self):
        self.states = []
        self.actions = []
        self.entropies = []
        self.values = []
        self.rewards = []
        self.log_probs = []
        self.raw_log_probs = []
        self.raw_probs = []
    
    def set_model(self, model):
        self.a3c_model = model
        self.optimizer = optim.SGD(self.a3c_model.parameters(), lr=lr)
    
    def get_model(self):
        return self.a3c_model
    
    @staticmethod
    def unit2input(unit):
        loc = [unit.location.x / 7000., unit.location.y / 7000.]
        return loc


    @staticmethod
    def print_unit_list(state):
        for unit in state.units:
            if unit.unit_type == CMsgBotWorldState.UnitType.Value('HERO') or unit.unit_type == CMsgBotWorldState.UnitType.Value('LANE_CREEP'):
                print('[debug] unit list: {}'.format(unit))
    
    @staticmethod
    def get_unit_list(state):
        r = []
        for unit in state.units:
            if unit.unit_type == CMsgBotWorldState.UnitType.Value('LANE_CREEP')\
                or unit.unit_type == CMsgBotWorldState.UnitType.Value('TOWER')\
                or unit.unit_type == CMsgBotWorldState.UnitType.Value('BUILDING')\
                or unit.unit_type == CMsgBotWorldState.UnitType.Value('BARRACKS')\
                or unit.unit_type == CMsgBotWorldState.UnitType.Value('FORT')\
                or unit.unit_type == CMsgBotWorldState.UnitType.Value('HERO'):
                r.append(unit)
        return r
    
    
    
    def ppo_train_actor(self, old_model):
        self.a3c_model.zero_grad()
        self.optimizer.zero_grad()

        l = 0.0
        R = torch.zeros(1, 1)

        reduced_r = []
        for i in reversed(range(len(self.rewards))):
            R = self.gamma * R + self.rewards[i][0]
            reduced_r.append(R)
        
        reduced_r = list(reversed(reduced_r))

        for i in range(len(self.rewards)):
            reduced_r[i] += self.rewards[i][1]

        idxs = list(range(len(self.rewards)))
        random.shuffle(idxs)

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
        
        l = l / len(idxs)
        #print("train/policy_loss", l.item())
        l.backward(retain_graph=True)
        self.optimizer.step()
        return l.item()
    
    def ppo_train_critic(self):
        
        self.a3c_model.zero_grad()
        self.optimizer.zero_grad()

        R = torch.zeros(1, 1)
        l = 0.0

        reduced_r = []
        for i in reversed(range(len(self.rewards))):
            R = self.gamma * R + self.rewards[i][0]
            reduced_r.append(R)
        
        reduced_r = list(reversed(reduced_r))

        for i in range(len(self.rewards)):
            reduced_r[i] += self.rewards[i][1]

        idxs = list(range(len(self.rewards)))
        random.shuffle(idxs)

        for i in idxs:
            adv = reduced_r[i] - self.a3c_model(self.states[i])[1]
            l = l + adv ** 2
        
        l = l / len(idxs)
        l.backward(retain_graph=True)

        #print("train/value_loss", l.item())
        self.optimizer.step()
        return l.item()
    
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

        self.optimizer.step()
    
    def train(self):
        #just make it simple
        self.a3c_model.train()

        old_model = self.a3c_model.clone()

        #self.teacher_train()
        l = 0

        for _ in range(self.update_steps):
            l = self.ppo_train_actor(old_model)
        
        print("train/policy_loss", l)
        self.writer.add_scalar("train/policy_loss", l / len(self.states))

        if self.MODE == MODE_NORMAL:
            for _ in range(self.update_steps):
                l= self.ppo_train_critic()
        
            print("train/value_loss", l)
            self.writer.add_scalar("train/value_loss", l / len(self.states))

            total_reward = np.sum(self.rewards)
        
            print("MODE {} total reward {} avg reward {}".format(self.MODE, total_reward, total_reward / len(self.rewards)))
            self.writer.add_scalar("train/reward", total_reward)
    
    def default_action(self, state):
        hero = state[0]
        for unit in state:
            if unit[4] > 0:
                if math.hypot(hero[0]-unit[0],hero[1]-unit[1]) < 600. / 7000.:
                    return 1
        
        if (hero[0] < 0 and hero[1] < 0):
            return 5
        else:
            return 1

        return 5
    
    async def run_a_game(self):
        #tracker = SummaryTracker()
        #print('using model id {}'.format(id(self.a3c_model)))
        self.reset()
        self.MODE = np.random.randint(2)
        #print('Mode:{}'.format(self.MODE))
        # start a game
        while True:
            try:
                t_start = time()
                await asyncio.wait_for(self.env.clear(Empty()), timeout=60)
                state = await asyncio.wait_for(self.env.reset(config), timeout=60)
                #print('start time {}'.format(time() - t_start))
                break
            except Exception as e:
                print('Exception on env.reset: {}'.format(e))
                return
        
        while True:
            # fetch hero
            #tick_start = time()
            state = state.world_state
            if state.dota_time > 130:
                break
            prev_state = state
            # print(state.dota_time)
            hero = get_hero_unit(state)

            all_units = DotaServiceEnv.get_unit_list(state)

            input_state = []
            
            hero_loc = (hero.location.x, hero.location.y)

            for unit in all_units:
                loc = [unit.location.x, unit.location.y]
                d = math.sqrt((unit.location.x - hero_loc[0])**2 + (unit.location.y - hero_loc[1])**2)
                

                if d >= 1200:
                    continue

                if unit is not hero:
                    loc = [hero_loc[0] - unit.location.x, hero_loc[1] - unit.location.y]

                loc = [loc[0] / 7000., loc[1] / 7000.]

                unit_state = list(loc)
                type_tup = [0] * 6
                if unit.unit_type == CMsgBotWorldState.UnitType.Value('HERO') and unit.player_id == 0:
                    type_tup[0] = 1
                elif unit.unit_type == CMsgBotWorldState.UnitType.Value('LANE_CREEP') and unit.team_id == hero.team_id:
                    type_tup[1] = 1
                elif unit.unit_type == CMsgBotWorldState.UnitType.Value('LANE_CREEP') and unit.team_id != hero.team_id:
                    type_tup[2] = 1
                elif unit.unit_type == CMsgBotWorldState.UnitType.Value('TOWER') and unit.team_id == hero.team_id:
                    type_tup[3] = 1
                elif unit.unit_type == CMsgBotWorldState.UnitType.Value('TOWER') and unit.team_id != hero.team_id:
                    type_tup[4] = 1
                else:
                    type_tup[5] = 1
                unit_state.extend(type_tup)

                unit_state = np.array(unit_state)
                unit_state = torch.from_numpy(unit_state).float()
                if unit is hero:
                    hero_state = unit_state
                else:
                    input_state.append(unit_state)
            
            input_state_wo_hero = sorted(input_state, key=lambda x:math.hypot(x[0],x[1]))
            input_state = [hero_state]
            input_state.extend(input_state_wo_hero)
            #print(input_state)

            raw_input_state = input_state

            input_state = torch.stack(input_state)
            
            self.states.append(input_state)

            action_out, value_out = self.a3c_model(input_state)
            #print(action_out , value_out, input_state)

            prob = F.softmax(action_out)
            self.raw_probs.append(prob)
            log_prob = F.log_softmax(action_out)
            self.raw_log_probs.append(log_prob)

            self.values.append(value_out)
            self.log_probs.append(log_prob)

            entropy = - (log_prob * prob).sum(1, keepdim=True)
            self.entropies.append(entropy)

            if self.MODE == MODE_NORMAL:
                action = prob.multinomial(num_samples=1).data
                #action = torch.argmax(log_prob, 1).data.view(-1,1)
            elif self.MODE == MODE_AUTO:
                action = self.default_action(raw_input_state)
            self.actions.append(action)

            action_pb = CMsgBotWorldState.Action()
            action_pb.actionType = CMsgBotWorldState.Action.Type.Value(
                'DOTA_UNIT_ORDER_MOVE_TO_POSITION')
            mx, my = get_moving_vec(action)
            scale = 500
            hloc = hero.location
            m = CMsgBotWorldState.Action.MoveToLocation()
            m.location.x = mx * scale + hloc.x
            m.location.y = my * scale + hloc.y
            m.location.z = 0
            action_pb.moveToLocation.CopyFrom(m)
            action_pb.actionDelay=0
            # print(action, action_pb)
            # print('tick cost {}'.format(time() - tick_start))

            try:
                state = await asyncio.wait_for(self.env.step(Action(action=action_pb)), timeout=11)
                reward = calc_reward(state.world_state, prev_state)
                self.rewards.append(reward)
                
            except Exception as e:
                print('Exception on env.step: {}'.format(repr(e)))
                raise
                break

        self.train()
        #await asyncio.get_event_loop().run_in_executor(None, self.train)
        #tracker.print_diff()


def main():
    tmp_dir = '/root/Dota2BotStepByStep/runs'
    shutil.rmtree(tmp_dir)
    host = '172.18.5.31'
    base_port = 13337

    concurrent_num = 12

    eps = [
            {'host':host, 'port':i} for i in range(base_port, base_port + concurrent_num)
        ]
    '''
    host = '172.18.5.30'

    eps.extend([
            {'host':host, 'port':i} for i in range(base_port, base_port + concurrent_num)
        ])
    '''
    
    thread_num = os.cpu_count()
    if thread_num > len(eps):
        thread_num = len(eps)

    shared_model = ActorCritic(8, 9, num_hidden)
    shared_model.share_memory()
    threads = [mp.Process(target=worker_thread,args=(i, eps[i::thread_num],shared_model)) for i in range(thread_num)]
    for t in threads:
        t.start()
    
    for t in threads:
        t.join()

def worker_thread(rank, eps, shared_model):
    loop=asyncio.get_event_loop()
    actors = [
            DotaServiceEnv(rank=rank,config=config,
                **ep) for ep in eps
        ]
    print('thread start {} actors end points {}'.format(len(actors), eps))
    for a in actors:
        a.set_model(shared_model)
    
    np.random.seed()
    loop.run_until_complete(working_loop(actors))

async def working_loop(actors):
    while True:
        actor_output = await asyncio.gather(*[a.run_a_game() for a in actors])

if __name__ == '__main__':
    main()
