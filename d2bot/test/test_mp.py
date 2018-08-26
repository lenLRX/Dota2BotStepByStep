import torch
import torch.multiprocessing as mp

import time

class warpper():
    def __init__(self, t):
        self.t = t

def fn(t):
    #it is not safe
    '''
        PS F:\Dota2BotStepByStep> python .\d2bot\test\test_mp.py
        tensor([ 3.5505e+05])
        tensor([ 3.8177e+05])
        tensor([ 4.7613e+05])
        tensor([ 5.4674e+05])
        tensor([ 4.7676e+05])
        tensor([ 5.9204e+05])
        tensor([ 6.4995e+05])
        tensor([ 6.8313e+05])
        tensor([ 7.0968e+05])
        tensor([ 7.2394e+05])
        tensor([ 7.2394e+05])
    '''
    for _ in range(100000):
        t += 1
    print(t)

def main():
    t = torch.tensor([1])
    t.share_memory_()

    processes = []

    for _ in range(10):
        proc = mp.Process(target=fn, args=(t,))
        proc.start()
        processes.append(proc)

    for p in processes:
        p.join()
    
    print(t)

if __name__ == '__main__':
    main()