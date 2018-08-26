from .game_env import GameEnv

import torch.multiprocessing as mp

def start_parallel(env_clazz, model,**kwargs):
    np = kwargs['np']
    fn = kwargs['func']
    args = kwargs['args']

    model.share_memory()

    processes = []

    for i in range(np):
        proc = mp.Process(target=fn, args=(env_clazz, model, i, args))
        proc.start()
        processes.append(proc)

    for p in processes:
        p.join()

