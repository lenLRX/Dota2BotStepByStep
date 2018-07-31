import d2bot.torch.a3c.A3CEnv as A3CEnv

import d2bot.visualizer as visualizer

def test():
    v = visualizer.Visualizer(A3CEnv)
    v.visualize()

def test_without_gui():
    env = A3CEnv()
    gen = env.run(True)
    while True:
        gen.send(None)

if __name__ == '__main__':
    test()
    #test_without_gui()