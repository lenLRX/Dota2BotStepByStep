import d2bot.core as core

def test():
    env = core.GameEnv()
    def time_cond(env):
        print('current time %f'%env.engine.get_time())
        return env.engine.get_time() > 100
    env.stop_cond_fn = time_cond
    env.run()

if __name__ == '__main__':
    test()