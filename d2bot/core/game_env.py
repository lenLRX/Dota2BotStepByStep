"""

"""

from .. import simulator

class GameEnv:

    def __init__(self, canvas=None):
        self.canvas = canvas
        self.init_fn = self._default_init_fn
        self.step_fn = self._default_step_fn
        self.stop_cond_fn = self._default_stop_cond_fn
        self.cleanup_fn = self._default_cleanup_fn

    def run(self, generator_mode=False, input_=None):
        if generator_mode:
            return self._generator_run(input_)
        else:
            self._sync_run(input_)
    
    def _generator_run(self, input_):
        self.init_fn(input_)

        self.engine = simulator.Simulator(canvas=self.canvas)

        while True:
            self.engine.loop()

            yield

            if self.stop_cond_fn(self):
                break

        self.cleanup_fn()
    
    def _sync_run(self, input_):
        """
            main loop of the game
        """

        self.init_fn(input_)

        self.engine = simulator.Simulator(canvas=self.canvas)

        while True:
            self.engine.loop()

            if self.stop_cond_fn(self):
                break

        self.cleanup_fn()

    def _default_init_fn(self, arg=None):
        #print('_default_init_fn')
        return None

    def _default_stop_cond_fn(self, arg=None):
        #print('_default_stop_cond_fn')
        return False

    def _default_cleanup_fn(self, arg=None):
        #print('_default_cleanup_fn')
        return None

    def _default_step_fn(self, arg=None):
        #print('_default_step_fn')
        return None

def test_fn():
    game_env = GameEnv()
    def time_cond(env):
        return env.engine.get_time() < 100
    game_env.run()

if __name__ == '__main__':
    test_fn()