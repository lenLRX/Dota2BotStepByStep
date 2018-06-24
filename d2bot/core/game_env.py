"""

"""

from .. import simulator

class GameEnv:

    def __init__(self):
        self.init_fn = self._default_init_fn
        self.step_fn = self._default_step_fn
        self.stop_cond_fn = self._default_stop_cond_fn
        self.cleanup_fn = self._default_cleanup_fn

    def run(self, generator_mode=False, input_=None):
        """
            main loop of the game
        """

        if generator_mode:
            init_params = yield
            self.init_fn(init_params)
        else:
            self.init_fn(input_)

        while True:

            if generator_mode:
                yield

            if self.stop_cond_fn():
                break

        self.cleanup_fn()

    def _default_init_fn(self, arg=None):
        return None

    def _default_stop_cond_fn(self, arg=None):
        return False

    def _default_cleanup_fn(self, arg=None):
        return None

    def _default_step_fn(self, arg=None):
        return None
