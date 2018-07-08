import d2bot.visualizer as visualizer
import d2bot.core.game_env as game_env
import d2bot.simulator as simulator

class DefaultGameEnv(game_env.GameEnv):
    def _generator_run(self, input_):
        self.init_fn(input_)

        self.engine = simulator.Simulator(canvas=self.canvas)

        

        while True:
            dire_predefine_step = self.engine.predefined_step("Dire",0)
            self.engine.loop()
            self.engine.set_order("Dire",0,dire_predefine_step)

            yield

            if self.stop_cond_fn(self):
                break

        self.cleanup_fn()

def test():
    v = visualizer.Visualizer(DefaultGameEnv)
    v.visualize()

if __name__ == '__main__':
    test()