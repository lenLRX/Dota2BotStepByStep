from tkinter import *

from .. import core

windows_size = 600

class Visualizer:

    def __init__(self, clazz=core.GameEnv):
        self.master = Tk()
        self.canvas = Canvas(self.master, width=windows_size, height=windows_size)
        self.canvas.pack()
        self.env = clazz(self.canvas)
        self.running = True
        self.env.cleanup_fn = lambda _: self.stop()
    
    def stop(self):
        self.running = False

    def loop(self):
        self.gen.send((None,))
        if self.running:
            self.master.after(1, self.loop)

    def visualize(self):
        self.canvas.delete("all")
        self.gen = self.env.run(True)
        self.gen.send(None)
        self.master.after(1,self.loop)
        self.master.mainloop()
