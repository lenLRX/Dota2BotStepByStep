from tkinter import *

from .. import core

windows_size = 600

class Visualizer:

    def __init__(self):
        self.master = Tk()
        self.canvas = Canvas(self.master, width=windows_size, height=windows_size)
        self.canvas.pack()
        self.env = core.GameEnv(self.canvas)
        self.running = True
        self.env.cleanup_fn = lambda _: self.stop()
    
    def stop(self):
        self.running = False

    def loop(self):
        self.gen.send((None,))
        if self.running:
            self.master.after(1, self.loop)

    def visualize(self):
        self.gen = self.env.run(True)
        self.gen.send(None)
        self.canvas.delete("all")
        self.master.after(1,self.loop)
        self.master.mainloop()
