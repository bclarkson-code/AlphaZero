import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle

fig = plt.figure()
with open('test_game_history.txt', 'rb') as f:
    results_str = f.read()
history = pickle.loads(results_str)

im = plt.imshow(history[0], animated=True, vmin=-2, vmax=2, cmap=plt.gray())


def updatefig(i):
    im.set_array(history[i])
    return im,

ani = animation.FuncAnimation(fig, updatefig, interval=1000, blit=True)
plt.show()