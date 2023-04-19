import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib
from matplotlib.patches import Rectangle
import numpy as np

matplotlib.use('qtagg')


fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(6, 2))
majors = [-1, -0.2, 0, 0.2, 1]


def clean_ax(ax):
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_major_locator(ticker.NullLocator())
    ax.xaxis.set_major_locator(ticker.FixedLocator(majors))


clean_ax(ax1)
clean_ax(ax2)

ax1.set_xticks([-1, -0.2, 0, 0.2, 1],
               ["$-x_{max}$",
                "$-x_{min}$",
                "$x_{zero}$",
                "$x_{min}$",
                "$x_{max}$"])

ax2.set_xticks([-1, -0.2, 0, 0.2, 1],
               ["$-v_{max}$",
                "$-v_{max}$",
                "$v_{zero}$",
                "$v_{min}$",
                "$v_{max}$"])
plt.tight_layout()

plt.savefig("./numberlines.png")
plt.close()


fig, ax = plt.subplots(1, 1)

ax.set_xticks([-1, -0.2, 0, 0.2, 1],
              ["$x_{max}$",
               "$x_{max}$",
               "$x_{zero}$",
               "$-x_{min}$",
               "$-x_{max}$"])

ax.set_yticks([-1, -0.2, 0, 0.2, 1],
              ["$v_{max}$",
               "$v_{max}$",
               "$v_{zero}$",
               "$-v_{min}$",
               "$-v_{max}$"])

plt.vlines(0, -1, 1)
plt.hlines(0, -1, 1)
plt.xlim(-1, 1)
plt.ylim(-1, 1)

ax.add_patch(Rectangle(
    (-1, -0.2),
    2,
    .4,
    hatch="/",
    fill=False,
    label="do nothing"
))

ax.add_patch(Rectangle(
    (-0.2, -1),
    .4,
    2,
    hatch="/",
    fill=False,
    label="do nothing"
))

plt.legend()

plt.savefig("do_nothing.png")
plt.close()

plt.figure()

plt.xticks([0, 1], [0, 1])
plt.yticks([0, 1], [0, 1])

plt.xlabel("S")
plt.ylabel("Q")

x = np.linspace(0, 1, 10000)

plt.plot(x, x, label="linear")
plt.plot(x, x**2, label="quadratic")
plt.plot(x, x**3, label="cubic")
plt.plot(x, x**4, label="quartic")
plt.plot(x, x**0.5, label="root")
plt.plot(x, x**(1/3), label="third root")
plt.plot(x, x**0.25, label="fourth root")
plt.plot(x, np.tanh(x * 3.14), label="tanh")

plt.xlim(0, 1)
plt.ylim(0, 1)

plt.legend()

plt.savefig("quartic.png")
plt.show()
plt.close()
