import numpy as np
from matplotlib import pyplot as plt

x = np.linspace(1, 100, 1000)
y = np.log(x)
y1 = np.sin(x)
fig = plt.figure("Line plot")
legendFig = plt.figure("Legend plot", figsize=(20, 1), )
ax = fig.add_subplot(111)
line1, = ax.plot(x, y, lw=8, )
line2, = ax.plot(x, y, lw=8, )
line3, = ax.plot(x, y, lw=8, )
line4, = ax.plot(x, y, lw=8, )
line5, = ax.plot(x, y, lw=8, )
line6, = ax.plot(x, y, lw=8, )

leg = legendFig.legend([line1, line2, line3, line4, line5, line6],
                       ["SPEQ (Ours)", "SMR-SAC", "SMR-RedQ", "SAC", "RedQ", "DroQ"], loc='center', ncol=6, fontsize=30)
# change the line width for the legend
for line in leg.get_lines():
    line.set_linewidth(12.0)
legendFig.savefig('legend.pdf')
