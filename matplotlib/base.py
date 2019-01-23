import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ==================================
# Basic  Usage
x = np.linspace(-3, 3, 50)
y = x ** 2 + 1
y1 = 2 * x + 1
plt.figure()
plt.plot(x, y)
plt.plot(x, y1, color='red', linewidth=1.0, linestyle='--')

# ==================================
# set axis limits && labels && ticks
plt.xlim((-1, 2))
plt.ylim((-2, 4))

plt.xlabel('I am x')
plt.ylabel('I am y')

new_xticks = np.linspace(-1, 2, 5)    # when tick range is exceed compare with limit, tick range will take effect
new_yticks = np.linspace(-2, 3, 5)
plt.xticks(new_xticks)
plt.yticks(new_yticks)
# plt.yticks([-2, -1.8, -1, 1.22, 3, ], [r'$really\ bad$', r'$bad$', r'$normal$', r'$good$', r'$really\ good \alpha$'])

# ========================
# gca = 'get current axis'
ax = plt.gca()
# splines is bounding box
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.spines['bottom'].set_position(('data', 0))    #outward, axes(position baseline)
ax.spines['left'].set_position(('data', 0))

# =========================
# set legend
line_1, = plt.plot(x, y, color='blue', lw=2.0, label='up')
line_2, = plt.plot(x, y1, color='red', linewidth=2.0, linestyle='--', label='down')
plt.legend(handles=[line_1, line_2], 
           labels=['legend set a', 'legend set b'], 
           loc='lower right',
          )    # upper/lower/center left/right best

# =========================
# annotation
x0 = 1
y0 = 2 * x0 + 1
plt.scatter(x0, y0, s=50, color='black')
plt.plot([x0, x0], [y0, 0], 'k--', lw=2)
# method 1 
plt.annotate(rf'$2x+1={y0}$', xy=(x0, y0), xycoords='data', xytext=(30, -30), textcoords='offset points',
            fontsize=16, arrowprops=dict(arrowstyle='->', connectionstyle='arc3, rad=0.2'))
# method2
plt.text(0, 3, r'$This\ is\ the\ some\ test.\ \mu\ \sigma_i\ \alpha_t$',
        fontdict={'size': 16, 'color': 'red'})

# set label visibility
for label in ax.get_yticklabels() + ax.get_xticklabels():
    label.set_fontsize(18)
    label.set_bbox(dict(
        facecolor='white',
        edgecolor='None',
        alpha=0.8
    ))
plt.show()
    
# ==================================
# scatter
plt.figure()
n = 1024
X = np.random.normal(0, 1, n)
Y = np.random.normal(0, 1, n)
T = np.arctan2(Y, X)    # color value
plt.scatter(X, Y, s=75, c=T, alpha=0.5)
plt.xlim((-1.5, 1.5))
plt.ylim((-1.5, 1.5))
plt.xticks(())
plt.yticks(())
plt.show()

# ==================================
# scatter
plt.figure()
n = 12
X = np.arange(n)
Y1 = (1 - X/float(n)) * np.random.uniform(0.5, 1.0, n)
Y2 = (1 - X/float(n)) * np.random.uniform(0.5, 1.0, n)
plt.bar(X, Y1, facecolor='#9999ff', edgecolor='white')
plt.bar(X, -Y2, facecolor='#ff9999', edgecolor='white')

for x, y in zip(X, Y1):
    # ha horizontal alignment
    plt.text(x, y, f'{y:.2f}', ha='center', va='bottom')
for x, y in zip(X, Y2):
    # ha horizontal alignment
    plt.text(x, -y, f'{y:.2f}', ha='center', va='top')
plt.show()

# ==================================
# image
img = np.array([0.3, 0.4, 0.5, 0.35, 0.45, 0.55, 0.42, 0.52, 0.62]).reshape(3, 3)
plt.imshow(img, interpolation='nearest', cmap='bone', origin='upper')    # lower
plt.colorbar(shrink=0.9)

# ==================================
# subplot
plt.figure()

plt.subplot(2, 2, 1)
plt.plot([0, 1], [0, 1])

plt.subplot(2, 2, 2)
plt.plot([0, 1], [0, 2])

plt.subplot(223)
plt.plot([0, 1], [0, 3])

plt.subplot(224)
plt.plot([0, 1], [0, 4])

plt.show()

plt.figure()

plt.subplot(2, 1, 1)
plt.plot([0, 1], [0, 1])

plt.subplot(2, 3, 4)
plt.plot([0, 1], [0, 2])

plt.subplot(235)
plt.plot([0, 1], [0, 3])

plt.subplot(236)
plt.plot([0, 1], [0, 4])

plt.show()

# method 1: subplot2grid
plt.figure()

ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=3, rowspan=1)
ax1.plot([1, 2], [1, 2])
ax1.set_title('ax1_title')

ax2 = plt.subplot2grid((3, 3), (1, 0), colspan=2, rowspan=1)
ax3 = plt.subplot2grid((3, 3), (1, 2), colspan=2, rowspan=2)
ax4 = plt.subplot2grid((3, 3), (2, 0), colspan=1, rowspan=1)
ax5 = plt.subplot2grid((3, 3), (2, 1), colspan=1, rowspan=1)

plt.show()

# method 2: gridspec
plt.figure()
gs = gridspec.GridSpec(3, 3)
ax1 = plt.subplot(gs[0, :])
ax2 = plt.subplot(gs[1, :2])
ax3 = plt.subplot(gs[1:, 2])
ax4 = plt.subplot(gs[-1, 0])
ax5 = plt.subplot(gs[-1, -2])
plt.show()

# method 3: easy to define structure
plt.figure()
f, ((ax11, ax12), (ax21, ax22)) = plt.subplots(2, 2, sharex=True, sharey=True)
ax11.scatter([1, 2], [1, 2])   
plt.tight_layout()
plt.show()

# ==================================
# plot in plot
fig = plt.figure()
x = [1, 2, 3, 4, 5, 6, 7]
y = [1, 3, 4, 2, 5, 8, 6]

left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
ax1 = fig.add_axes([left, bottom, width, height])
ax1.plot(x, y, 'r')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title("title")

left, bottom, width, height = 0.2, 0.6, 0.25, 0.25
ax2 = fig.add_axes([left, bottom, width, height])
ax2.plot(y, x, 'b')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title("title inside 1")

plt.axes([0.6, 0.2, 0.25, 0.25])
plt.plot(y[::-1], x, 'g')
plt.xlabel('x')
plt.ylabel('y')
plt.title('title inside 2')

plt.show()

# ==================================
# second axis
plt.figure()
x = np.arange(0, 10, 0.1)
y1 = 0.05 * x ** 2
y2 = -1 * y1

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(x, y1, 'g-')
ax2.plot(x, y2, 'b--')
ax1.set_xlabel('X data')
