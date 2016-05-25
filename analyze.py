import numpy as np 						# for computing
import argparse 						# for argument parsing from command line
import pickle 							# for data dumping as checkpoint files
import matplotlib.pyplot as plt 	# for plotting
import matplotlib.patches as mpatches

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='training file', default="hello.txt")
parser.add_argument('-c', '--checkpoint', help='checkpoint file', default='')
args = parser.parse_args()

# Define sigmoid function
def sigmoid(x):
	return 1.0 / (1.0 + np.exp(-x))

# Define derivate of sigmoid function
def der_sigmoid(x):
	return sigmoid(x) * (1.0 - sigmoid(x))

def tanh(x):
	return np.tanh(x)

def der_tanh(x):
	return 1.0 - np.tanh(x) ** 2

def loadWeights():
	checkpoint = open(args.checkpoint, 'rb')
	weights = pickle.load(checkpoint)
	checkpoint.close()
	global Wz, Wi, Wf, Wo, Rz, Ri, Rf, Ro, bz, bi, bf, bo, Wy, by, lstm_size
	Wz = weights['Wz']
	Wi = weights['Wi']
	Wf = weights['Wf']
	Wo = weights['Wo']
	Rz = weights['Rz']
	Ri = weights['Ri']
	Rf = weights['Rf']
	Ro = weights['Ro']
	bz = weights['bz']
	bi = weights['bi']
	bf = weights['bf']
	bo = weights['bo']
	Wy = weights['Wy']
	by = weights['by']
	lstm_size = weights['lstm_size']

def onestep(run):
	global h, c, x, t
	global z, i, f, o, y
	global z_, i_, f_, o_
	global x_next, t
	global ones

	x = x_next
	
	z_ = np.dot(Wz, x) + np.dot(Rz, h[t-1]) + bz
	z.append(tanh(z_))

	i_ = np.dot(Wi, x) + np.dot(Ri, h[t-1]) + bi
	i.append(sigmoid(i_))

	f_ = np.dot(Wf, x) + np.dot(Rf, h[t-1]) + bf
	f.append(sigmoid(f_))

	c.append(i[t] * z[t] + f[t] * c[t-1])

	o_ = np.dot(Wo, x) + np.dot(Ro, h[t-1]) + bo
	o.append(sigmoid(o_))

	h.append(tanh(c[t]) * o[t])

	y.append(np.dot(Wy, h[t]) + by)

	p = np.exp(y[t]) / np.sum(np.exp(y[t]))

	ix = np.random.choice(range(vocab_size), p=p.ravel())

	ch = ix_to_char[ix]
	if ch == '1':
		ones[run].append(t)
	#print "character: \'%s\'" % (ch)
	
	x_next = np.zeros((vocab_size, 1))
	x_next[ix] = 1

	t += 1

	return ch, p[ix]

def init():
	global vocab_size, ix_to_char

	
	chars = list(set(input_text))
	vocab_size = len(chars)
	ix_to_char = { ix:ch for ix,ch in enumerate(chars) }
	char_to_ix = { ch:i for i,ch in enumerate(chars) }

	#args.checkpoint = 'filename'
	loadWeights()

	global z, i, f, o, y
	z, i, f, o, y = [], [], [], [], []
	for v in (z, i, f, o, y):
		v.append(np.zeros((lstm_size, 1)))

	global h, c
	h, c = [], []
	h.append(np.zeros((lstm_size, 1)))
	c.append(np.zeros((lstm_size, 1)))

	global t, x, x_next
	t = 1
	x = np.zeros((vocab_size, 1))
	x[char_to_ix['1']] = 1
	x_next = x



main_path = ('./binary/')
for b in [1, 3, 5, 10, 20]:
	path = main_path + str(b) + '/'
	input_file = path + str(b) + '_zero.txt'
	input_text = open(input_file, 'r').read().decode('utf-8')

	fig = plt.figure()
	fig.suptitle('0'*b+'1', fontsize=14, fontweight='bold')

	ones = {}
	for run in [1, 2, 3]:
		ax = fig.add_subplot(3, 1, run)

		args.checkpoint = path + str(b) + '_zero_' + str(run) + '.checkpoint'

		init()

		ones[run] = []

		for k in range(2 + 2 * (b+1)):
			onestep(run)

		for hu in range(lstm_size):
			ix = 1
			for v in (z, i, f, o, c, h):
				colors = [int(u) for u in bin(ix)[2:]]
				while len(colors) < 3:
					colors.insert(0, 0)
				ix += 1
				re = [(v[u][hu][0]) for u in range(1, t)]
				#plt.subplot(3, 1, run)
				ax.plot(range(1, t), re, color = colors, lw = 6)
				ax.set_xlim((1, t-1))
				ax.set_ylim((-1.05, 1.05))
				ax.axvline(x = 1, ymin=-1, ymax=1, color = [0, 0, 0])
				for one in ones[run]:
					ax.axvline(x = one, ymin=-1, ymax=1, color = [0, 0, 0])
				ax.set_xticks(range(1, t))

	my_handles = []
	ix = 1
	for v in ('z', 'i', 'f', 'o', 'c', 'h'):
		colors = [int(u) for u in bin(ix)[2:]]
		while len(colors) < 3:
			colors.insert(0, 0)
		ix += 1
		my_handles.append(mpatches.Patch(color=colors, label=v))
	plt.legend(loc='lower right', bbox_to_anchor=(1.09, 1.40), handles=my_handles)
	mng = plt.get_current_fig_manager()
	mng.full_screen_toggle()
	plt.show()
	fig.savefig(main_path+str(b)+'.png')