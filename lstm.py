import numpy as np
from matplotlib import pyplot as plt

# Define sigmoid function
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# Define derivate of sigmoid function
def der_sigmoid(x):
    return x * (1.0 - x)

def tanh(x):
	return np.tanh(x)

def der_tanh(x):
	return 1.0 - np.tanh(x) ** 2

path = 'hello.txt'
hidden_size = 100
seq_length = 25
learning_rate = 1e-1
sample_interval = 500
sample_length = 200
weightScale = 1e-2

data = open(path , 'r').read()
chars = list(set(data))
data_size = len(data)
vocab_size = len(chars)

print 'input has %d chars, %d of which are unique.' % (data_size, vocab_size)

char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

Wz = np.random.randn(hidden_size, vocab_size) * weightScale
Wf = np.random.randn(hidden_size, vocab_size) * weightScale
Wi = np.random.randn(hidden_size, vocab_size) * weightScale
Wo = np.random.randn(hidden_size, vocab_size) * weightScale

Rz = np.random.randn(hidden_size, hidden_size) * weightScale
Rf = np.random.randn(hidden_size, hidden_size) * weightScale
Ri = np.random.randn(hidden_size, hidden_size) * weightScale
Ro = np.random.randn(hidden_size, hidden_size) * weightScale

bz = np.zeros((hidden_size, 1))
bf = np.zeros((hidden_size, 1))
bi = np.zeros((hidden_size, 1))
bo = np.zeros((hidden_size, 1))

Wy = np.random.randn(vocab_size, hidden_size) * weightScale
by = np.zeros((vocab_size, 1))


def lossFun(inputs, targets, hprev, cprev):
	z, z_, i, i_, f, f_, o, o_ = {}, {}, {}, {}, {}, {}, {}, {}
	x, c, h, y, p = {}, {}, {}, {}, {}
	
	h[-1] = np.copy(hprev)
	c[-1] = np.copy(cprev)
	loss = 0

	# forward pass
	for t in xrange(len(inputs)):
		# Input layer
		x[t] = np.zeros((vocab_size, 1))
		x[t][inputs[t]] = 1
		
		# LSTM layer 1
		z_[t] = np.dot(Wz, x[t]) + np.dot(Rz, h[t-1]) + bz
		z[t] = tanh(z_[t])

		i_[t] = np.dot(Wi, x[t]) + np.dot(Ri, h[t-1]) + bi
		i[t] = sigmoid(i_[t])
		
		f_[t] = np.dot(Wf, x[t]) + np.dot(Rf, h[t-1]) + bf
		f[t] = sigmoid(f_[t])

		c[t] = i[t] * z[t] + f[t] * c[t-1]

		o_[t] = np.dot(Wo, x[t]) + np.dot(Ro, h[t-1]) + bo
		o[t] = sigmoid(o_[t])

		h[t] = tanh(c[t]) * o[t]

		# output
		y[t] = np.dot(Wy, h[t]) + by

		# normalize
		p[t] = np.exp(y[t]) / np.sum(np.exp(y[t]))

		# loss
		loss += -np.log(p[t][targets[t], 0])

	dWz = np.zeros_like(Wz)
	dWf = np.zeros_like(Wf)
	dWi = np.zeros_like(Wi)
	dWo = np.zeros_like(Wo)

	dRz = np.zeros_like(Rz)
	dRf = np.zeros_like(Rf)
	dRi = np.zeros_like(Ri)
	dRo = np.zeros_like(Ro)

	dbz = np.zeros_like(bz)
	dbf = np.zeros_like(bf)
	dbi = np.zeros_like(bi)
	dbo = np.zeros_like(bo)

	dWy = np.zeros_like(Wy)
	dby = np.zeros_like(by)

	dz, di, df, do, dc = {}, {}, {}, {}, {}
	dy, dh = {}, {}
	dz[len(inputs)] = np.zeros_like(z[0])
	di[len(inputs)] = np.zeros_like(i[0])
	df[len(inputs)] = np.zeros_like(f[0])
	do[len(inputs)] = np.zeros_like(o[0])
	dc[len(inputs)] = np.zeros_like(c[0])
	f[len(inputs)] = np.zeros_like(f[0])
	
	# backprop
	for t in reversed(xrange(len(inputs))):
		dy[t] = np.copy(p[t])
		dy[t][targets[t]] -= 1

		dWy += np.outer(dy[t], h[t])
		dby += dy[t]
		
		dh[t] = np.dot(Wy.T, dy[t])
		dh[t] += np.dot(Rz.T, dz[t+1])
		dh[t] += np.dot(Ri.T, di[t+1])
		dh[t] += np.dot(Rf.T, df[t+1])
		dh[t] += np.dot(Ro.T, do[t+1])

		do[t] = dh[t] * tanh(c[t]) * der_sigmoid(o_[t])

		dc[t] = dh[t] * o[t] * der_tanh(c[t])
		dc[t] += dc[t+1] * f[t+1]

		df[t] = dc[t] * c[t-1] * der_sigmoid(f_[t])

		di[t] = dc[t] * z[t] * der_sigmoid(i_[t])

		dz[t] = dc[t] * i[t] * der_tanh(z_[t])

		dWz += np.outer(dz[t], x[t])
		dWi += np.outer(di[t], x[t])
		dWf += np.outer(df[t], x[t])
		dWo += np.outer(do[t], x[t])

		dRz += np.outer(dz[t+1], h[t])
		dRi += np.outer(di[t+1], h[t])
		dRf += np.outer(df[t+1], h[t])
		dRo += np.outer(do[t+1], h[t])

		dbz += dz[t]
		dbi += di[t]
		dbf += df[t]
		dbo += do[t]

	#for dparam in [dWz, dWi, dWf, dWo, dRz, dRi, dRf, dRo, dbz, dbi, dbf, dbo, dWy, dby]:
	#	np.clip(dparam, -5, 5, out=dparam)

	return loss, dWz, dWi, dWf, dWo, dRz, dRi, dRf, dRo, dbz, dbi, dbf, dbo, dWy, dby, h[len(inputs)-1], c[len(inputs)-1]



def sample(h, c, seed_ix, n):
	x = np.zeros((vocab_size, 1))
	x[seed_ix] = 1
	ixes = []
	for t in xrange(n):
		z_ = np.dot(Wz, x) + np.dot(Rz, h) + bz
		z = tanh(z_)

		i_ = np.dot(Wi, x) + np.dot(Ri, h) + bi
		i = sigmoid(i_)

		f_ = np.dot(Wf, x) + np.dot(Rf, h) + bf
		f = sigmoid(f_)

		c = i * z + f * c

		o_ = np.dot(Wo, x) + np.dot(Ro, h) + bo
		o = sigmoid(o_)

		h = tanh(c) * o

		y = np.dot(Wy, h) + by

		p = np.exp(y) / np.sum(np.exp(y))

		ix = np.random.choice(range(vocab_size), p=p.ravel())
		x = np.zeros((vocab_size, 1))
		x[ix] = 1
		ixes.append(ix)
	return ixes


mWz = np.zeros_like(Wz)
mWf = np.zeros_like(Wf)
mWi = np.zeros_like(Wi)
mWo = np.zeros_like(Wo)

mRz = np.zeros_like(Rz)
mRf = np.zeros_like(Rf)
mRi = np.zeros_like(Ri)
mRo = np.zeros_like(Ro)

mbz = np.zeros_like(bz)
mbf = np.zeros_like(bf)
mbi = np.zeros_like(bi)
mbo = np.zeros_like(bo)

mWy = np.zeros_like(Wy)
mby = np.zeros_like(by)

smooth_loss = -np.log(1.0 / vocab_size) * seq_length
n, p = 0, 0




lossData = []

try:
	while True:
		if p + seq_length + 1 >= len(data) or n == 0:
			hprev = np.zeros((hidden_size, 1))
			cprev = np.zeros((hidden_size, 1))
			p = 0

		inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
		targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

		if n % sample_interval == 0:
			sample_ix = sample(hprev, cprev, inputs[0], sample_length)
			txt = ''.join(ix_to_char[ix] for ix in sample_ix)
			print '----\n %s \n----' % (txt, )

		loss, dWz, dWi, dWf, dWo, dRz, dRi, dRf, dRo, dbz, dbi, dbf, dbo, dWy, dby, hprev, cprev = lossFun(inputs, targets, hprev, cprev)
		smooth_loss = smooth_loss * 0.999 + loss * 0.001
		lossData.append(smooth_loss)

		if n % sample_interval == 0:
			plt.clf()
			plt.plot(lossData)
			plt.pause(0.1)
			print 'iteration: %d, loss: %.2f' % (n, smooth_loss)

		for param, dparam, mem in zip(
			[Wz, Wf, Wi, Wo, Rz, Rf, Ri, Ro, bz, bf, bi, bo, Wy, by],
			[dWz, dWf, dWi, dWo, dRz, dRf, dRi, dRo, dbz, dbf, dbi, dbo, dWy, dby],
			[mWz, mWf, mWi, mWo, mRz, mRf, mRi, mRo, mbz, mbf, mbi, mbo, mWy, mby]):

			mem += dparam * dparam
			param += -learning_rate * dparam / np.sqrt(mem + 1e-8)

		p += seq_length
		n += 1

except KeyboardInterrupt:
	exit