#-*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='training file', default="hello.txt")
parser.add_argument('-o', '--output', help='sample file', default="sample.txt")
parser.add_argument('--lstm_size', help='number of hidden units', type=int, default=128)
parser.add_argument('--seq_length', help='sequence length', type=int, default=25)
parser.add_argument('--learning_rate', help='learning rate', type=int, default=-1)
parser.add_argument('--sample_interval', help='sample interval', type=int, default=100)
parser.add_argument('--sample_length', help='sample length', type=int, default=250)
parser.add_argument('--plot', help='plot the error', default=False)
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

inputFile = args.input
lstm_size = args.lstm_size
seq_length = args.seq_length
learning_rate = pow(10, int(args.learning_rate))
sample_interval = args.sample_interval
sample_length = args.sample_length
outputFile = args.output + '.out'
weightScale = 1e-2

data = open(inputFile, 'r').read().decode('utf-8')
words = list(set(data.split(' ')))
chars = list(set(data))
data_size = len(data)
vocab_size = len(chars)

#print 'input has %d chars, %d of which are unique.' % (data_size, vocab_size)

char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

Wz, Wi, Wf, Wo = [], [], [], []

Wz.append(np.random.randn(lstm_size, vocab_size) * weightScale)
Wi.append(np.random.randn(lstm_size, vocab_size) * weightScale)
Wf.append(np.random.randn(lstm_size, vocab_size) * weightScale)
Wo.append(np.random.randn(lstm_size, vocab_size) * weightScale)

Rz, Ri, Rf, Ro = [], [], [], []

Rz.append(np.random.randn(lstm_size, lstm_size) * weightScale)
Ri.append(np.random.randn(lstm_size, lstm_size) * weightScale)
Rf.append(np.random.randn(lstm_size, lstm_size) * weightScale)
Ro.append(np.random.randn(lstm_size, lstm_size) * weightScale)

bz, bi, bf, bo = [], [], [], []

bz.append(np.zeros((lstm_size, 1)))
bi.append(np.zeros((lstm_size, 1)))
bf.append(np.zeros((lstm_size, 1)))
bo.append(np.zeros((lstm_size, 1)))

Wy = np.random.randn(vocab_size, lstm_size) * weightScale
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
		z_[t] = np.dot(Wz[0], x[t]) + np.dot(Rz[0], h[t-1]) + bz[0]
		z[t] = tanh(z_[t])

		i_[t] = np.dot(Wi[0], x[t]) + np.dot(Ri[0], h[t-1]) + bi[0]
		i[t] = sigmoid(i_[t])
		
		f_[t] = np.dot(Wf[0], x[t]) + np.dot(Rf[0], h[t-1]) + bf[0]
		f[t] = sigmoid(f_[t])

		c[t] = i[t] * z[t] + f[t] * c[t-1]

		o_[t] = np.dot(Wo[0], x[t]) + np.dot(Ro[0], h[t-1]) + bo[0]
		o[t] = sigmoid(o_[t])

		h[t] = tanh(c[t]) * o[t]

		# output
		y[t] = np.dot(Wy, h[t]) + by

		# normalize
		p[t] = np.exp(y[t]) / np.sum(np.exp(y[t]))

		# loss
		loss += -np.log(p[t][targets[t], 0])

	dWz = np.zeros_like(Wz[0])
	dWi = np.zeros_like(Wi[0])
	dWf = np.zeros_like(Wf[0])
	dWo = np.zeros_like(Wo[0])

	dRz = np.zeros_like(Rz[0])
	dRi = np.zeros_like(Ri[0])
	dRf = np.zeros_like(Rf[0])
	dRo = np.zeros_like(Ro[0])

	dbz = np.zeros_like(bz[0])
	dbi = np.zeros_like(bi[0])
	dbf = np.zeros_like(bf[0])
	dbo = np.zeros_like(bo[0])

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
		dh[t] += np.dot(Rz[0].T, dz[t+1])
		dh[t] += np.dot(Ri[0].T, di[t+1])
		dh[t] += np.dot(Rf[0].T, df[t+1])
		dh[t] += np.dot(Ro[0].T, do[t+1])

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

	return loss, dWz, dWi, dWf, dWo, dRz, dRi, dRf, dRo, dbz, dbi, dbf, dbo, dWy, dby, h[len(inputs)-1], c[len(inputs)-1]



def sample(h, c, seed_ix, n):
	x = np.zeros((vocab_size, 1))
	x[seed_ix] = 1
	ixes = []
	for t in xrange(n):
		z_ = np.dot(Wz[0], x) + np.dot(Rz[0], h) + bz[0]
		z = tanh(z_)

		i_ = np.dot(Wi[0], x) + np.dot(Ri[0], h) + bi[0]
		i = sigmoid(i_)

		f_ = np.dot(Wf[0], x) + np.dot(Rf[0], h) + bf[0]
		f = sigmoid(f_)

		c = i * z + f * c

		o_ = np.dot(Wo[0], x) + np.dot(Ro[0], h) + bo[0]
		o = sigmoid(o_)

		h = tanh(c) * o

		y = np.dot(Wy, h) + by

		p = np.exp(y) / np.sum(np.exp(y))

		ix = np.random.choice(range(vocab_size), p=p.ravel())
		x = np.zeros((vocab_size, 1))
		x[ix] = 1
		ixes.append(ix)
	return ixes


mWz = np.zeros_like(Wz[0])
mWf = np.zeros_like(Wf[0])
mWi = np.zeros_like(Wi[0])
mWo = np.zeros_like(Wo[0])

mRz = np.zeros_like(Rz[0])
mRf = np.zeros_like(Rf[0])
mRi = np.zeros_like(Ri[0])
mRo = np.zeros_like(Ro[0])

mbz = np.zeros_like(bz[0])
mbi = np.zeros_like(bi[0])
mbf = np.zeros_like(bf[0])
mbo = np.zeros_like(bo[0])

mWy = np.zeros_like(Wy)
mby = np.zeros_like(by)

smooth_loss = -np.log(1.0 / vocab_size) * seq_length
n, p = 0, 0

lossData = []

try:
	while True:
		fo = open(outputFile, 'a')
		
		if p + seq_length + 1 >= len(data) or n == 0:
			hprev = np.zeros((lstm_size, 1))
			cprev = np.zeros((lstm_size, 1))
			p = 0

		inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
		targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

		if n % sample_interval == 0:
			sample_ix = sample(hprev, cprev, inputs[0], sample_length)
			txt = ''.join(ix_to_char[ix] for ix in sample_ix)
			txt = txt.encode('utf-8', 'ignore')
			sampled_words = list(set(txt.split(' ')))
			correct = 0
			for word in sampled_words:
				if word in words:
					correct = correct + 1
			accuracy = correct * 1.0 / len(sampled_words)
			fo.write('\n----\n' + txt + '\n----\n')

		loss, dWz, dWi, dWf, dWo, dRz, dRi, dRf, dRo, dbz, dbi, dbf, dbo, dWy, dby, hprev, cprev = lossFun(inputs, targets, hprev, cprev)
		smooth_loss = smooth_loss * 0.999 + loss * 0.001
		if args.plot:
			lossData.append(smooth_loss)

		for param, dparam, mem in zip(
			[Wz[0], Wf[0], Wi[0], Wo[0], Rz[0], Rf[0], Ri[0], Ro[0], bz[0], bf[0], bi[0], bo[0], Wy, by],
			[dWz, dWf, dWi, dWo, dRz, dRf, dRi, dRo, dbz, dbf, dbi, dbo, dWy, dby],
			[mWz, mWf, mWi, mWo, mRz, mRf, mRi, mRo, mbz, mbf, mbi, mbo, mWy, mby]):

			mem += dparam * dparam
			param += -learning_rate * dparam / np.sqrt(mem + 1e-8)

		if n % sample_interval == 0:
			if args.plot:
				plt.clf()
				plt.plot(lossData)
				plt.pause(0.1)
			fo.write('iteration: ' + str(n) + ', loss: ' + str(smooth_loss))
			print 'iteration: %d, loss: %.2f, accuracy: %.2f' % (n, smooth_loss, accuracy)

		p += seq_length
		n += 1
		
		fo.close()

except KeyboardInterrupt:
	fo.close()
	exit