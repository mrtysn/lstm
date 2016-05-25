#-*- coding: utf-8 -*-

import numpy as np 						# for computing
from matplotlib import pyplot as plt 	# for plotting
import argparse 						# for argument parsing from command line
import pickle 							# for data dumping as checkpoint files

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='training file', default="hello.txt")
parser.add_argument('-c', '--checkpoint', help='checkpoint file', default='')
parser.add_argument('-o', '--output', help='sample file', default="hello")
parser.add_argument('--lstm_size', help='number of hidden units', type=int, default=128)
parser.add_argument('--seq_length', help='sequence length', type=int, default=25)
parser.add_argument('--learning_rate', help='learning rate', type=int, default=-1)
parser.add_argument('--sample_interval', help='sample interval', type=int, default=1000)
parser.add_argument('--checkpoint_interval', help='checkpoint interval', type=int, default=1000)
parser.add_argument('--sample_length', help='sample length', type=int, default=500)
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

"""
	Creating checkpoints shall be done in a more compact manner. 
	Multi layers will be problematic and redundant if not done so.
"""

def saveWeights():
	weights = {}
	weights['Wz'] = Wz
	weights['Wi'] = Wi
	weights['Wf'] = Wf
	weights['Wo'] = Wo
	weights['Rz'] = Rz
	weights['Ri'] = Ri
	weights['Rf'] = Rf
	weights['Ro'] = Ro
	weights['bz'] = bz
	weights['bi'] = bi
	weights['bf'] = bf
	weights['bo'] = bo
	weights['Wy'] = Wy
	weights['by'] = by	
	weights['iteration'] = iteration
	weights['smooth_loss'] = smooth_loss
	weights['lstm_size'] = lstm_size

	checkpoint = open(args.output + '.checkpoint', 'wb')
	pickle.dump(weights, checkpoint)
	checkpoint.close()

def loadWeights():
	checkpoint = open(args.checkpoint, 'rb')
	weights = pickle.load(checkpoint)
	checkpoint.close()
	global Wz, Wi, Wf, Wo, Rz, Ri, Rf, Ro, bz, bi, bf, bo, Wy, by, lstm_size, iteration, smooth_loss
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
	iteration = weights['iteration']
	smooth_loss = weights['smooth_loss']
	lstm_size = weights['lstm_size'] # override default value or specified value

input_file = args.input
file_name = '.'.join(input_file.split('.')[0:-1])
lstm_size = args.lstm_size
seq_length = args.seq_length
learning_rate = pow(10, int(args.learning_rate))
output_file = args.output + '.out'
weight_scale = 1e-2

input_text = open(input_file, 'r').read().decode('utf-8')
words = list(set(input_text.split(' ')))
chars = list(set(input_text))
input_text_length = len(input_text)
vocab_size = len(chars)

char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

if args.checkpoint is not '':
	loadWeights()
else:
	Wz = np.random.randn(lstm_size, vocab_size) * weight_scale
	Wi = np.random.randn(lstm_size, vocab_size) * weight_scale
	Wf = np.random.randn(lstm_size, vocab_size) * weight_scale
	Wo = np.random.randn(lstm_size, vocab_size) * weight_scale
	
	Rz = np.random.randn(lstm_size, lstm_size) * weight_scale
	Ri = np.random.randn(lstm_size, lstm_size) * weight_scale
	Rf = np.random.randn(lstm_size, lstm_size) * weight_scale
	Ro = np.random.randn(lstm_size, lstm_size) * weight_scale
	
	bz = np.zeros((lstm_size, 1))
	bi = np.zeros((lstm_size, 1))
	bf = np.zeros((lstm_size, 1))
	bo = np.zeros((lstm_size, 1))
	
	Wy = np.random.randn(vocab_size, lstm_size) * weight_scale
	by = np.zeros((vocab_size, 1))

	iteration = 0
	smooth_loss = -np.log(1.0 / vocab_size) * seq_length


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
	dWi = np.zeros_like(Wi)
	dWf = np.zeros_like(Wf)
	dWo = np.zeros_like(Wo)

	dRz = np.zeros_like(Rz)
	dRi = np.zeros_like(Ri)
	dRf = np.zeros_like(Rf)
	dRo = np.zeros_like(Ro)

	dbz = np.zeros_like(bz)
	dbi = np.zeros_like(bi)
	dbf = np.zeros_like(bf)
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

	return loss, dWz, dWi, dWf, dWo, dRz, dRi, dRf, dRo, dbz, dbi, dbf, dbo, dWy, dby, h[len(inputs)-1], c[len(inputs)-1]



def sample(h, c, seed_ix, n):
	x = np.zeros((vocab_size, 1))
	x[seed_ix] = 1
	ixes = []
	t = 0
	while (t < n):# or (ix_to_char[ixes[-1]] != ' '): # also generate characters until the last word is completed
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
		t = t + 1
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
mbi = np.zeros_like(bi)
mbf = np.zeros_like(bf)
mbo = np.zeros_like(bo)

mWy = np.zeros_like(Wy)
mby = np.zeros_like(by)

loss_over_time = []

hprev = np.zeros((lstm_size, 1))
cprev = np.zeros((lstm_size, 1))
pointer = 0

fo = open(output_file, 'a')
fo.write('\n\n***********\n   start   \n***********\n'.encode('utf-8', 'ignore'))
fo.close()

try:
	while True:
		fo = open(output_file, 'a')
		
		if pointer + seq_length + 1 >= input_text_length:
			hprev = np.zeros((lstm_size, 1))
			cprev = np.zeros((lstm_size, 1))
			pointer = 0

		inputs = [char_to_ix[ch] for ch in input_text[pointer:pointer+seq_length]]
		targets = [char_to_ix[ch] for ch in input_text[pointer+1:pointer+seq_length+1]]

		if iteration % args.sample_interval == 0:
			sample_ix = sample(hprev, cprev, inputs[0], args.sample_length)
			txt = ''.join(ix_to_char[ix] for ix in sample_ix)
			sampled_words = list(set(txt.split(' ')))
			correct = 0
			for word in sampled_words:
				if word in words:
					correct = correct + 1
			accuracy = correct * 1.0 / len(sampled_words)
			txt = txt.encode('utf-8', 'ignore')
			fo.write('\n-----\n' + txt + '\n-----\n')

		loss, dWz, dWi, dWf, dWo, dRz, dRi, dRf, dRo, dbz, dbi, dbf, dbo, dWy, dby, hprev, cprev = lossFun(inputs, targets, hprev, cprev)
		smooth_loss = smooth_loss * 0.999 + loss * 0.001
		if args.plot:
			loss_over_time.append(smooth_loss)

		for param, dparam, mem in zip(
			[Wz, Wf, Wi, Wo, Rz, Rf, Ri, Ro, bz, bf, bi, bo, Wy, by],
			[dWz, dWf, dWi, dWo, dRz, dRf, dRi, dRo, dbz, dbf, dbi, dbo, dWy, dby],
			[mWz, mWf, mWi, mWo, mRz, mRf, mRi, mRo, mbz, mbf, mbi, mbo, mWy, mby]):

			mem += dparam * dparam
			param += -learning_rate * dparam / np.sqrt(mem + 1e-8)

		
		if iteration % args.sample_interval == 0:
			if args.plot:
				plt.clf()
				plt.plot(loss_over_time)
				plt.pause(0.1)
			status = 'iteration: %d, loss: %.2f, accuracy: %.2f' % (iteration, smooth_loss, accuracy)
			fo.write(status)
			print status

		if iteration % args.checkpoint_interval == 0:
			saveWeights()

		pointer += seq_length
		iteration += 1
		
		fo.close()

except KeyboardInterrupt:
	fo.close()
	print "\n"
	exit