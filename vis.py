#-*- coding: utf-8 -*-

import numpy as np 						# for computing
import argparse 						# for argument parsing from command line
import pickle 							# for data dumping as checkpoint files
import sys

argv = sys.argv[1:]
sys.argv = sys.argv[:1]
if "--" in argv:
	index = argv.index("--")
	kivy_args = argv[index+1:]
	argv = argv[:index]
	sys.argv.extend(kivy_args)

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='training file', default="hello.txt")
parser.add_argument('-c', '--checkpoint', help='checkpoint file', default='')
args = parser.parse_args(argv)

def rgb(value):
	if value > 0:
		r = value
		b = 0
	else:
		r = 0;
		b = - value
	g = 0
	return r, g, b

from kivy.app import App
from kivy.uix.widget import Widget
from kivy.core.window import Window
from kivy.clock import Clock
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty, StringProperty, ListProperty
from kivy.uix.gridlayout import GridLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.garden.graph import Graph, MeshLinePlot, SmoothLinePlot, LinePlot

class MyLabel(Label):
	pass

class LstmGate(Widget):
	r = NumericProperty(0)
	g = NumericProperty(0)
	b = NumericProperty(0)
	ch = StringProperty()
	pass


class LstmCell(Widget):
	z_gate = ObjectProperty(None)
	i_gate = ObjectProperty(None)
	f_gate = ObjectProperty(None)
	o_gate = ObjectProperty(None)
	c_gate = ObjectProperty(None)
	h_gate = ObjectProperty(None)

	def paint_h(self, x):
		x = round(x, 2)
		r, g, b = rgb(x)
		self.h_gate.r = r
		self.h_gate.g = g
		self.h_gate.b = b
		self.h_gate.ch = str(x)

	def paint_c(self, x):
		x = round(x, 2)
		r, g, b = rgb(x)
		self.c_gate.r = r
		self.c_gate.g = g
		self.c_gate.b = b
		self.c_gate.ch = str(x)

	def paint_z(self, x):
		x = round(x, 2)
		r, g, b = rgb(x)
		self.z_gate.r = r
		self.z_gate.g = g
		self.z_gate.b = b
		self.z_gate.ch = str(x)

	def paint_i(self, x):
		x = round(x, 2)
		r, g, b = rgb(x)
		self.i_gate.r = r
		self.i_gate.g = g
		self.i_gate.b = b
		self.i_gate.ch = str(x)

	def paint_f(self, x):
		x = round(x, 2)
		r, g, b = rgb(x)
		self.f_gate.r = r
		self.f_gate.g = g
		self.f_gate.b = b
		self.f_gate.ch = str(x)

	def paint_o(self, x):
		x = round(x, 2)
		r, g, b = rgb(x)
		self.o_gate.r = r
		self.o_gate.g = g
		self.o_gate.b = b
		self.o_gate.ch = str(x)

	def set_char(self, ch):
		self.cell_char = ch


class MyLayout(GridLayout):
	mainLayout = None
	leftLayout = None
	rightLayout = None

	cellArea = None
	characterArea = None

	graphArea = None
	equationArea = None

	cells = []
	graphs = []

	def __init__(self, **kwargs):
		super(MyLayout, self).__init__(**kwargs)
		self._keyboard = Window.request_keyboard(self.close, self)
		self._keyboard.bind(on_key_down=self.press)

	def close(self):
		self._keyboard.unbind(on_key_down=self.press)
		self._keyboard = None

	def press(self, keyboard, keycode, text, modifiers):
		if keycode[1] == 'spacebar':
			self.update()
		return True

	def build(self):
		self.mainLayout = BoxLayout(orientation='horizontal')
		
		self.leftLayout = BoxLayout(orientation='vertical', size_hint=(.3, 1))
		self.rightLayout = BoxLayout(orientation='vertical', size_hint=(.7, 1))

		self.mainLayout.add_widget(self.leftLayout)
		self.mainLayout.add_widget(self.rightLayout)

		self.graphArea = BoxLayout(orientation='vertical', size_hint=(1, .9), padding = 10, spacing = 15)

		self.leftLayout.add_widget(self.graphArea)

		tempStr = '[1, 0] = ' + ix_to_char[0] + '\n[0, 1] = ' + ix_to_char[1]

		self.leftLayout.add_widget(MyLabel(text=tempStr, size_hint=(1, .1)))

		for _y in range(lstm_size):
			graph = Graph(	xlabel = 'Time', ylabel = 'Value', y_grid_label = True, x_grid_label = True, 
							padding = 10, x_grid = False, y_grid = False, ymin = -1.05, ymax = 1.05, size_hint=(1, 1.0/lstm_size))
			self.graphs.append(graph)
			self.graphArea.add_widget(graph)


		if dispEquations:
			self.equationArea = MyLabel(text='', size_hint=(1, .5))
			self.rightLayout.add_widget(self.equationArea)
			self.cellArea = BoxLayout(orientation='vertical', size_hint=(1, .4))
		else:
			self.cellArea = BoxLayout(orientation='vertical', size_hint=(1, .9))
		

		
		for _y in range (lstm_size):
			row = BoxLayout(orientation='horizontal', spacing=15, padding = 10, size_hint=(1, 1.0/lstm_size))
			for width in range(time_window):
				row.add_widget(Widget(size_hint=(1.0/time_window, 1)))
			self.cellArea.add_widget(row)
			self.cells.append(row)

		self.characterArea = BoxLayout(orientation='horizontal', size_hint=(1, .1))
		for c in range(time_window):
			self.characterArea.add_widget(Widget(size_hint=(1.0/time_window, 1)))

		
		self.rightLayout.add_widget(self.cellArea)
		self.rightLayout.add_widget(self.characterArea)

		
		
		return self.mainLayout

	def update(self):
		global t, x

		ch, prob = onestep()
		for _y in range(lstm_size):
			widget = self.cells[_y].children[-1]
			self.cells[_y].remove_widget(widget)
			box = LstmCell(size_hint=(1.0/time_window, 1))
			box.paint_z(z[t][_y])
			box.paint_i(i[t][_y])
			box.paint_f(f[t][_y])
			box.paint_o(o[t][_y])
			box.paint_h(h[t][_y])
			box.paint_c(c[t][_y])
			#box.set_char(ch)
			self.cells[_y].add_widget(box)
		widget = self.characterArea.children[-1]
		self.characterArea.remove_widget(widget)
		ch = ' ' + ch + "\n" + str(round(prob, 2))
		self.characterArea.add_widget(MyLabel(text=ch, size_hint=(1.0/time_window, 1)))

		if dispEquations:
			myStr = ""
			for _y in range(lstm_size):
				myStr += 'z[t] = tanh([' + str(round(Wz[0][0], 2)) + ', ' + str(round(Wz[0][1], 2)) + '] · [' + str(int(x[0])) + ', ' + str(int(x[1])) + '].T + [' + str(round(Rz[0][0], 2)) + '] · [' + str(round(h[t-1], 2)) + '] + ' + str(round(bz, 2)) +') = tanh('+ str(round(z_, 2)) +') = ' + str(round(z[t], 2)) + "\n"
				myStr += 'i[t] = sig([' + str(round(Wi[0][0], 2)) + ', ' + str(round(Wi[0][1], 2)) + '] · [' + str(int(x[0])) + ', ' + str(int(x[1])) + '].T + [' + str(round(Ri[0][0], 2)) + '] · [' + str(round(h[t-1], 2)) + '] + '+ str(round(bi, 2)) +') = sig('+ str(round(i_, 2)) +') = ' + str(round(i[t], 2)) + "\n"
				myStr += 'f[t] = sig([' + str(round(Wf[0][0], 2)) + ', ' + str(round(Wf[0][1], 2)) + '] · [' + str(int(x[0])) + ', ' + str(int(x[1])) + '].T + [' + str(round(Rf[0][0], 2)) + '] · [' + str(round(h[t-1], 2)) + '] + '+ str(round(bf, 2)) +') = sig('+ str(round(f_, 2)) +') = ' + str(round(f[t], 2)) + "\n"
				myStr += 'c[t] = '+ str(round(i[t], 2)) +' * '+ str(round(z[t], 2)) +' + '+ str(round(f[t], 2)) +' * ' + str(round(c[t-1], 2)) + ' = ' + str(round(i[t] * z[t], 2)) + ' + ' + str(round(f[t] * c[t-1], 2)) + ' = ' + str(round(c[t], 2)) + "\n"
				myStr += 'o[t] = sig([' + str(round(Wo[0][0], 2)) + ', ' + str(round(Wo[0][1], 2)) + '] · [' + str(int(x[0])) + ', ' + str(int(x[1])) + '].T + [' + str(round(Ro[0][0], 2)) + '] · [' + str(round(h[t-1], 2)) + '] + '+ str(round(bo, 2)) +') = sig('+ str(round(o_, 2)) +') = ' + str(round(o[t], 2)) + "\n"
				myStr += 'h[t] = tanh(' + str(round(c[t], 2)) + ') * ' + str(round(o[t], 2)) + ' = ' + str(round(tanh(c[t]) , 2)) + ' * ' + str(round(o[t], 2)) + ' = ' + str(round(h[t], 2)) + "\n"
				myStr += 'y[t] = [' + str(round(Wy[0][0], 2)) + ', ' + str(round(Wy[1][0], 2)) + '] · ' + str(round(h[t], 2)) + ' + [' + str(round(by[0], 2)) + ', ' + str(round(by[1], 2)) + '].T = [' + str(round(y[t][0], 2)) + ', ' + str(round(y[t][1], 2)) + '].T' + "\n"
				myStr += 'Input: [' + str(int(x[0])) + ', ' + str(int(x[1])) + '] --> Output: [' + str(int(x_next[0])) + ', ' + str(int(x_next[1])) + ']'
			self.equationArea.text = myStr

		for _y in range(lstm_size):
			self.graphs[_y].xmin = t - time_window
			self.graphs[_y].xmax = t

			for plot in self.graphs[_y].plots:
				self.graphs[_y].remove_plot(plot)

			ix = 1
			for v in (z, i, f, o, c, h):
				colors = [int(u) for u in bin(ix)[2:]]
				while len(colors) < 3:
					colors.insert(0, 0)
				ix += 1
				plot = LinePlot(color = colors, line_width = 2)
				plot.points = [(u, v[u][_y]) for u in range(max(0, t-time_window), t+1)]
				self.graphs[_y].add_plot(plot)

		t += 1
		return



class LstmVisualApp(App):
	def build(self):
		skel = MyLayout()
		#Clock.schedule_interval(skel.update, 1.0)
		return skel.build()


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

input_file = args.input

input_text = open(input_file, 'r').read().decode('utf-8')
chars = list(set(input_text))
vocab_size = len(chars)

char_to_ix = { ch:ix for ix,ch in enumerate(chars) }
ix_to_char = { ix:ch for ix,ch in enumerate(chars) }


if args.checkpoint is not '':
	loadWeights()
else:
	exit

def onestep():
	global h, c, x, t
	global z, i, f, o, y
	global z_, i_, f_, o_
	global x_next

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
	#print "character: \'%s\'" % (ch)
	
	x_next = np.zeros((vocab_size, 1))
	x_next[ix] = 1

	return ch, p[ix]


time_window = 10

z, i, f, o, y = [], [], [], [], []
for v in (z, i, f, o, y):
	v.append([0])
h, c = [], []
h.append(np.zeros((lstm_size, 1)))
c.append(np.zeros((lstm_size, 1)))
t = 1


x = np.zeros((vocab_size, 1))
#x[np.random.randint(vocab_size)] = 1
x[0] = 1

x_next = x

dispEquations = True

if __name__ == '__main__':
	Window.size = (1920, 1080)
	#Window.fullscreen = True
	LstmVisualApp().run()
	