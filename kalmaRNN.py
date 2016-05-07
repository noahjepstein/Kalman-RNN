# kalmaRNN.py
# Mimialistic RNN implementation with single hidden layer
# Performs logistic regression on an IMU-data timeseries
# Predicts kalman-filtered orientation (quaternion-representation!)
# 
# This is a modified theano RNN implementation!!
# Original version by T. Ramalho
# https://github.com/tmramalho

# Modified by Noah J Epstein
# Modified Date: 5/3/2016
# noahjepstein@gmail.com

import theano
import theano.tensor as T
import theano.printing as tPrint	
import numpy as np
import cPickle
import random
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

class RNN(object):

	def __init__(self, nin, n_hidden, nout):
		
		rng = np.random.RandomState(1234)
		
		W_uh = np.asarray(
			rng.normal(size=(nin, n_hidden), scale= .01, loc = .0), dtype = theano.config.floatX)

		W_hh = np.asarray(
			rng.normal(size=(n_hidden, n_hidden), scale=.01, loc = .0), dtype = theano.config.floatX)

		W_hy = np.asarray(
			rng.normal(size=(n_hidden, nout), scale =.01, loc=0.0), dtype = theano.config.floatX)

		b_hh = np.zeros((n_hidden,), dtype=theano.config.floatX)
		b_hy = np.zeros((nout,), dtype=theano.config.floatX)
		
		self.activate = T.nnet.sigmoid
		
		lr = T.scalar()
		u = T.matrix()
		t = T.matrix()

		W_uh = theano.shared(W_uh, 'W_uh')
		W_hh = theano.shared(W_hh, 'W_hh')
		W_hy = theano.shared(W_hy, 'W_hy')
		b_hh = theano.shared(b_hh, 'b_hh')
		b_hy = theano.shared(b_hy, 'b_hy')

		h0_tm1 = theano.shared(np.zeros(n_hidden, dtype=theano.config.floatX))

		h, _ = theano.scan(self.recurrentLoop, sequences = u,
					   outputs_info = [h0_tm1],
					   non_sequences = [W_hh, W_uh, W_hy, b_hh])

		# calculates output based on 
		y = T.dot(h[-1], W_hy) + b_hy
		
		# cost function is just mean squared error at the outputs_info
		cost = ((t - y)**2).mean(axis=0).sum()

		# calculate gradients across weights and biases wrt cost
		gW_hh, gW_uh, gW_hy, gb_hh, gb_hy = T.grad(cost, [W_hh, W_uh, W_hy, b_hh, b_hy])	


		self.trainModel = theano.function(inputs = [u, t, lr], outputs = cost,
							on_unused_input='warn',
							updates=[ (W_hh, W_hh - lr * gW_hh),
									  (W_uh, W_uh - lr * gW_uh),
									  (W_hy, W_hy - lr * gW_hy),
									  (b_hh, b_hh - lr * gb_hh),
									  (b_hy, b_hy - lr * gb_hy)],
							allow_input_downcast=True)

		self.test = theano.function(inputs = [u], outputs = cost,
									on_unused_input = 'warn', 
									allow_input_downcast = True)

	
	# update 
	def recurrentLoop(self, u_t, h_tm1, W_hh, W_uh, W_hy, b_hh):
		h_t = self.activate(T.dot(h_tm1, W_hh) + T.dot(u_t, W_uh) + b_hh)
		return h_t

###############################################################################
############################## JUST FOR RUNNING MODEL #########################
###############################################################################

# so we can run as a library too
if __name__ == '__main__':
	
	RNNModel = RNN(9, 30, 4)
	lr = 0.001
	e = 1
	errorList = []
	minSeqLen = 100
	
	# unpickle training data
	trajectories = cPickle.load(open('trainingBin.pickle', 'rb'))
	nTraj = len(trajectories)
	print nTraj

	print "training time!"

	testList = cPickle.load(open('testingBin.pickle', 'rb'))
	testCosts = []	

	for i in xrange(int(2)):

		for j, traj in enumerate(trajectories): 

			# trains on each trajectory truncated to length 100
			# so we don't have to deal with time series of differing lengths
			u = traj['nnInput'][:100,:]
			t = traj['KalmanOrientEst'][:100,:]
			c = RNNModel.trainModel(u, t, lr)

			print "iteration {0}: {1}".format(i * nTraj + j, np.sqrt(c))
		
			if j % 20 == 0: 
				e = 0.1*np.sqrt(c) 
				errorList.append(e)
	
		for traj in testList: 
			
			cost = RNNModel.test(traj['nnInput'])
			err = 0.1 * np.sqrt(err)
			testCosts.append(err)

	
	plt.plot(errorList)
	plt.savefig('plots/error.png')

	plt.plot(testCosts)
	plt.savefig('plots/testCosts.png')

	# sys.setrecursionlimit(3000) # bc the nn is highly recurisve
	# cPickle.dump(rnn, open('trainedModel.pickle', 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)


	

