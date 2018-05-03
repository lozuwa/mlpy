import numpy as np

class LogisticRegression(object):
	def __init__(self, x = None, y = None):
		super(LogisticRegression, self).__init__()
		# Data.
		self.x = x
		self.y = y
		# Weights
		self.w = self.initializeWeights
		# Hyperparameters.
		self.epochs = 1000
		self.learningRate = 0.001

	def initializeWeights(self, n = None):
		return np.random.rand(n).reshape(-1, 1)

	def costFunction(self, hwx = None, deriv = None):
		# Log cost function.
		# -1*((y)*log(hwx) + (1-y)*log(1-hwx))
		# Local variables.
		m = self.x.shape[0]
		# Logic.
		if (deriv == True):
			jw = (1/m)
		else:
			jw = (1/m)
		return jw

	def train(self):
		# BGD
		for e in range(self.epochs):
			hwx = np.dot(self.x, self.w)
			self.w = self.w - (self.learningRate * self.costFunction(hwx = hwx, deriv = True))
			if (e%10==0):
				jw = self.costFunction(hwx = hwx, deriv = False)
				print("INFO: Cost function: ", jw)
		