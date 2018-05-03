import numpy as np

class LinearRegression(object):
	def __init__(self, x = None, y = None):
		super(LinearRegression, self).__init__()
		# Data. 
		# X E R(mxn) m=data points ; n=features.
		# Y E R(mx1) m=data points ; single column.
		self.x = x
		self.y = y
		# Weights. W X R(nx1) n=features ; single column.
		self.w = self.initializeWeights(n = x.shape[1])
		# Hyperparameters
		self.epochs = 500
		self.learningRate = 0.01
		
	def initializeWeights(self, n = None, seed = None):
		if (seed != None):
			np.random.seed(seed)
		return np.random.rand(n).reshape(-1, 1)

	def costFunction(self, hwx = None, deriv = None):
		# Local variables.
		m = self.x.shape[0]
		# Logic
		if (deriv == True):
			jw = (1/m) * np.dot(self.x.T, (hwx - self.y))
		else:
			jw = (1/(2*m)) * np.sum((hwx - self.y)**2)
		return jw

	def train(self):
		# BGD
		for e in range(self.epochs):
			hwx = np.dot(self.x, self.w)
			self.w = self.w - (self.learningRate * self.costFunction(hwx = hwx, deriv = True))
			if (e%10 == 0):
				jw = self.costFunction(hwx = hwx, deriv = False)
				print("INFO: Cost function: ", jw)

