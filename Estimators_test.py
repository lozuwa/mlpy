import unittest
import numpy as np
from LinearRegression import *
from LogisticRegression import *
import matplotlib.pyplot as plt

class Estimators_test(unittest.TestCase):
	def setUp(self):
		# Linear.
		# self.x = np.arange(0, 3, 0.1).reshape(-1, 1)
		# self.y = np.arange(0, 3, 0.1).reshape(-1, 1)
		# Sin.
		self.samp_x = np.arange(0, 3, 0.1).reshape(-1, 1)
		self.x = self.samp_x
		# Add bias
		self.x = np.c_[self.x, np.ones([self.x.shape[0], 1])]
		# Add polymonial terms.
		for i in range(2, 4):
			self.x = np.c_[self.x, self.samp_x**i]
		self.y = np.array([np.sin(i) for i in self.samp_x]).reshape(-1, 1)
		# Polynomial
		# self.x = np.arange(0, 3, 0.1).reshape(-1, 1)
		# self.y = np.arange(0, 3, 0.1).reshape(-1, 1)

	def test_LinearRegression(self):
		# Create linear regression object.
		lr = LinearRegression(x = self.x, y = self.y)
		# Train data.
		lr.train()
		# Plot data.
		plt.scatter(self.samp_x, np.dot(lr.x, lr.w), c = "r")
		plt.scatter(self.samp_x, self.y,  c = "b")
		plt.show()

if __name__ == "__main__":
	unittest.main()
