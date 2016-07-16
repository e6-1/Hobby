"""
Simple linear regression implementation using least squares.

"""

import numpy as np

class LinReg(object):
	"""Initialized by two arrays, this will compute a line of best fit using least squares.

	Attributes:
		x_hat: Initial X array.
		y_hat: Initial Y array.
		beta: Array of coefficients.
		bias: Offset.
	"""

	def __init__(self, x, y):
		"""Initializes LinReg with x and y.

		NOTE:
			Input vectors must be of the same shape.

		ARGS:
			x : Training inputs.
			y : Expected outputs.
		"""
		if isinstance(x, basestring):
			if not isinstance(x[0], (int, long, float)) or not isinstance(y[0], (int, long, float)):
				print "Vector entries' types must be int/long/float."
			elif len(x) != len(y):
				print "Input vectors must have the same shape."
			else:
				x = np.array(x)
				y = np.array(y)
		else:
			if not isinstance(x, (int, long, float)) or not isinstance(y, (int, long, float)):
				print "Entries' types must be int/long/float."
			else:
				x = np.array(x)
				y = np.array(y)
		if isinstance(x, np.ndarray):
			if x.shape != y.shape:
				print "Input vectors must have the same shape."
			self.x_hat = x
			self.y_hat = y
			self.beta = np.zeros_like(x)
			self.bias = np.zeros_like(x)
		else:
			print("Inputs must be of a list or numpy array of int/long/float.")

	def fit(self):
		"""Computes the beta coefficients."""
		x_transpose = np.transpose(self.x_hat)
		x_product = np.dot(x_transpose, self.x_hat)
		if not isinstance(x_product, np.ndarray) and isinstance(x_product, (int, long, float)):
			x_product_inv = 1.0 / x_product
		else:
			x_product_inv = np.linalg.inv(x_product)
		x_product_inv_trans = np.dot(x_product_inv, x_transpose)
		self.beta = np.dot(x_product_inv_trans, self.y_hat)
		self.bias = np.mean(self.y_hat) - self.beta * np.mean(self.x_hat)
		print("Computed model: y = {:2f} * x + {:2f}".format(self.beta, self.bias))

	def predict(self, x):
		"""Predicts y given x."""
		return x * self.beta + self.bias
