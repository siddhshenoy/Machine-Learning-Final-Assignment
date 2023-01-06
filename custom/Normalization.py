import numpy as np

class Normalization:
	@staticmethod
	def z_score(x):
		return (x - x.mean()) / (x.std())

	@staticmethod
	def gaussian(x):
		return np.exp(-pow(x, 2))

	@staticmethod
	def minmax(x):
		return (x - x.min()) / (x.max() - x.min())
