import numpy as np 
import csv

class Searcher:
	def __init__(self,indexPath):
		self.indexPath = indexPath

	def search(self, queryFeatures, limit=10):
		results = {}

		with open(self.indexPath) as i:
			reader = csv.reader(i)

			for row in reader:
				features = [float(x) for x in row[1:]]
				d = self.chi_squared_distance(features,queryFeatures)
				results[row[0]] = d

		i.close()

		results = sorted([(v,k) for (k,v) in results.items()])

		return results[:limit]

	def _gsearch(self, queryFeatures, limit=10):
		results = {}
		with open(self.indexPath) as i:
			reader = csv.reader(i)

			for row in reader:
				features = [float(x) for x in row[1:]]
				error = np.sum((queryFeatures - features)**2)
				results[row[0]] = error

		i.close()

		results = sorted([(v,k) for (k,v) in results.items()])
		return results[:limit]

	def chi_squared_distance(self, histA, histB, eps=1e-10):

		d = 0.5*np.sum([((a-b)**2)/(a+b+eps) for (a,b) in zip(histA,histB)])

		return d