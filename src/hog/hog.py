from skimage.feature import hog
import numpy as np
import cv2

class HOGDescriptor:
	def __init__(self):
		self.n_bins = 10
		self.n_slice = 6
		self.n_orient = 8
		self.pixels_per_cell = (2,2)
		self.cells_per_block = (1,1)

	def describe(self,image):
		height,width,channel = image.shape

		hist = np.zeros((self.n_slice,self.n_slice,self.n_bins)) 
		h_silce = np.around(np.linspace(0, height, self.n_slice+1, endpoint=True)).astype(int)
		w_slice = np.around(np.linspace(0, width, self.n_slice+1, endpoint=True)).astype(int)
		for hs in range(len(h_silce)-1):
			for ws in range(len(w_slice)-1):
		 		img_r = image[h_silce[hs]:h_silce[hs+1], w_slice[ws]:w_slice[ws+1]]  # slice img to regions
		 		hist[hs][ws] = self._HOG(img_r,self.n_bins)

		hist /= np.sum(hist)

		return hist.flatten()

	def _HOG(self,image,n_bins):
		gray_img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
		feats = hog(gray_img,orientations=self.n_orient,pixels_per_cell=self.pixels_per_cell,cells_per_block=self.cells_per_block)
		bins = np.linspace(0, np.max(feats), n_bins+1, endpoint=True)
		hist, b = np.histogram(feats,bins=bins)

		hist = np.array(hist)/np.sum(hist)

		return hist 

