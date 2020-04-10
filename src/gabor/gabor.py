import numpy as np 
from scipy import ndimage as ndi
from skimage import data
from skimage.util import img_as_float
from skimage.filters import gabor_kernel
import cv2


class GaborDescriptor:
	def __init__(self,params):
		self.theta = params['theta']
		self.frequency = params['frequency']
		self.sigma = params['sigma']
		self.n_slice = params['n_slice']

	def kernels(self):
		kernels = []
		for theta in range(self.theta):
			theta = theta/4. * np.pi
			for frequency in self.frequency:
				for sigma in self.sigma:
					kernel = gabor_kernel(frequency,theta=theta,sigma_x=sigma,sigma_y=sigma)
					kernels.append(kernel)
		return kernels

	def gaborHistogram(self,image,gabor_kernels):
		height,width,channel = image.shape
		#height & width of image will equally sliced into N slices
		hist = np.zeros((self.n_slice,self.n_slice,2*len(gabor_kernels))) #2*len coz to store mean and variance
		h_silce = np.around(np.linspace(0, height, self.n_slice+1, endpoint=True)).astype(int)
		w_slice = np.around(np.linspace(0, width, self.n_slice+1, endpoint=True)).astype(int)
		for hs in range(len(h_silce)-1):
			for ws in range(len(w_slice)-1):
		 		img_r = image[h_silce[hs]:h_silce[hs+1], w_slice[ws]:w_slice[ws+1]]  # slice img to regions
		 		hist[hs][ws] = self._gabor(img_r,gabor_kernels)

		hist /= np.sum(hist)
		#print(hist.shape)
		return hist.flatten()

	def _power(self,image,kernel):
		image = (image - image.mean()) / image.std() 
		f_img = np.sqrt(ndi.convolve(image, np.real(kernel), mode='wrap')**2 + ndi.convolve(image, np.imag(kernel), mode='wrap')**2)
		feats = np.zeros(2, dtype=np.double)
		feats[0] = f_img.mean()
		feats[1] = f_img.var()
		return feats

	def _gabor(self,image,gabor_kernels):
		gray_img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
		results = []
		for kernel in gabor_kernels:
			results.append(self._power(gray_img,kernel))

		hist = np.array(results)
		hist = hist / np.sum(hist, axis=0)
		#print(hist.flatten())
		#print(hist.T.flatten())

		return hist.T.flatten() # .T -> transpose
