import numpy as np
import cv2
import imutils

class ColorDescriptor:
	def __init__(self,bins):
		self.bins  = bins

	def describe(self,image):
		image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
		features = []

		(h,w) = image.shape[:2]
		#divide the image into 5 parts(top-left,top-right,bottom-right,bottom-left,center)
		(cx,cy) = (int(w*0.5), int(h*0.5))
		#4 corner segments
		segments = [(0,cx,0,cy),(cx,w,0,cy),(cx,w,cy,h),(0,cx,cy,h)]
		#center (ellipse shape)
		(ex,ey) = (int(w*0.75)//2, int(h*0.75)//2) #axes length
		#elliptical black mask
		ellipMask = np.zeros(image.shape[:2],dtype= "uint8")
		cv2.ellipse(ellipMask,(cx,cy),(ex,ey),0,0,360,255,-1)  # -1 :- fills entire ellipse with 255(white) color
		
		for (startX, endX, startY, endY) in segments:
			# construct a mask for each corner of the image, subtracting the elliptical center from it
			cornerMask = np.zeros(image.shape[:2],dtype = "uint8")
			cv2.rectangle(cornerMask,(startX,startY),(endX,endY),255,-1)
			cornerMask = cv2.subtract(cornerMask,ellipMask)

			hist = self.histogram(image,cornerMask)
			features.extend(hist)

		hist = self.histogram(image,ellipMask)
		features.extend(hist)

		return features


	def histogram(self,image,mask):
		hist = cv2.calcHist([image],[0,1,2],mask,self.bins,[0,180,0,256,0,256])

		if imutils.is_cv2():
			hist = cv2.normalize(hist).flatten()  #cv 2.4
		else: 
			hist = cv2.normalize(hist,hist).flatten() #cv 3+

		return hist