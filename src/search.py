import argparse
import cv2
import numpy as np
import h5py

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from color.color import ColorDescriptor
from gabor.gabor import GaborDescriptor
from hog.hog import HOGDescriptor
from vgg16.vgg16 import VGGNet

from searcher import Searcher

ap = argparse.ArgumentParser()
ap.add_argument("-q", "--query", required = True,help = "Path to the query image")
ap.add_argument("-c", "--class", required = True,help = "Feature class")
args = vars(ap.parse_args())

if args["class"] == "color" or args["class"] == "gabor" or args["class"] == "hog":

	if(args["class"] == "color"):
		bins = (8,12,3)
		cd = ColorDescriptor(bins)

		query = cv2.imread(args["query"])
		features = cd.describe(query)

		searcher = Searcher("color/index.csv")
		results = searcher.search(features)

	elif args["class"] == "gabor":
		params = {"theta":4, "frequency":(0,1,0.5,0.8), "sigma":(1,3),"n_slice":2}
		gd = GaborDescriptor(params)
		gaborKernels = gd.kernels()

		query = cv2.imread(args["query"])
		features = gd.gaborHistogram(query,gaborKernels)

		searcher = Searcher("gabor/index.csv")
		results = searcher._gsearch(features)

	elif args["class"] == "hog":
		hog = HOGDescriptor()

		query = cv2.imread(args["query"])
		features = hog.describe(query)

		searcher = Searcher("hog/index.csv")
		results = searcher.search(features)


	myWindow = cv2.resize(query,(960,960))
	cv2.imshow("query",myWindow)

	for (score,resultId) in results:
		print(score)
		result = cv2.imread("../" +  "database" +"/"+resultId)
		print(resultId)
		myWindow = cv2.resize(result,(480,480))
		cv2.imshow("Result: "+str(score),myWindow)
		ch = cv2.waitKey(0)
		if ch == ord('q'):
			pass

elif args["class"] == "vgg16":

	h5f = h5py.File("vgg16/index.h5",'r')
	feats = h5f['dataset_1'][:]
	#print(feats)
	imgNames = h5f['dataset_2'][:]
	#print(imgNames)
	h5f.close()
   
	query = cv2.imread(args["query"])

	model = VGGNet()

	queryVec = model.extract_feat(args["query"])
	# dot product between two vectors can be used as aggregate for similarity as the projection of vector u on vector v (u^T.v) is considered as similar 
	# when the angle between them is 0 degrees. Therefore, more is the resultant of their product implies more is the similarity b/n them
	scores = np.dot(queryVec, feats.T)
	#print(scores)
	rank_ID = np.argsort(scores)[::-1]
	#print(rank_ID)
	rank_score = scores[rank_ID]
	print(rank_score)

	myWindow = cv2.resize(query,(960,960))
	cv2.imshow("query",myWindow)

	maxres = 10
	imlist = [imgNames[index] for i,index in enumerate(rank_ID[0:maxres])]
	print("top %d images in order are: " %maxres, imlist)

	# show top #maxres retrieved result one by one
	for i,im in enumerate(imlist):
	    result = cv2.imread("../" +  "database" +"/"+str(im, 'utf-8'))
	    myWindow = cv2.resize(result,(480,480))
	    cv2.imshow("Result: ",myWindow)
	    ch = cv2.waitKey(0)
	    if ch == ord('q'):
	    	pass

#run
# python3 search.py --query ../query_images/results_pyramids.jpg --class color 
