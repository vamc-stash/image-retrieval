import numpy as np
import argparse
import glob
import cv2

from hog import HOGDescriptor

ap = argparse.ArgumentParser()
ap.add_argument("--index", required = True,help = "Name of index file")
args = vars(ap.parse_args())

hog = HOGDescriptor()

output = open(args["index"],"w")

for imagePath in glob.glob("../../"+"database"+"/*.jpg"):
	imageId = imagePath[imagePath.rfind("/")+1:]
	image = cv2.imread(imagePath)

	features = hog.describe(image)
	features = [str(f) for f in features]

	output.write("%s,%s\n" % (imageId,",".join(features)))

output.close()