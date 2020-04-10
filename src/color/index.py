import argparse
import glob
import cv2

from color import ColorDescriptor

ap = argparse.ArgumentParser()
ap.add_argument("-i","--index",required=True,help="Path to where the computed index will be stored")
args = vars(ap.parse_args())

#in this project we use 8 bins for hue channel,12 for saturation, 3 for value channel
bins = (8,12,3)
#initialize object
cd = ColorDescriptor(bins)

output = open(args["index"],"w")

for imagePath in glob.glob("../../"+"database"+"/*.jpg"):
	imageId = imagePath[imagePath.rfind("/")+1:]
	image = cv2.imread(imagePath)

	features = cd.describe(image)

	features = [str(f) for f in features]
	#feature vector length 5(segments)*8*12*3 = 1440
	output.write("%s,%s\n" % (imageId,",".join(features)))

output.close()

#run
#python3 index.py --index index.csv