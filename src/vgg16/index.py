import os
import h5py
import numpy as np
import argparse
import glob

from vgg16 import VGGNet

ap = argparse.ArgumentParser()
ap.add_argument("--index", required = True,help = "Name of index file")
args = vars(ap.parse_args())

if __name__ == "__main__":
  
    feats = []
    names = []

    model = VGGNet()
    # i=0
    for imgPath in glob.glob("../../"+"database"+"/*.jpg"):
        imageId = imgPath[imgPath.rfind("/")+1:]
        norm_feat = model.extract_feat(imgPath)
        feats.append(norm_feat)
        names.append(imageId)
        # print("extracting feature from image No. %d " %((i+1)))
        # i+=1

    feats = np.array(feats)

    output = args["index"]
    
    h5f = h5py.File(output, 'w')
    h5f.create_dataset('dataset_1', data = feats)
    # h5f.create_dataset('dataset_2', data = names)
    h5f.create_dataset('dataset_2', data = np.string_(names))
    h5f.close()