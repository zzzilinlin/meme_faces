# USAGE
# python cluster_faces.py --encodings encodings.pickle

#code adapted from: https://pyimagesearch.com/2018/07/09/face-clustering-with-python/
# import packages
from sklearn.cluster import DBSCAN
from imutils import build_montages
import numpy as np
import pickle
import cv2
import csv


# load the serialized face encodings + bounding box locations from disk
print("[INFO] loading encodings...")
data = pickle.loads(open('/memefaces_230724/test.pickle', "rb").read())
data = np.array(data)
# extract the set of encodings for clustering
encodings = [d["encoding"] for d in data]

# cluster the embeddings (min sample size = 5)
print("[INFO] clustering...")
clt = DBSCAN(metric="euclidean", min_samples=5, n_jobs=-1)
clt.fit(encodings)

# determine the total number of unique faces found in the dataset
labelIDs = np.unique(clt.labels_)
numUniqueFaces = len(np.where(labelIDs > -1)[0])
print("[INFO] # unique faces: {}".format(numUniqueFaces))


# loop over the unique face integers
for labelID in labelIDs:
    print("[INFO] faces for face ID: {}".format(labelID))
    
    # find all indexes
    idxs = np.where(clt.labels_ == labelID)[0]
    all_paths = []
    for i in idxs:
        all_paths.append(data[i]["imagePath"])  
    ## save all paths file as a CSV
    with open("Face_ID_{}_paths.csv".format(labelID), mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['imagePath'])
        writer.writerows([[path] for path in all_paths])
        
    # create a montage with 25 samples   
    idxs = np.random.choice(idxs, size=min(25, len(idxs)), replace=False)
    faces = []
    paths = []

    # loop over the sampled indexes
    for i in idxs:
        # load the input image and extract the face ROI
        image = cv2.imread(data[i]["imagePath"])
        (top, right, bottom, left) = data[i]["loc"]
        face = image[top:bottom, left:right]

        # force resize the face ROI to 96x96 and then add it to the faces montage list
        face = cv2.resize(face, (96, 96))
        faces.append(face)
        paths.append(data[i]["imagePath"])

    ## save the motage path file as a CSV
    with open("Montage_Face_ID_{}_paths.csv".format(labelID), mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['imagePath'])
        writer.writerows([[path] for path in paths])
        
    # create a montage using 96x96 "tiles" with 5 rows and 5 columns
    montage = build_montages(faces, (96, 96), (5, 5))[0]

    # save the output montage
    title = "Face ID #{}.jpg".format(labelID)
    title = "Unknown Faces.jpg" if labelID == -1 else title
    cv2.imwrite(title, montage)
