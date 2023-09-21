# USAGE
# python encode_faces.py --dataset dataset --encodings encodings.pickle


# import packages
from imutils import paths
import face_recognition
import pickle
import cv2


# the raw data
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images("/memefaces_230724/#test"))
data = []

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
	# load the input image and convert it from RGB (OpenCV ordering)
	# to dlib ordering (RGB)
	print("[INFO] processing image {}/{}".format(i + 1,
		len(imagePaths)))
	print(imagePath)
	image = cv2.imread(imagePath)
	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	# detect the (x, y)-coordinates of the bounding boxes
	# corresponding to each face in the input image
	boxes = face_recognition.face_locations(rgb,
		model="cnn")

	# compute the facial embedding for the face
	encodings = face_recognition.face_encodings(rgb, boxes)

	# build a dictionary of the image path, bounding box location,
	# and facial encodings for the current image
	d = [{"imagePath": imagePath, "loc": box, "encoding": enc}
		for (box, enc) in zip(boxes, encodings)]
	data.extend(d)

# dump the facial encodings data to disk
print("[INFO] serializing encodings...")
f = open("/memefaces_230724/test.pickle", "wb")
f.write(pickle.dumps(data))
f.close()