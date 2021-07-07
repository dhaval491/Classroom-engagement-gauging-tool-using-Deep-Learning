import cv2
import matplotlib.pyplot as plt
import sys

def face_detect(image):
    # Get user supplied values
    cascPath = "haarcascade_frontalface_default.xml"

    # Create the haar cascade
    faceCascade = cv2.CascadeClassifier(cascPath)

    # Read the image    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )
    return faces

def face_crop(image):
    faces=face_detect(image)
    if len(faces)==0:
        #print("No face. Returning original.")
        return image
    x,y,w,h=faces[0]
    if w<10 or h<10:
        #print("Face too small. Returning original.")
        return image
    image_crop=image[y:y+h,x:x+w]
    return image_crop

if __name__ == "__main__":
    imagePath = sys.argv[1]
    image = cv2.imread(imagePath)
    
    plt.imshow(plt.imread(imagePath))
    plt.show()
    
    faces=face_detect(image)
    print("Found {0} faces!".format(len(faces)))

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    plt.imshow(image)
    plt.show()
