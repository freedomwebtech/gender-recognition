import cv2
import cvlib as cv
import numpy as np
img=cv2.imread("test3.jpeg")
img=cv2.resize(img,(640,480))
faces, confidences = cv.detect_face(img)
for face,conf in zip(faces,confidences):
    x,y=face[0],face[1]
    x1,y1=face[2],face[3]
    cv2.rectangle(img,(x,y),(x1,y1),(0,0,255),3)
    crop=img[y:y1,x:x1]
    (label, confidence) = cv.detect_gender(crop)
    idx = np.argmax(confidence)
    label = label[idx]
    print(label)

    cv2.putText(img,str(label),(x,y),cv2.FONT_HERSHEY_PLAIN,2,(0,255,0),2)
cv2.imshow("img",img)
cv2.waitKey(0)