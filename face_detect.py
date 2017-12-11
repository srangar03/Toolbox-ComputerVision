""" Experiment with face detection and image filtering using OpenCV """

import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('~/Downloads/haarcascade_frontalface_alt.xml')
cap = cv2.VideoCapture(0)
kernel = np.ones((21,21),'uint8')

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.2, minSize=(20, 20))
    for (x, y, w, h) in faces:
        frame[y:y+h, x:x+w, :] = cv2.dilate(frame[y:y+h, x:x+w, :], kernel)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255))

        #smile
        cv2.ellipse(frame, (x+int(.5*w),y+int(.65*h)), (int(w/4), int(h/5)), 0, 0, 180, (0,0,255), int(.5*10))

        #eyes
        cv2.circle(frame, (int(x+.35*w), int(y+.4*h)), 20, (200, 106, 30), 10)
        cv2.circle(frame, (int(x+.65*w), int(y+.4*h)), 20, (129, 102, 39), 10)
    # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
    	       break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
