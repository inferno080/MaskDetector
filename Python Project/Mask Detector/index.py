import cv2

# Enable Webcam

Livefeed = cv2.VideoCapture(0)
Livefeed.set(3, 640)
Livefeed.set(4, 480)

# xml files

pathForDetection = 'haarcascade_frontalface_default.xml'
maskDetection = '#'

# main

maskCascade = cv2.CascadeClassifier(maskDetection)
faceCascade = cv2.CascadeClassifier(pathForDetection)
while True:
    success, Gourmet = Livefeed.read()
    Gourmet = cv2.flip(Gourmet, 1)
    Gourmet = cv2.resize(Gourmet, (540, 402))
    GourmetGray = cv2.cvtColor(Gourmet, cv2.COLOR_BGR2GRAY)

    masks = faceCascade.detectMultiScale(GourmetGray, 1.1, 3)
    faces = faceCascade.detectMultiScale(GourmetGray, 1.1, 3)  # Trial and Error for scale and min neighbours

    for (x, y, w, h) in faces:
        cv2.rectangle(Gourmet, (x, y), (x + w, y + h), (0, 0, 255), 2)

    for (x, y, w, h) in masks:
        cv2.rectangle(Gourmet, (x, y), (x + w, y + h), (0, 255, 0), 2)


    cv2.imshow("Video", Gourmet)
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break

# cv2.imshow("Output 0", Gourmet)
# cv2.waitKey(0)


