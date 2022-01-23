import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
import numpy as np

cap = cv2.VideoCapture(0)
detector = FaceMeshDetector(maxFaces=2)


textList = ["welcome to face depth",
            "recharge the battery",
            "press the button to start",
            "press the button to stop",
            "easy to read text",
            "dynamic text change"]

sensitivy = 10  # more sensitive, less sensitive

while True:
    success, img = cap.read()
    imgText = np.zeros_like(img)
    img, faces = detector.findFaceMesh(img)
    # img, faces = detector.findFaceMesh(img, draw=False)

    if faces:
        face = faces[0]
        pointLeft = face[145]
        pointRight = face[374]
        nose = face[19]
        cv2.line(img, pointLeft, pointRight, (0, 255, 0), 2)
        # cv2.line(img, nose, pointLeft, (0, 255, 0), 2)
        # cv2.line(img, nose, pointRight, (0, 255, 0), 2)
        cv2.circle(img, pointLeft, 5, (0, 255, 255), cv2.FILLED)
        cv2.circle(img, pointRight, 5, (0, 255, 255), cv2.FILLED)
        cv2.circle(img, nose, 5, (0, 255, 255), cv2.FILLED)
        w, _ = detector.findDistance(pointLeft, pointRight)
        # print(w)
        W = 6.3

        ### Finding the distance ###
        f = 840
        d = (W*f)/w
        print(d)

        cvzone.putTextRect(img, f'Depth: {int(d)}cm',
                           (face[10][0] - 100, face[10][1] - 50),
                           scale=2)

        for i, text in enumerate(textList):
            ## text size calculation ###
            singleHeight = 20 + int((int(d/sensitivy) * sensitivy)/4)
            scale = 0.4 + (int(d/sensitivy)*sensitivy)/200
            ### put text on image ###
            cv2.putText(imgText, text, (50, 50+(i*singleHeight)),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 255, 0), 2)

    imgStacked = cvzone.stackImages([img, imgText], 2, 1)
    cv2.imshow("Image", imgStacked)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
