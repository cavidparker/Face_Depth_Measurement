import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector

cap = cv2.VideoCapture(0)
detector = FaceMeshDetector(maxFaces=2)


while True:
    success, img = cap.read()
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

        # # Finding the focal length of the camera
        # W = 6.3  # cm for male person in front of camera
        # d = 50
        # f = (w*d)/W
        # if f > 600:
        #     print("lost")
        # else:
        #     print("good")
        # print(f)

        ### Finding the distance ###
        f = 840
        d = (W*f)/w
        if d < 90:
            print("Go Backward")
            cvzone.putTextRect(img, f'Go Backward',
                               (face[10][0] - 100, face[10][1] - 100), scale=2)
        else:
            print("Perfect position")
            cvzone.putTextRect(img, f'   Perfect   ',
                               (face[10][0] - 100, face[10][1] - 100), scale=2)
        print(d)

        cvzone.putTextRect(img, f'Depth: {int(d)}cm',
                           (face[10][0] - 100, face[10][1] - 50),
                           scale=2)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
