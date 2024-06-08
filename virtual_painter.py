import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector

width, height = 1280, 720
cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

detector = HandDetector(detectionCon=0.8, maxHands=1)

# Variables
color_code = [(120, 110, 0), (168, 168, 168), (159, 53, 4), (255, 81, 0), (151, 255, 0),
              (252, 245, 95), (59, 181, 3), (3, 193, 231), (0, 0, 255), (192, 1, 201), (0, 0, 0),
              (255, 255, 255), (209, 209, 209), (203, 131, 20), (244, 163, 192), (163, 216, 244)]
color = (0, 0, 0)
xsel, ysel = 0, 0
xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img, flipType=False)

    # Placing color boxes in image
    xacc = 0
    for i in range(16):
        if i != 10:
            cv2.rectangle(img, (xacc, 0), ((xacc+80), 80), color_code[i], -1)
        else:
            cv2.rectangle(img, (820, 20), (860, 60), (0, 0, 0), 2)
        xacc = xacc+80

    # Correcting thumb hand
    if hands:
        hand = hands[0]
        fingers = detector.fingersUp(hand)
        if fingers[0] == 0:
            fingers[0] = 1
        else:
            fingers[0] = 0

    # Tracing Index and Middle Finger
    lmlist = detector.lmList
    if len(lmlist) > 0:
        if fingers[0] == 0 and fingers[1] == 1 and \
                fingers[2] == 1 and fingers[3] == 0 and fingers[4] == 0:
            xp, yp = 0, 0
            x1, y1 = lmlist[8][0], lmlist[8][1]
            x2, y2 = lmlist[12][0], lmlist[12][1]
            length, info = detector.findDistance((x1, y1),
                                                 (x2, y2))
            x3, y3 = (x1 + x2) / 2, (y1 + y2) / 2
            x3, y3 = int(x3), int(y3)
            cv2.circle(img, (x3, y3), 10, (255, 0, 0), 2)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
            length, info = detector.findDistance((x1, y1), (x2, y2))

            # Selection Mode
            if length < 40:
                xst = 0
                for i in range(16):
                    if (x3 < (xst+80)) and (x3 > xst) and (y3 < 100):
                        color = color_code[i]
                        cv2.rectangle(img, (xst, 0), ((xst + 80), 80), (255, 255, 255), -1)
                        xsel = xst
                        break
                    xst = xst + 80
        # Drawing Mode
        elif fingers[0] == 0 and fingers[1] == 1 and \
                fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 0:
            x1, y1 = lmlist[8][0], lmlist[8][1]
            cv2.circle(img, (x1, y1), 10, (255, 0, 0), cv2.FILLED)
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if color == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), color, 10)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), color, 10)
            else:
                cv2.line(img, (xp, yp), (x1, y1), color, 2)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), color, 2)
            xp, yp = x1, y1
    cv2.rectangle(img, (xsel, 0), ((xsel + 80), 80), (0, 0, 0), 2)

    # Image merging with canvas
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    # Showing of Image
    cv2.imshow("Video", img)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
