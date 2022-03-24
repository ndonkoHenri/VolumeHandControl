import cv2 as cv
import cvzone, numpy as np
from cvzone import HandTrackingModule
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

hand_detect = HandTrackingModule.HandDetector(maxHands=1, detectionCon=0.7)
cap = cv.VideoCapture(0)
fpsReader = cvzone.FPS()

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
# print(volume.GetVolumeRange())      # The volume Range Ex: (-96.0, 0.0, 1.5)
# volume.SetMasterVolumeLevel(-20.0, None)


while True:
    success, image = cap.read()  # Reading of the video capture
    image = cv.flip(image, 1)
    all_hands, image = hand_detect.findHands(image, flipType=False)
    minVol = volume.GetVolumeRange()[0]
    maxVol = volume.GetVolumeRange()[1]

    if all_hands:
        thumb_landmarks_info = all_hands[0]["lmList"][4]
        index_landmarks_info = all_hands[0]["lmList"][8]
        length, info = hand_detect.findDistance((thumb_landmarks_info[0], thumb_landmarks_info[1]),
                                                (index_landmarks_info[0], index_landmarks_info[1]))

        x1, y1, x2, y2, cx, cy = info
        cv.circle(image, (x1, y1), 11, (255, 0, 255), cv.FILLED)
        cv.circle(image, (x2, y2), 11, (255, 0, 255), cv.FILLED)
        cv.line(image, (x1, y1), (x2, y2), (255, 0, 255), 2)
        volumeValue = np.interp(length, [35, 140], [minVol, maxVol])
        volumeBar = np.interp(length, [35, 140], [150, 400])
        if length < 35:
            cv.circle(image, (cx, cy), 11, (0, 255, 0), cv.FILLED)
            volume.SetMasterVolumeLevel(minVol, None)
        else:
            cv.circle(image, (cx, cy), 15, (255, 0, 255), cv.FILLED)
            volume.SetMasterVolumeLevel(volumeValue, None)
    # print(type(volume.GetMasterVolumeLevel()))
    volumeBar = np.interp(volume.GetMasterVolumeLevel(), [minVol, maxVol], [400, 150])
    volumePercent = np.interp(volumeBar, [150, 400], [100, 0])
    cv.putText(image,f"Vol = {int(volumePercent)}%",(33,142),cv.FONT_HERSHEY_PLAIN,1.8,(0,255,255),2)
    cv.rectangle(image, (50, 150), (85, 400), (20, 19, 255), 3)
    cv.rectangle(image, (50, int(volumeBar)), (85, 400), (200, 200, 0), cv.FILLED)
    fps, image = fpsReader.update(image)
    cv.imshow("HandLandmarks", image)
    if cv.waitKey(1) & 0xFF == ord('q'):  # If "q" is pressed
        break  # Breaks and stops
"""cap.release()
cv.destroyAllWindows()  # Destroys all opened windows"""
