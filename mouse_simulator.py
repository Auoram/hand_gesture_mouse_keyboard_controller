import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time

cam = cv2.VideoCapture(0)
w_screen,h_screen = pyautogui.size()
handM = mp.solutions.hands
hands = handM.Hands()
draw = mp.solutions.drawing_utils
prev_x,prev_y=None,None
clicked,rClicked=False,False
prev_scroll = None
gFlag,sFlag = False,False
while True:
    success,frame = cam.read()
    if not success:
        break
    frame = cv2.flip(frame,1)
    mp_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    res = hands.process(mp_rgb)

    if res.multi_hand_landmarks:
        for land in res.multi_hand_landmarks:
            draw.draw_landmarks(frame,land,handM.HAND_CONNECTIONS)
            landmarks = []
            for id,lm in enumerate(land.landmark):
                frame_h, frame_w,c = frame.shape
                cx , cy = int(lm.x*frame_w),int(lm.y*frame_h)
                landmarks.append((cx,cy))
            # This is for mouse moving
            tips = [4,8,12,16,20]
            ix,iy = landmarks[8]
            screen_x = np.interp(ix,[0,frame_w],[0,w_screen])
            screen_y = np.interp(iy,[0,frame_h],[0,h_screen])
            if prev_x is None or prev_y is None:
                prev_x, prev_y = screen_x, screen_y
            smooth_x = prev_x + (screen_x - prev_x) / 5
            smooth_y = prev_y + (screen_y - prev_y) / 5
            pyautogui.moveTo(smooth_x,smooth_y)
            prev_x,prev_y=smooth_x,smooth_y
            # This is for click
            tx,ty=landmarks[4]
            dist_left = np.hypot(tx-ix,ty-iy)
            if dist_left < 40:
                if not clicked:
                    pyautogui.click()
                    clicked = True
            else:
                clicked = False
            # This is for right click
            mx,my=landmarks[12]
            dist_right = np.hypot(ix-mx,iy-my)
            if dist_right < 40:
                if not rClicked:
                    pyautogui.rightClick()
                    rClicked = True
            else:
                rClicked = False
            # This is for scrolling
            if landmarks[8][1] < landmarks[6][1] and landmarks[12][1] < landmarks[10][1]:
                    avg = (landmarks[12][1] + landmarks[8][1]) / 2
                    if prev_scroll is None:
                        prev_scroll = avg
                    movement = prev_scroll - avg
                    if not sFlag:
                        if movement > 10:
                            pyautogui.scroll(50)
                            sFlag=True
                        elif movement < -10:
                            pyautogui.scroll(-50)
                            sFlag = True
                    prev_scroll = avg
            else:
                sFlag = False
                prev_scroll = None
            fingUp = [
                landmarks[8][1] < landmarks[6][1],
                landmarks[12][1] < landmarks[10][1],
                landmarks[16][1] < landmarks[14][1],
                landmarks[20][1] < landmarks[18][1]
            ]
            # This is for copy
            if fingUp[0] and fingUp[1] and not fingUp[2] and not fingUp[3]:
                if not gFlag:
                    pyautogui.hotkey('ctrl','c')
                    gFlag = True
            # This is for paste
            elif not any(fingUp):
                if not gFlag:
                    pyautogui.hotkey('ctrl','v')
                    gFlag = True
            # This is for open chrome
            elif all(fingUp):
                if not gFlag:
                    pyautogui.hotkey('win','r')
                    time.sleep(0.1)
                    pyautogui.write('chrome')
                    pyautogui.press('enter')
                    gFlag = True
            else:
                gFlag = False

    cv2.imshow("Mouse simulator",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()