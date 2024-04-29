import numpy as np
import cv2
import time
import mss
import pyautogui
pyautogui.PAUSE = 0
time.sleep(1)
print("start")
def my_filled_circle(img, center):
    thickness = -1
    line_type = 8
    cv2.circle(img,
               center,
               100 // 32,
               (0, 0, 255),
               thickness,
               line_type)

bounding_box = {'top': 40, 'left': 0, 'width': 490, 'height': 900}
sct = mss.mss()
go_right = True
score = 0

while True:
    sct_img = sct.grab(bounding_box)
    screen = np.array(sct_img)
    screen = cv2.subtract(screen, 50)
    screen2 = np.copy(screen)
    screen = screen[400:470, :]
    s_hsv = cv2.cvtColor(screen, cv2.COLOR_RGB2HSV)
    s_hsv2 = cv2.cvtColor(screen2, cv2.COLOR_RGB2HSV)
    s_rgb = cv2.cvtColor(screen , cv2.COLOR_BGR2RGB)

    hsv_lower = np.array([145, 150, 170])
    hsv_upper = np.array([155, 250, 230])


    mask1 = cv2.inRange(s_hsv, hsv_lower, hsv_upper)
    canny_lines = cv2.Canny(screen, 98, 30)
    ker =  np.array([[1,1,1],
            [1,1,1],
            [1,1,1]],np.uint8)
    mask1 = cv2.dilate(mask1, ker, iterations=3)



    canny_lines[mask1>0] = 0
    rho =1
    theta = np.pi/180
    threshold = 16
    min_line_length = 26
    max_line_gap = 3
    
    
    
    lines = cv2.HoughLinesP(canny_lines, rho,theta, threshold, np.array([]), min_line_length, max_line_gap)
    
    s_gray = cv2.cvtColor(s_hsv, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(s_gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=18, minRadius=10, maxRadius=14)
    y_screen = int(screen.shape[0])
    x_screen = int(screen.shape[1])
    black_screen = np.zeros((y_screen,x_screen))
  
    if lines is not None:
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(black_screen, (x1,y1), (x2,y2), (255,255,255), 5)
                cv2.line(screen, (x1,y1), (x2,y2), (0,255,0), 5)
    if circles is not None:
        circles = np.uint16(np.around(circles))  
        count=0
        for circle in circles[0, :]:
            
            x_c=circle[0]+3
            y_c=31
            r = circle[2]  

            Far = 48
            near = 20
          
            my_filled_circle(screen,(x_c,y_c))
            
            if np.sum(black_screen[y_c, x_c + r + near : x_c + Far]) > 0 and go_right:
                pyautogui.press('space')
                time.sleep(0.01)
                go_right=False
            elif np.sum(black_screen[y_c, x_c - Far : x_c - r - near  ]) > 0 and not go_right:
                pyautogui.press('space')
                time.sleep(0.01)
                go_right=True
            elif np.sum(black_screen[y_c-3, x_c + r + near : x_c + Far]) > 0 and go_right:
                pyautogui.press('space')
                time.sleep(0.01)
                go_right=False
       
           
     
            
    











    cv2.imshow('screen',screen)
    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        filename = 'savedImage.jpg'
        cv2.imwrite(filename, screen2) 
        break