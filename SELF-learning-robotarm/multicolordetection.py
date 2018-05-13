import cv2
import numpy as np
import imutils
from time import sleep
import keyboard as key
cap = cv2.VideoCapture(1)
#cap1 = cv2.VideoCapture(1)


def detect_object(v, w,a,b,c,d,e,f,z):
    _, frame = v.read(c)
    frame = imutils.resize(frame, width=w)
    lower_color = np.array([a,b,c])
    upper_color = np.array([d,e,f])
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    kernel = np.ones((11,11),np.float32)/225
    mask = cv2.inRange(hsv, lower_color, upper_color)
    mask1 = cv2.inRange(hsv, lower_color, upper_color)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    output = cv2.bitwise_and(frame, frame, mask = mask1)

    kernel = np.ones((11,11),np.float32)/225

    red_center = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)[-2]
    center_red = None


    if len(red_center) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(red_center, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        x_red = int(M["m10"] / M["m00"])
        y_red = int(M["m01"] / M["m00"])

       # sc = str("x-red:",x_red, "y_red:",y_red)
        # only proceed if the radius meets a minimum size. Correct this value for your obect's size
        if radius > 0.5:
            # draw the circle and centroid on the frame,
            # then update the list of tracked points
            cv2.circle(frame, (int(x), int(y)), int(radius), (255,0,0), 2)
            #cv2.putText(frame, sc, (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0,255,0),2)
            print(z, x_red, y_red)

    cv2.imshow("frame", output)
    return output
a=50
b=50
c=50
d=50
count=0
while True:
    cv2.imshow("green", detect_object(cap, 600, a  ,b, c, d ,255, 255, "green"))
    #cv2.imshow("blue",detect_object(cap1, 600,97, 100, 117,117,255,255,"blue"))
    #[a(q...e)    b (a.....d)   c  (z ......c)   d (r.......y)]
    if key.is_pressed('q'):  # if key 'q' is pressed
        a=a+1
        print("value of a",a)
        sleep(0.2)
    if key.is_pressed('e'):  # if key 'q' is pressed
        a=a-1
        print("value of a",a)
        sleep(0.2)
    if key.is_pressed('a'):  # if key 'q' is pressed
        b=b+1
        print("value of b",b)
        sleep(0.2)
    if key.is_pressed('d'):  # if key 'q' is pressed
        b = b - 1
        print("value of b", b)
        sleep(0.2)
    if key.is_pressed('z'):  # if key 'q' is pressed
        c=c+1
        print("value of c",c)
        sleep(0.2)
    if key.is_pressed('c'):  # if key 'q' is pressed
        c=c-1
        sleep(0.2)
        print("value of c",c)
    if key.is_pressed('r'):  # if key 'q' is pressed
        d=d+1
        sleep(0.2)
        print("value of d",d)
    if key.is_pressed('y'):  # if key 'q' is pressed
        d = d - 1
        print("value of d", d)
        sleep(0.2)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()
cap.release
