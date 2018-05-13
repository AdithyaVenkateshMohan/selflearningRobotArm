import sys

sys.path.append('C:\python64\Lib\site-packages')

import numpy as np
import time
from time import sleep
import cv2
from uf.wrapper.swift_api import SwiftAPI
from uf.utils.log import *

logger_init(logging.INFO)

SERVO_BOTTOM = 0
SERVO_LEFT = 1
SERVO_RIGHT = 2
SERVO_HAND = 3

arm = SwiftAPI()
sleep(2.0)
position_arr = []

angles_arr = []
'''
       Set servo angle, 0 - 180 degrees, this Function will include the manual servo offset.

       Args:
           servo_id: SERVO_BOTTOM, SERVO_LEFT, SERVO_RIGHT, SERVO_HAND
           angle: 0 - 180 degrees
           wait: if True, will block the thread, until get response or timeout

       Returns:

          succeed True or failed False
       '''

arm.flush_cmd()
arm.reset()

for _ in range(100):
    rand = np.random.rand(2)
    x = rand[0]*300
    y = rand[1]*600
    y=y-300
    z_const = 100
    print("aimed positions are",x,y,z_const)
    arm.set_position(x= x,y= y,z=z_const,speed=4000,wait=True,timeout=100)
    a = arm.get_is_moving()
    print("the status is", a, "\n")
    pos = arm.get_position()  # float array of the format [x, y, z] of the robots current location
    print("x values " , pos[0])
    print("y values" , pos[1])
    print("z values", pos [2])
    if(pos[2]!= z_const):
        print("invalid data height misconception")
        sys.stdout.write('/a')
        sys.stdout.flush()
        sleep(2.0)
        continue
    else:
        print("valid data with desired height")
        servo_angles=arm.get_servo_angle()
        print("angles all", servo_angles)
        servo_angles=servo_angles[0:3]
        print("angles desired", servo_angles)
        print("the position is ", pos, "\n")
        position_arr.append(pos)
        sleep(2.0)
        angles_arr.append(servo_angles)


