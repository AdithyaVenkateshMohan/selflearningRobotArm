import sys
sys.path.append('C:\python64\Lib\site-packages')

import numpy as np
import time
from time import sleep
import cv2
from uf.wrapper.swift_api import SwiftAPI
from uf.utils.log import *
logger_init(logging.INFO)

def moveRandom():

    SERVO_BOTTOM = 0
    SERVO_LEFT = 1
    SERVO_RIGHT = 2
    SERVO_HAND = 3
    position_arr=[]

    angles_arr=[]

    arm = SwiftAPI()
    sleep(2.0)
    rand = np.random.rand(3)
    angles = rand * 180

    print("angles are", angles)

    arm.flush_cmd()
    arm.reset()

    arm.set_servo_angle_speed(SERVO_RIGHT, angles[0], wait=True, timeout=100, speed=5000)

    arm.set_servo_angle_speed(SERVO_BOTTOM, angles[1], wait=True, timeout=100, speed=5000)

    arm.set_servo_angle_speed(SERVO_LEFT, angles[2], wait=True, timeout=100, speed=5000)
    a = arm.get_is_moving()
    print("the status is", a, "\n")
    pos = arm.get_position()  # float array of the format [x, y, z] of the robots current location
    # pixelpostion= get the pixel position here

    print("the position is ", pos, "\n")
    position_arr.append(pos)
    angles = angles.tolist()
    angles_arr.append(angles)  # [right bottom left]
    sleep(2.0)

def Test_Neural_Network_model(wat) :
    counto=0;
    for w in wat:
        print("ohmm",w,"count",counto)
        counto=counto+1
        if(isinstance(w,str)==1):
            del wat[counto-1]

    print("wat",wat)
    hidden_Layer_1 = wat[0]
    hidden_Layer_2 = wat[1]
    hidden_Layer_3 =wat[2]
    output_Layer = wat[3]
    print("hl1",hidden_Layer_1)
    print("hl2", hidden_Layer_2)
    print("hl3", hidden_Layer_3)
    print("hl4", output_Layer)

    layer1=np.zeros((1,no_nodes_hl1))
    layer2=np.zeros((1,no_nodes_hl2))
    layer3=np.zeros((1,no_nodes_hl3))
    i=0
    print("test data",test_x[0])
    while i< no_nodes_hl1:
        print("values" ,hidden_Layer_1[:,i])
        layer1[:,i]= (np.dot(test_x[2], hidden_Layer_1[:,i] ))
        i=i+1

    layer1 = 1 / (1 + np.exp(-(layer1)))
    i=0
    print("layer1",layer1)
    print("weights layer 2",hidden_Layer_2)
    while i < no_nodes_hl2:
        print("values layer 2" ,hidden_Layer_2[:,i])
        layer2[:,i]=(np.dot(layer1,hidden_Layer_2[:,i]))
        i=i+1
    layer2 = 1/(1+np.exp(-(layer2)))
    j=0
    print("layer2", layer2)
    print("weights layer 2", hidden_Layer_3)
    while j < no_nodes_hl3:
        print("values layer 3", hidden_Layer_3[:, j])
        layer3[:,j] = np.dot(layer2,hidden_Layer_3[:,j])
        j=j+1
    layer3 = 1 / (1 + np.exp(-(layer3)))
    j=0
    output=np.zeros((1,n_classes))
    while j < n_classes:
        output[:,j]= (np.dot(layer3,output_Layer[:,j]))
        j=j+1
    print("op",output)
    output = 1 / (1 + np.exp(-(output)))
    print("exp op", output)
    print("expected angles",output *180)
    return output


def Train_Neural_Network_model_feedforward(wat) :
    no_nodes_hl1=2
    n_classes=3
    print("wat",wat)
    hidden_Layer_1 = wat[0]
    hidden_Layer_2 = wat[1]
    hidden_Layer_3 =wat[2]
    output_Layer = wat[3]
    print("hl1",hidden_Layer_1)
    print("hl2", hidden_Layer_2)
    print("hl3", hidden_Layer_3)
    print("hl4", output_Layer)

    layer1=np.zeros((1,no_nodes_hl1))
    i=0
    print("test data",test_x[0])
    while i< no_nodes_hl1:
        print("values" ,hidden_Layer_1[:,i])
        layer1[:,i]= (np.dot(test_x[2], hidden_Layer_1[:,i] ))
        i=i+1

    layer1 = 1 / (1 + np.exp(-(layer1)))
    i=0
    print("layer1",layer1)
    print("weights layer 2",hidden_Layer_2)

    j=0
    output=np.zeros((1,n_classes))
    while j < n_classes:
        output[:,j]= (np.dot(layer1,output_Layer[:,j]))
        j=j+1
    print("op",output)
    output = 1 / (1 + np.exp(-(output)))
    print("exp op", output)
    print("expected angles",output *180)
    layerdash = np.multiply(layer1,(1-layer1))
    outputdash=np.multiply(output,(1-output))
    error = calulateError()
    for x in range(n_classes):
        for y in range(no_nodes_hl1):

    return output


