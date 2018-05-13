import sys
sys.path.append('C:\python64\Lib\site-packages')
import tensorflow as tf
import numpy as np
import struct
from fuzzywuzzy import fuzz


import time
from time import sleep
import cv2
from uf.wrapper.swift_api import SwiftAPI
from uf.utils.log import *
logger_init(logging.INFO)
import imutils

#array intialization
train_x=[]
test_x=[]
newang=[]
posarray=[]
angarray=[]
posarray_test=[]
angarray_test=[]
newbitang=[]
removebit=[]
bit=[]

newbitang_test=[]
removebit_test=[]
bit_test=[]

train_y=[]
test_y=[]
checkint=[]
checkint_test=[]


def moveRandom():

    SERVO_BOTTOM = 0
    SERVO_LEFT = 1
    SERVO_RIGHT = 2
    position_arr=[]
    pixel_position=[]
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
    cap = cv2.VideoCapture(0)
    cap1 = cv2.VideoCapture(0)
    red = detect_w_avg(cap, 600, 166, 84, 80, 286, 255, 255, "red")
    print("red", red)
    blue = detect_w_avg(cap1, 600, 97, 100, 117, 117, 255, 255, "blue")
    print("blue", blue)
    pixel_position.append(red)
    pixel_position.append(blue)
    cv2.destroyAllWindows()
    cap.release()
    cap1.release()
    # pixelpostion= get the pixel position here

    print("the position is ", pos, "\n")
    position_arr.append(pos)
    angles = angles.tolist()
    angles_arr.append(angles)  # [right bottom left]
    sleep(2.0)
    angles = np.array(angles)
    pixel_position = np.array(pixel_position)
    return angles,pixel_position



def moveAndCheck(angles):

    SERVO_BOTTOM = 0
    SERVO_LEFT = 1
    SERVO_RIGHT = 2
    position_arr=[]
    pixel_position=[]
    angles_arr=[]

    arm = SwiftAPI()
    sleep(2.0)

    print("angles are", angles)

    arm.flush_cmd()
    arm.reset()

    arm.set_servo_angle_speed(SERVO_RIGHT, angles[0], wait=True, timeout=100, speed=5000)

    arm.set_servo_angle_speed(SERVO_BOTTOM, angles[1], wait=True, timeout=100, speed=5000)

    arm.set_servo_angle_speed(SERVO_LEFT, angles[2], wait=True, timeout=100, speed=5000)
    a = arm.get_is_moving()
    print("the status is", a, "\n")
    pos = arm.get_position()  # float array of the format [x, y, z] of the robots current location
    cap = cv2.VideoCapture(0)
    cap1 = cv2.VideoCapture(0)
    red = detect_w_avg(cap, 600, 166, 84, 80, 286, 255, 255, "red")
    print("red", red)
    blue = detect_w_avg(cap1, 600, 97, 100, 117, 117, 255, 255, "blue")
    print("blue", blue)
    pixel_position.append(red)
    pixel_position.append(blue)
    cv2.destroyAllWindows()
    cap.release()
    cap1.release()
    # pixelpostion= get the pixel position here

    print("the position is ", pos, "\n")
    position_arr.append(pos)
    angles = angles.tolist()
    angles_arr.append(angles)  # [right bottom left]
    sleep(2.0)
    pixel_position=np.array(pixel_position)

    return pixel_position
















class find:
    def detect_object(self, v, w,a,b,c,d,e,f,z):
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
        kernel = np.ones((11,11),np.float32)/225
        red_center = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)[-2]
        center_red = None
        xc = 0
        yc = 0
        if len(red_center) > 0:
            c = max(red_center, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            xc = int(M["m10"] / M["m00"])
            yc = int(M["m01"] / M["m00"])

        self.color = z
        self.xcoord = xc
        self.ycoord = yc
        self.maskoutput = mask1
    def getcoordx(self):
        return self.xcoord
    def getcoordy(self):
        return self.ycoord
    def getcolor(self):
        return self.color
    def getOuput(self):
        return self.maskoutput


def detect_w_avg(s,w,a,b,c,d,e,f,color):
    i = 0
    xred = 0
    yred = 0
    while i < 10:
        red = find()
        red.detect_object(s, w,a,b,c,d,e,f,color)
        print(red.getcoordx(), red.getcoordy())
        xred = xred + red.getcoordx()
        yred = yred + red.getcoordy()
        i += 1
    #print((xred/10),(yred/10))
    return (color,(xred/10), (yred/10))
    cv2.imshow(color, red.getOuput())






def Test_Neural_Network_model(wat) :
    counto=0;
    for w in wat:
        print("ohmm",w,"count",counto)
        counto=counto+1
        if(isinstance(w,str)==1):
            del wat[counto-1]
    hidden_Layer_1 = wat[0]
    output_Layer = wat[1]
    layer1=np.zeros((1,no_nodes_hl1))
    i=0
    print("test data",test_x[0])
    while i< no_nodes_hl1:
        print("values" ,hidden_Layer_1[:,i])
        layer1[:,i]= (np.dot(test_x[2], hidden_Layer_1[:,i] ))
        i=i+1

    layer1 = 1 / (1 + np.exp(-(layer1)))
    j=0
    output=np.zeros((1,n_classes))
    while j < n_classes:
        output[:,j]= (np.dot(layer1,output_Layer[:,j]))
        j=j+1
    print("op",output)
    output = 1 / (1 + np.exp(-(output)))
    print("exp op", output)
    print("expected angles",output *180)
    output = moveAndCheck(output*180) #to check with vision data
    return output

no_nodes_hl1=2
no_nodes_hl2=2
no_nodes_hl3=2

n_classes=3
batch_size = 1

train_y,train_x=moveRandom()
x=tf.placeholder('float',[None,len(train_x)])

y=tf.placeholder('float')

def Neural_Network_model(data) :
    hidden_Layer_1={'weights': tf.Variable(tf.random_normal([len(train_x),no_nodes_hl1]))}
    output_Layer = {'weights': tf.Variable(tf.random_normal([no_nodes_hl1, n_classes]))}
    layer1 = tf.matmul(data, hidden_Layer_1['weights'])
    layer1 = tf.nn.sigmoid(layer1)
    output= tf.matmul(layer1, output_Layer['weights'])
    output = tf.nn.sigmoid(output)
    output = moveAndCheck(output * 180)
    # get the co-ord usinng vision data return that as predict moveAndCheck(output)
    return output

def train_neural_networks(x):
    predict = Neural_Network_model(x)
    #cost= tf.reduce_mean(tf.pow(predict-y,2), name='loss' )
    cp = (predict-y)**2
    cost = (cp[0,0]+cp[0,1])**(0.5)+(cp[1,0]+cp[1,1])**(0.5) # distance bettween the desired and actucal pt in top view + front view is the cost
    #new cost calculation based on the vision
    optimize = tf.train.GradientDescentOptimizer(0.01).minimize(cost)  # training seems to be dreasing
    Total_epoch=1000
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(Total_epoch):
            epoch_loss=0
        # code for coverting the data to batch wise and giving it to the neural network
            i=0
            while True:
                start=i
                end =i+batch_size
                train_y,train_x=moveRandom()
                batch_x=np.array(train_x)
                batch_y=np.array(train_x)
                ohmm,c=sess.run([optimize,cost], feed_dict={x : batch_x , y : batch_y})
                print("cost",c)
                epoch_loss += c
                i+=batch_size
                variables_names = [v.name for v in tf.trainable_variables()]
            values = sess.run(variables_names)

            print("the epoch",epoch , "loss is",epoch_loss)
            correct = tf.pow(predict-y,2)
            accuracy = tf.reduce_mean(tf.cast(correct,'float'))
            print("the accuracy: ",accuracy.eval({x: test_x , y : test_y}))
            for val in tf.trainable_variables():
                    print("the weights :", val)
            for k, v in zip(variables_names, values):
                print("weights ", k, v)
            if(epoch == 999):
                weights=[]
                for k, v in zip(variables_names, values):
                        print("aeeeeee ethyam", k, v)
                        weights.append(k)
                        weights.append(v)

    return weights
                    # oh = tf.get_variable("hidden_Layer_2", [1])
                    # print(oh)
w=train_neural_networks(x)
print("out",w)
testmega = Test_Neural_Network_model(w)
print(testmega)
