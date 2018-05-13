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
train_text_file = open(r"C:\Users\adith\Desktop\Meng Robotics\Bio inspired robotics\RoboPython-mybranch01\old\traindata_set_100.txt", "r")
test_text_file = open(r"C:\Users\adith\Desktop\Meng Robotics\Bio inspired robotics\RoboPython-mybranch01\old\traindata_set_temp.txt", "r")
lines_train = train_text_file.readlines()
lines_test = test_text_file.readlines()
count=0
for line_train in lines_train:
    count=count+1
    line_train = line_train.rstrip('\n')
    line_train = line_train.split()
    if(count<=100):
        posarray.append(line_train)
    else:
        angarray.append(line_train)
count=0
for line_test in lines_test:
    count=count+1
    line_test = line_test.rstrip('\n')
    line_test = line_test.split()
    if(count<=10):
        posarray_test.append(line_test)
    else:
        angarray_test.append(line_test)

for pos in posarray:
    count = 0
    for p in pos:
        position = float(p)
        count = count + 1
        if (count == 1):
            position = position / 300
        if (count == 2):
            position = (position + 300) / 600
        if (count == 3):
            position = (position + 150) / 300
            count = 0
        train_x.append(position)

for pos in posarray_test:
    count=0
    for p in pos:
        position=float(p)
        count=count+1
        if(count==1):
            position=position/300
        if(count==2):
            position=(position+300)/600
        if (count == 3):
            position = (position + 150) / 300
            count=0
        test_x.append(position)

for ang in angarray:
    for a in ang:
        angles=float(a)
        print("ohh my god", angles/180)
        train_y.append(angles/180)
    #     newbitang.append(bin(struct.unpack('!i', struct.pack('!f', angles/180))[0]))
    #     aaa = newbitang[-1]
    #     bbb = aaa.replace("b","")   #code to encode the o/p angles to bits as per IEEE 32 bit method
    #     bbb = list(bbb)
    #     for b in bbb:
    #         b = int(b)
    #         checkint.append(b)
    #     train_y.append(checkint)
    #     checkint=[]
#
for ang in angarray_test:
    for a in ang:
        angles=float(a)
        test_y.append(angles / 180)
#         newbitang_test.append(bin(struct.unpack('!i', struct.pack('!f', angles/180))[0]))
#         aaa = newbitang_test[-1]
#         bbb_test = aaa.replace("b","")
#         bbb_test = list(bbb_test)
#         for b in bbb_test:
#             b = int(b)
#             checkint_test.append(b)   #code to encode the o/p angles to bits as per IEEE 32 bit method
#         test_y.append(checkint_test)
#         checkint_test=[]

print("train angles",test_y)
print("test angles ",train_y)
print("postion test",test_x)

print("position train",train_x)
# print("vector length",len(train_y),len(train_x),len(test_y),len(test_x))

train_x = np.asarray(train_x)
train_x=np.reshape(train_x,(100,3))
train_y = np.asarray(train_y)
train_y=np.reshape(train_y,(100,3))
test_x = np.asarray(test_x)
test_x=np.reshape(test_x,(10,3))
test_y = np.asarray(test_y)
test_y=np.reshape(test_y,(10,3))

print("np train angles",test_y)
print("np test angles ",train_y)
print("postion test",test_x)

print("position train",train_x)

# print("np train angles",len(train_y[0]))
# print("np test angles ",test_y)
# print("np postion test",test_x)
#
# print("np position train",len(train_x[0]))

def moveRandom():

    SERVO_BOTTOM = 0
    SERVO_LEFT = 1
    SERVO_RIGHT = 2
    SERVO_HAND = 3
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
    return angles,pixel_position





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

#
#
#
#
#training starts hereeee  hold on babyyyyyyyyyy!!!!!!!!!
#
#
#
#
no_nodes_hl1=2
no_nodes_hl2=2
no_nodes_hl3=2

n_classes=3
batch_size = 50

x=tf.placeholder('float',[None,len(train_x[0])])
y=tf.placeholder('float')

def Neural_Network_model(data) :
    hidden_Layer_1={'weights': tf.Variable(tf.random_normal([len(train_x[0]),no_nodes_hl1]))}

    hidden_Layer_2={'weights':tf.Variable(tf.random_normal([no_nodes_hl1,no_nodes_hl2]))}

    hidden_Layer_3 = {'weights': tf.Variable(tf.random_normal([no_nodes_hl2, no_nodes_hl3]))}

    output_Layer = {'weights': tf.Variable(tf.random_normal([no_nodes_hl3, n_classes]))}

    # layer1 = tf.add(tf.matmul(data,hidden_Layer_1['weights']),hidden_Layer_1['biases'] )
    # layer1 = tf.nn.sigmoid(layer1)
    # layer2 = tf.add(tf.matmul(layer1, hidden_Layer_2['weights']), hidden_Layer_2['biases'])
    # layer2 = tf.nn.sigmoid(layer2)
    # layer3 = tf.add(tf.matmul(layer2, hidden_Layer_3['weights']),hidden_Layer_3['biases'])
    # layer3 = tf.nn.sigmoid(layer3) #with bias for all layers

    layer1 = tf.matmul(data, hidden_Layer_1['weights'])
    layer1 = tf.nn.sigmoid(layer1)
    layer2 = tf.matmul(layer1, hidden_Layer_2['weights'])
    layer2 = tf.nn.sigmoid(layer2)
    layer3 = tf.matmul(layer2, hidden_Layer_3['weights'])
    layer3 = tf.nn.sigmoid(layer3)
    # without bias for all layers - sigmoid function

    output= tf.matmul(layer3, output_Layer['weights']) #+ output_Layer['biases'] removed bias
    output = tf.nn.sigmoid(output)

    return output

def train_neural_networks(x):
    predict = Neural_Network_model(x)
    cost= tf.reduce_mean(tf.pow(predict-y,2), name='loss' )
    optimize = tf.train.GradientDescentOptimizer(0.01).minimize(cost)  # training seems to be dreasing
    Total_epoch=1000
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(Total_epoch):
            epoch_loss=0
        # code for coverting the data to batch wise and giving it to the neural network
            i=0
            while i<len(train_x):
                start=i
                end =i+batch_size

                batch_x=np.array(train_x[start:end])
                batch_y=np.array(train_y[start:end])
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
