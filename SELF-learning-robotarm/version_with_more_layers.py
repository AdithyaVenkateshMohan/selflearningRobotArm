import sys
sys.path.append('C:\python64\Lib\site-packages')
import tensorflow as tf
import numpy as np
import struct
from fuzzywuzzy import fuzz

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
train_text_file = open(r"C:\Users\adith\Desktop\Meng Robotics\Bio inspired robotics\RoboPython-mybranch01\old\traindata_set_vision100.txt", "r")
test_text_file = open(r"C:\Users\adith\Desktop\Meng Robotics\Bio inspired robotics\RoboPython-mybranch01\old\vision_data.txt", "r")
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
        if(p != 0):
            train_x.append(position)

for pos in posarray_test:
    count=0
    for p in pos:
        position=float(p)
        if (p != 0):
            test_x.append(position)

for ang in angarray:
    for a in ang:
        angles=float(a)
        print("ohh my god", angles/180)
        train_y.append(angles/180)

for ang in angarray_test:
    for a in ang:
        angles=float(a)
        test_y.append(angles / 180)


print("train angles",test_y)
print("test angles ",train_y)
print("postion test",test_x)

print("position train",train_x,"len",len(train_x))
train_x = np.asarray(train_x)
test_x = np.asarray(test_x)
train_x=train_x[train_x!=0]
test_x=test_x[test_x!=0]
print("postion test",test_x)

print("position train",train_x,"len",len(train_x))
# print("vector length",len(train_y),len(train_x),len(test_y),len(test_x))


train_x=np.reshape(train_x,(95,2))
train_y = np.asarray(train_y)
train_y=np.reshape(train_y,(100,3))

test_x=np.reshape(test_x,(10,2))
test_y = np.asarray(test_y)
test_y=np.reshape(test_y,(10,3))

print("np train angles",test_y)
print("np test angles ",train_y)
print("postion test",test_x)

print("position train",train_x)
def Test_Neural_Network_model(wat) :
    counto=0;
    for w in wat:
        print("ohmm",w,"count",counto)
        counto=counto+1
        if(isinstance(w,str)==1):
            del wat[counto-1]
    hidden_Layer_1 = wat[0]
    hidden_Layer_2 = wat[1]
    output_Layer = wat[2]
    layer1=np.zeros((1,no_nodes_hl1))
    layer2 = np.zeros((1, no_nodes_hl2))
    i=0
    print("test data",test_x[0])
    while i< no_nodes_hl1:
        print("values" ,hidden_Layer_1[:,i])
        layer1[:,i]= (np.dot(test_x[0], hidden_Layer_1[:,i] ))
        i=i+1

    layer1 = 1 / (1 + np.exp(-(layer1)))

    i = 0
    while i < no_nodes_hl2:
        print("values", hidden_Layer_2[:, i])
        layer2[:, i] = (np.dot(layer1, hidden_Layer_2[:, i]))
        i = i + 1

    layer2 = 1 / (1 + np.exp(-(layer2)))

    j=0
    output=np.zeros((1,n_classes))
    while j < n_classes:
        output[:,j]= (np.dot(layer2,output_Layer[:,j]))
        j=j+1
    print("op",output)
    output = 1 / (1 + np.exp(-(output)))
    print("exp op", output)
    print("expected angles",output *180)
    return output

no_nodes_hl1=4
no_nodes_hl2=4
no_nodes_hl3=4

n_classes=3
batch_size = 5

x=tf.placeholder('float',[None,len(train_x[0])])
y=tf.placeholder('float')

def Neural_Network_model(data) :
    hidden_Layer_1={'weights': tf.Variable(tf.random_normal([len(train_x[0]),no_nodes_hl1]))}
    hidden_Layer_2 = {'weights': tf.Variable(tf.random_normal([no_nodes_hl1, no_nodes_hl2]))}
    output_Layer = {'weights': tf.Variable(tf.random_normal([no_nodes_hl2, n_classes]))}
    layer1 = tf.matmul(data, hidden_Layer_1['weights'])
    layer1 = tf.nn.sigmoid(layer1)
    layer2 = tf.matmul(layer1, hidden_Layer_2['weights'])
    layer2 = tf.nn.sigmoid(layer2)
    output= tf.matmul(layer2, output_Layer['weights'])
    output = tf.nn.sigmoid(output)
    # get the co-ord usinng vision data return that as predict
    return output

def train_neural_networks(x):
    print("hey")
    predict = Neural_Network_model(x)
    cost= tf.reduce_mean(tf.pow(predict-y,2), name='loss' )
    #new cost calculation based on the vision
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
