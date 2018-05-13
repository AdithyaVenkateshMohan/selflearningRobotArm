
import sys
sys.path.append('C:\python64\Lib\site-packages')
import numpy as np


def intializeToRandom( n):

    rand =np.random.rand(n)

    return rand

weights=intializeToRandom(3);

print("weights",weights)

#input = array of 2 x and y co-ordinates
#weights from input to hidden will be size of input x size of hidden layer
# weights from hidden to output will size of hidden x no. of output
# accordingly the for the feedforward we have just multiply the input with weights and take a summation
#given the weights structure should be numpy array of size of the input x no.of hidden neurons


def forwardfeedInstant(weights,input):
    input=input
    output=[]
    for w in weights:
        output=np.append(output,np.dot(input,w))

    output = activationFunction(output) # activation function is been called here
    delerror = output*(1-output) # for error calulation

    return output

def activationFunction(output):

    output = 1/ (1 + np.exp(- (output))) #sigmoid function

    return output

def getinput(count):
    datalimit=10

    if (count < datalimit):
        input = np.random.rand(2) # just for now later get the input from the vision , code available in the github feed it as np.array
        return input
    else:
        return "over"


def getexpected(count):

    expected = np.random.rand(3)
    expected = expected*3
    return expected




def trainNeuralNet():
    countnew = 0  # gobal variable used as counnt for the below function

    epoch = 500

    for e in range(epoch):

        while (getinput(count)!= "over") :

            count = countnew # to avoid local variable collision
            count = count+1

            NoHiddenneurons=3 # hardcoded try to change
            Nooutputs=3
            LearningrateL2 = 0.01
            LearningrateL1 =0.01

            input = getinput(count) # give the input as np array of size 2 in here [ x y ] co-ordinates

            sizeL1 = input.size * NoHiddenneurons
            sizeL2 = NoHiddenneurons * Nooutputs

            if count ==1 and epoch ==1 :
                weightsL1 = intializeToRandom(sizeL1)
                weightsL1 = np.resize(weightsL1,(input.size,NoHiddenneurons))
                weightsL2 = intializeToRandom(sizeL2)
                weightsL2 = np.resize(weightsL2,(Nooutputs , NoHiddenneurons))



            hiddenop = forwardfeedInstant(weightsL1,input) # activated hidden output size of 3 x1

            forerrorL1 = hiddenop*(1-hiddenop) #differentiation for error cal sigmoid specific # size of 3 x1

            output = forwardfeedInstant(weightsL2,hiddenop)# activated output # size of 3x1

            predict = output # size of 3x1

            expected = getexpected(count) # size of 3x1

            forerrorL2 = output*(1-output) #differentiation for error cal sigmoid specific # size of 3x1

            error = expected - predict # size of 3x1

            deltaL2 = (forerrorL2*error) # size of 3x1



            # trainng with back prop starts here weights is changed in accorandance to the output
            transdeltaL2 = np.reshape(deltaL2,(Nooutputs,1)) # size 1x3

            delJL2 = transdeltaL2*error # size 3 X 3

            delWeightsL2 = - (LearningrateL2 * delJL2) # size 3 X 3

            weightsL2 = weightsL2 - delWeightsL2 # size 3 X 3 # training the weights from hidden to the output

            summationL2 = np.matmul(deltaL2,weightsL2)

            deltaL1 = summationL2 * forerrorL1

            inputtrans = np.reshape(input, (input.size , 1))

            delJL1 = - ( inputtrans * deltaL1)

            delWeightsL1 = LearningrateL1 * delJL1

            weightsL1 = weightsL1 - delWeightsL1








