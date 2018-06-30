from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import math
NUM_OF_NEURONS_OUTPUT_LAYER=1
NUM_OF_LAYERS = 2

def GetDataFromMatlab(matFile):
    '''
    :param matFile: .m file
    :return: X-data np.array(dim=NXD) ,y-labels np.array(dim=NX1)
    '''
    data = loadmat(matFile)
    X = np.array(data['X']).T
    Y = data['y']
    y = np.array([elem[0] for elem in Y])
    y = y.reshape(y.shape[0],1,1)
    return X,y

def SplitDataToTrainAndTestSet(X,y):
    '''
    mix the indexes and choose training and test set in random choice
    :param X:data. NXD
    :param y: labels
    :return: training and test set
    '''
    N = X.shape[0]
    idxs = np.linspace(0, N - 1, N)
    # random.shuffle(idxs)
    idxs_training_set = idxs[0:int(0.8 * N)]
    idxs_test_set = idxs[int(0.8 * N):]

    X_training_set = np.array([X[int(i),:] for i in idxs_training_set])
    X_training_set = X_training_set.reshape(X_training_set.shape[0],X_training_set.shape[1],1)
    X_test_set = np.array([X[int(i),:] for i in idxs_test_set])
    X_test_set = X_test_set .reshape(X_test_set.shape[0], X_test_set.shape[1], 1)
    y_training_set = np.array([y[int(i)] for i in idxs_training_set])
    y_test_set = np.array([y[int(i)] for i in idxs_test_set])
    return X_training_set,X_test_set,y_training_set,y_test_set

#activation functions
Tanh = lambda x: np.tanh(x/2)

Sigmoid = lambda x: 1 / (1 + np.exp(-x))

Relu = lambda x: np.maximum(0,x)


#derivations of activation functions
d_Tanh = lambda x: 0.5*(1+Tanh(x))*(1-Tanh(x))

d_Sigmoid = lambda x: Sigmoid(x)*(1-Sigmoid(x))

def d_Relu(x):
    r = x.copy()
    r[x<=0]=0
    r[x>0]=1
    return r


class Layer():
    '''
    general layer (hidden or output layer)
    '''
    def __init__(self,nextLayer,inputDim,outputDim,activation,d_activation,weights,
                 lr,dE_do_last_layer,GetAccuracyAndLoss):
        self.m_nextLayer = nextLayer
        self.m_dE_do_last_layer = dE_do_last_layer
        self.m_inputDim = inputDim
        self.m_outputDim = outputDim #num of neurons at this layer
        self.m_activation = activation
        self.m_d_activation = d_activation
        self.m_weights = weights
        self.m_lr = lr
        self.m_GetAccuracyAndLoss = GetAccuracyAndLoss


    def ForwardPass(self,x):
        '''
        calculate output and derivative of activation function of this layer for back pass
        :param x: input to this layer
        :return: o,do_dv
        '''
        v = self.m_weights @ x #dim (N1,N2,1)
        o = self.m_activation(v)#dim (N1,N2,1)
        do_dv = self.m_d_activation(v)#dim (N1,N2,1)
        return o,do_dv

    def BackwardPass(self,dE_do,do_dv,x):
        '''
        calculate derivative of error by input to this layer: dE_dx
        :param dE_do: derivative of error by output of this layer
        :param do_dv: derivative of activation function
        :param x: input to this layer
        :return: dE_dx - derivative of error by input to this layer
        '''
        dE_dv = np.multiply(dE_do,do_dv) #dim (N1,N2,1)
        dE_dw = dE_dv @ np.transpose(x,(0,2,1)) #dim (N1,N2,D)
        self.m_weights = np.mean(self.m_weights - np.multiply(self.m_lr,dE_dw),axis=0)
        #dE_dx = np.transpose(np.transpose(dE_dv,(0,2,1)) @ self.m_weights,(0,2,1)) #dim (N1,1,D)
        dE_dx = self.m_weights.T @ dE_dv #dim (N1,1,D)
        return dE_dx


    def fit(self,y,x):
        '''
        RunTraining - back propagation
        :param y: real labels
        :param x:input for this layer
        :return:
        '''
        o, do_dv = self.ForwardPass(x)
        if self.m_nextLayer:
            dE_do, accuracy,loss= self.m_nextLayer.fit(y, o)
            # if len(dE_do) == 1:
            #     return 1
        else: #output layer
            dE_do = self.m_dE_do_last_layer(o,y)
            accuracy,loss = self.m_GetAccuracyAndLoss(o, y)
            # finish = self.m_IsNeedToFinish(o, y):
        dE_dx = self.BackwardPass(dE_do,do_dv,x)
        return dE_dx,accuracy,loss

    def Predict(self,x):
        '''
        predict labels of input x
        :param x: array. 3d.
             dim 1 - for samples.
             dim 2 - for attributes.
             dim 3 - len 1
        :return: predicted labels. 1d array
        '''
        v = self.m_weights @ x  # dim (N1,N2,1)
        if self.m_nextLayer==None:#in last layer do LogisticRegression
            return 1/(1+np.exp(-v))
        else:
            o = self.m_activation(v)#dim (N1,N2,1)
            return self.m_nextLayer.Predict(o)

class NeuralNetwork():
    def __init__(self,X,Y,Test_x,Test_y,activation,weightsHiddenLayer,weightsOutputLayer,lr,
                 threshToFinish,maxEpochs,preProccess):
        '''
        :param X: input with dimension of NXD
        :param Y: real labels with dimension of 2XN
        :param numOfNeuronsToHiddenLayer: number of neurons to hidden layer
        :param activationFunction: which activation function to use
        :param weightsHiddenLayer: matrix NXD (N-numOfNeuronsToHiddenLayer)
        :param weightsOutputLayer: matrix 2XD
        :param maxEpochs: max epochs to run
        :param preProccess: bool: preProccess of the data
        '''
        idxsTrainSet =np.arange(0,np.round(0.8*X.shape[0]),dtype=int)
        idxsValSet =np.arange(np.round(0.8*X.shape[0]),X.shape[0],dtype=int)
        self.m_X = X #total train
        self.m_Y = Y #total train
        self.m_X_train = X[idxsTrainSet,:,:]
        self.m_X_val = X[idxsValSet,:,:]
        self.m_X_test = Test_x
        self.m_Y_train = Y[idxsTrainSet, :, :]
        self.m_Y_val = Y[idxsValSet, :, :]
        self.m_Y_test = Test_y
        self.m_D = X.shape[1]
        self.m_numOfSamples = X.shape[0]
        self.m_lr = lr
        self.m_maxEpochs = maxEpochs
        self.m_preProccess = preProccess
        self.m_numberOfNeuronsToHiddenLayer = weightsHiddenLayer.shape[1]#weightsHiddenLayer need to be 3d
        self.m_activation = {'Tanh':Tanh,'Sigmoid':Sigmoid,'Relu':Relu}.get(activation)
        self.m_d_activation = {'Tanh':d_Tanh,'Sigmoid':d_Sigmoid,'Relu':d_Relu}.get(activation)
        self.m_threshToFinish = threshToFinish
        self.m_outputLayer = Layer(None,self.m_numberOfNeuronsToHiddenLayer,1,Sigmoid,
                                   d_Sigmoid,weightsOutputLayer,lr,self.dE_do_last_layer,
                                   self.GetAccuracyAndLoss)

        self.m_hiddenLayer = Layer(self.m_outputLayer,self.m_D,self.m_numberOfNeuronsToHiddenLayer
                                   ,self.m_activation,self.m_d_activation,weightsHiddenLayer,lr,None,
                                   self.GetAccuracyAndLoss)

    def fit(self):
        '''
        main function - for running training of the neural network
        :return: list of accuracy and loss of: train set, validation set and test set:
        accuracyTrain,lossTrain,accuracyVal,lossVal,accuracyTotalTrain,lossTotalTrain,accuracyTotalTest,lossTotalTest
        '''
        accuracyTrain = [];lossTrain = []
        accuracyVal = [];lossVal = []
        accuracyTotalTrain = [];lossTotalTrain=[]
        accuracyTotalTest = [];lossTotalTest = []
        for i in range(self.m_maxEpochs):
            print('epoch num ' + str(i+1))
            dE_dx, accuracy_train,loss_train = self.m_hiddenLayer.fit(self.m_Y_train, self.m_X_train)
            accuracyTrain.append(accuracy_train)
            lossTrain.append(loss_train)
            y_hat = self.m_hiddenLayer.Predict(self.m_X_val)
            accuracy_val,loss_val = self.GetAccuracyAndLoss(y_hat, self.m_Y_val)
            accuracyVal.append(accuracy_val)
            lossVal.append(loss_val)

            #final res - total train
            y_hat_total_train = self.m_hiddenLayer.Predict(self.m_X)
            accuracy_total_train, loss_total_train = self.GetAccuracyAndLoss(y_hat_total_train , self.m_Y)
            accuracyTotalTrain.append(accuracy_total_train)
            lossTotalTrain.append(loss_total_train)
            # final res - test
            y_hat_test = self.m_hiddenLayer.Predict(self.m_X_test)
            accuracy_test, loss_test = self.GetAccuracyAndLoss(y_hat_test , self.m_Y_test)
            accuracyTotalTest.append(accuracy_test)
            lossTotalTest.append(loss_test)

            if accuracy_val >= self.m_threshToFinish:
                break
        return accuracyTrain,lossTrain,accuracyVal,lossVal,accuracyTotalTrain,lossTotalTrain,accuracyTotalTest,lossTotalTest

    def dE_do_last_layer(self,y_hat, y):
        '''
        derivative of 'MSE'
        '''
        return y_hat - y


    def GetAccuracyAndLoss(self,y_hat,y):
        '''
        get accuracy and loss of labels
        :param y_hat: array 1 dim
        :param y:array 1 dim
        :return: accuracy,loss - each of them is int
        '''
        y_hat_decision = np.round(y_hat)
        accuracy = np.sum(y_hat_decision==y)/y_hat.size
        loss = 1-accuracy
        return accuracy,loss

def PlotLearningCurveFinalTest(accuracyTotalTrain, lossTotalTrain, accuracyTotalTest, lossTotalTest,lr,activation,numberOfNeuronsToHiddenLayer,preProccess,i):
    '''
    plot final graphs of total train ant test set
    :param accuracyTotalTrain:  <list>: accuracy of total train set along epochs
    :param lossTotalTrain: <list>: loss of total train set along epochs
    :param accuracyTotalTest: <list>: accuracy of test set along epochs
    :param lossTotalTest: <list>: loss of total test set along epochs
    :param lr: <int>: learning rate
    :param activation: <str> : activation function
    :param numberOfNeuronsToHiddenLayer: <int> num of neurons of hidden layer
    :param preProccess: <bool> doing preprocess of the data or not
    :param i: <int>: num of graph in subplot
    :return:
    '''
    plt.figure()
    plt.plot(np.arange(1,len(accuracyTotalTrain)+1),accuracyTotalTrain,label='accuracy total train')
    plt.plot(np.arange(1, len(accuracyTotalTest) + 1), accuracyTotalTest, label='accuracy test')
    plt.plot(np.arange(1,len(lossTotalTrain)+1),lossTotalTrain,label='loss total train')
    plt.plot(np.arange(1,len(lossTotalTest)+1),lossTotalTest,label='loss test')
    plt.xlabel('epoch')
    plt.ylabel('accuracy, loss')
    plt.title('Learning Curve - final results:\n lr='+str(lr)+', activation='+activation+', num of neurons='+str(numberOfNeuronsToHiddenLayer)+
              ', preProccess='+str(preProccess)+\
              '\nfinal accuracy: total train='+str("{0:.3f}".format(accuracyTotalTrain[-1]))+', test='+str("{0:.3f}".format(accuracyTotalTest[-1]))+
              '| final loss: total train='+str("{0:.3f}".format(lossTotalTrain[-1]))+', test='+str("{0:.3f}".format(lossTotalTest[-1])))
    plt.legend()
    if(i==3):
        plt.tight_layout()
        plt.savefig('res.jpg', bbox_inches='tight')
    plt.show(block=False)


def PlotLearningCurve(accuracyTrain,lossTrain,accuracyVal,lossVal,lr,activation,numberOfNeuronsToHiddenLayer,preProccess,i):
    '''
    plot results of learning curve of the neural network
    :param accuracyTrain: <list>: accuracy of train set along epochs
    :param lossTrain: <list>: loss of train set along epochs
    :param accuracyVal: <list>: accuracy of validation set along epochs
    :param lossVal: <list>: loss of validation set along epochs
    :param lr: <int>: learning rate
    :param activation: <str> : activation function
    :param numberOfNeuronsToHiddenLayer: <int> num of neurons of hidden layer
    :param preProccess: <bool> doing preprocess of the data or not
    :param i: num of graph in subplot
    '''
    plt.subplot(3,1,i)
    plt.plot(np.arange(1,len(accuracyTrain)+1),accuracyTrain,label='accuracy train')
    plt.plot(np.arange(1, len(accuracyVal) + 1), accuracyVal, label='accuracy val')
    plt.plot(np.arange(1,len(lossTrain)+1),lossTrain,label='loss train')
    plt.plot(np.arange(1,len(lossVal)+1),lossVal,label='loss val')
    plt.xlabel('epoch')
    plt.ylabel('accuracy, loss')
    plt.title('Learning Curve: lr='+str(lr)+', activation='+activation+', num of neurons='+str(numberOfNeuronsToHiddenLayer)+
              ', preProccess='+str(preProccess)+\
              '\nfinal accuracy: train='+str("{0:.3f}".format(accuracyTrain[-1]))+', val='+str("{0:.3f}".format(accuracyVal[-1]))+
              '| final loss: train='+str("{0:.3f}".format(lossTrain[-1]))+', val='+str("{0:.3f}".format(lossVal[-1])))
    plt.legend()
    if(i==3):
        plt.tight_layout()
        plt.savefig('res.jpg', bbox_inches='tight')
    plt.show(block=False)


def main():
    matFile = r'BreastCancerData.mat'
    X, Y = GetDataFromMatlab(matFile)
    preProccess = True
    if preProccess:
        X=(X-np.mean(X,axis=0))/np.std(X,axis=0)
    X = np.insert(X, 0, np.ones((1, X.shape[0])), axis=1)
    #insert ones to x
    Train_x, Test_x, Train_y, Test_y = SplitDataToTrainAndTestSet(X, Y)
    D = Train_x.shape[1]
    activations = ['Sigmoid','Relu','Tanh']
    threshToFinish = 0.97
    # threshToFinish = 0.99
    lrSpace = [1,0.5,0.1]
    numberOfNeuronsHiddenLayerSpace=[30,25,20]
    numberOfNeuronsHiddenLayerSpace=[5,10,15]
    numberOfNeuronsHiddenLayerSpace=[1,2,3]
    activation = 'Relu'
    lr = 1
    # numberOfNeuronsHiddenLayer = D-1#
    numberOfNeuronsHiddenLayer = 2
    # for i, activation in enumerate(activations):
    # for i, lr in enumerate(lrSpace):
    # for i, numberOfNeuronsHiddenLayer in enumerate(numberOfNeuronsHiddenLayerSpace):
    i=0
    weightsHiddenLayer = np.random.rand(numberOfNeuronsHiddenLayer,D)#(N2,D) uniform distribution over [0, 1)
    # weightsHiddenLayer = np.random.normal(0,1,(numberOfNeuronsHiddenLayer,D))#(N2,D)
    # weightsHiddenLayer = np.zeros((numberOfNeuronsHiddenLayer, D))  # (N2,D)
    weightsOutputLayer= np.random.rand(NUM_OF_NEURONS_OUTPUT_LAYER,numberOfNeuronsHiddenLayer)#(N1,dim_output,N2)uniform distribution over [0, 1)
    # weightsOutputLayer= np.random.normal(0,1,(NUM_OF_NEURONS_OUTPUT_LAYER,numberOfNeuronsHiddenLayer))#(N1,dim_output,N2)
    # weightsOutputLayer= np.zeros((NUM_OF_NEURONS_OUTPUT_LAYER,numberOfNeuronsHiddenLayer))#(N1,dim_output,N2)
    maxEpochs = 1000
    neuralNetwork = NeuralNetwork(Train_x,Train_y,Test_x,Test_y,activation,weightsHiddenLayer,
                  weightsOutputLayer,lr,threshToFinish,maxEpochs,preProccess)
    accuracyTrain, lossTrain, accuracyVal, lossVal, accuracyTotalTrain, lossTotalTrain, accuracyTotalTest, lossTotalTest= neuralNetwork.fit()
    #for training
    # PlotLearningCurve(accuracyTrain,lossTrain,accuracyVal,lossVal,lr,activation,numberOfNeuronsHiddenLayer,preProccess,i+1)
    # for final results
    PlotLearningCurveFinalTest(accuracyTotalTrain, lossTotalTrain, accuracyTotalTest, lossTotalTest,lr,activation,numberOfNeuronsHiddenLayer,preProccess,i+1)

    print('finish single iteration')
    print('finish all!')

if __name__=='__main__':
    main()