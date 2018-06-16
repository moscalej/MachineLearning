import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.stats import entropy
from tqdm import tqdm
from sklearn.model_selection import KFold


def los_fact(criretion):
    def calc_entropy(y):
        # Entropy calculation
        if y.shape[0] == 0:
            return 0
        y0 = float(np.sum(y == 0)) / y.shape[0]
        y1 = float(np.sum(y == 1)) / y.shape[0]
        return entropy((y0, y1))

    def calc_gini_index( y):
        # Gini index calcualtion
        if y.shape[0] == 0:
            return 0
        y0 = float(np.sum(y == 0)) / y.shape[0]
        y1 = float(np.sum(y == 1)) / y.shape[0]
        return y0 * (1 - y0) + y1 * (1 - y1)

    def calc_classification_error(y):
        # Classification error calculation
        if y.shape[0] == 0:
            return 0
        y0 = float(np.sum(y == 0)) / y.shape[0]
        y1 = float(np.sum(y == 1)) / y.shape[0]
        return 1 - max(y0, y1)
    return dict(Enthropy= calc_entropy,
                GiniIndex=calc_gini_index ,
                 ClassificationError=calc_classification_error )[criretion]


class Node:
    def __init__(self, value, feature, node_type, chields=[], balance=[0,0]):
        self.feature = feature
        self.value=value
        self.node_type=node_type
        self.chields=chields
        self.balance = balance

    def predict(self,x):
        if self.node_type == 'Error':
            raise Exception('We made a error node need to de-bug')
        if self.node_type == 'leaf':
            return self.value
        elif x[self.feature] <= self.value:
            return self.chields[0].predict(x)
        else:
            return self.chields[1].predict(x)

    def __str__(self):
        return f'Type: {self.node_type} , value: {round(self.value,2)},' \
               f' feature: {self.feature}, balance: {self.balance}'


class Tree:
    def __init__(self, depht, loss,dropping = True):
        self.dropping = dropping
        self.depht = depht
        self._test = los_fact(loss)
        self.root = None

    def fit(self, X, Y):
        """
        This method will generate a binary tree where each node will ask a relevant question
        :param X:
        :param Y:
        :return:
        """

        self.root = self._split_tree(X, Y, 0)

    def predict(self,x):
        # y_hat = pd.Series(index=x.index)
        y_hat = x.apply(self.root.predict,axis=1)
        return y_hat

    def score(self,x,y):
        return np.mean(self.predict(x) == y)


    def _split_tree(self, X_data, Y_labels, depht):
        if self.depht == depht or X_data.shape[1] == 1:
            return Node(np.argmax([np.sum(Y_labels == 0), np.sum(Y_labels == 1)]), 'leaf', 'leaf', [], [sum(Y_labels == 0), sum(Y_labels == 1)])
        if len(X_data) == 0:
            raise Exception(f'We build and error node on the fit with depht {depht} and y:{Y_labels.shape}')
        elif len(Y_labels.unique()) != 2:
            node = Node(Y_labels.values[0], 'leaf', 'leaf', [], [sum(Y_labels == 0), sum(Y_labels == 1)])
            return node


        info = self._find_cut(X_data, Y_labels)
        feature = info['feature']
        value = info['value']
        left_x = X_data[X_data[feature] <= value]
        if self.dropping == True:
            left_x =left_x.drop(columns=feature)
        left_y = Y_labels[left_x.index]
        right_x = X_data[X_data[feature] > value]
        if self.dropping == True:
            right_x = right_x.drop(columns=feature)
        right_y =Y_labels[right_x.index]



        if len(left_y) == 0 :
            return Node(np.argmax([np.sum(right_y == 0), np.sum(right_y == 1)]),
                        'leaf', 'leaf', [], [sum(right_y == 0), sum(right_y == 1)])
        if len(right_y) == 0 :
            return Node(np.argmax([np.sum(left_y == 0), np.sum(left_y == 1)]),
                        'leaf', 'leaf', [], [sum(left_y == 0), sum(left_y == 1)])

        node_left = self._split_tree(left_x, left_y, depht + 1)
        node_right= self._split_tree(right_x, right_y, depht + 1)
        node = Node(value, feature, 'Desition', [node_left, node_right],[left_y.size,right_y.size])
        return node

    def _find_cut(self, X, Y):
        """

        :param X:
        :param Y:
        :return:
        {'score':gini ,'criterion':'le' or 'g','value': value ,'feature': wicch feature should cut}
        """
        result = self._min_loss_cut(X[0], Y)
        for feature in X.columns:
            feature_r  = self._min_loss_cut(X[feature], Y)
            if feature_r['score'] <= result['score']:
                result = feature_r
                result['feature']= feature
        return result

    def _min_loss_cut(self, feature, Y):
        classes = Y.unique()
        # fea_order = feature.sort_values().unique()
        treshold = np.linspace(np.min(feature), np.max(feature), np.min([100,y.size]))
        ent,value = 999 ,999
        best_th = np.nan
        for th in treshold:
            # Usin th, split the input data
            left_x = feature[feature <= th]
            left_y = Y[left_x.index]

            right_x = feature[feature > th]
            right_y = Y[right_x.index]

            left_ent = self._test(left_y)
            right_ent = self._test(right_y)

            this_ent = (left_x.shape[0] * left_ent + right_x.shape[0] * right_ent) \
                       / (left_x.shape[0] + right_x.shape[0])

            if this_ent < ent:
                ent = this_ent
                value = th
        return {'score':ent ,'value':value}


class RandonForest():
    def __init__(self, number_of_trees, depth, critiria):
        self.number_of_trees = number_of_trees
        self.depth = depth
        self.critiria = critiria
        self.fold = KFold(number_of_trees,shuffle=True)
        self._trees = [Tree(depth, critiria, dropping = True) for _ in range(number_of_trees)]

    def fit(self,x,y):
        for index, train_x, _,train_y ,_ in enumerate(self.fold.split(x,y)):
            self._trees[index].fit(train_x,train_y)

    #Not ready
    def predict(self,x):
        np.mean([tree.predict(x) for tree in self._trees])




from sklearn.model_selection import train_test_split
a = loadmat(r'C:\Users\amoscoso\Documents\Technion\MachineLearning\data\BreastCancerData.mat',appendmat=False)
x = pd.DataFrame(a['X'].T)
y = pd.Series(a['y'].reshape(-1))
    #%%
results = pd.DataFrame(columns = ['GiniIndex','ClassificationError', 'Enthropy'], index= [0.8])


train_x , test_x, train_y, test_y = train_test_split(x,y,train_size=200,test_size=0.2, random_state=42)
# tree = Tree(10, 'GiniIndex')
# tree.fit(train_x,train_y)
tree_c = Tree(10, 'ClassificationError')
tree_c.fit(train_x,train_y)
# tree_1 = Tree(10, 'Enthropy')
# tree_1.fit(train_x,train_y)
# results.loc[0.8,'GiniIndex']=tree.score(test_x, test_y)
results.loc[0.8,'ClassificationError']=tree_c.score(test_x, test_y)
# results.loc[0.8,'Enthropy']=tree_1.score(test_x, test_y)

"""
" normally you don't train a tree with all your data, each time you train a tree it will over fit the training data in to the tree (each tree remembers all the decisions) the best you can do with decision trees is to make a kfold with your training data and for each fold make a tree and train, then for predict you run your data for all the trees and take the arg max of then (that is a random forest)
"""


#%%
value =[]
x_val = np.arange(0.1,0.85,0.08)
for i in x_val:

    train_x , test_x, train_y, test_y = train_test_split(x,y,train_size=i,test_size=0.2)
    tree = Tree(10, 'Enthropy')
    tree.fit(train_x,train_y)
    value.append(tree.score(test_x, test_y))

#%%
plt.plot(x_val,value)

