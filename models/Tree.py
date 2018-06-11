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
    return dict(ClassificationError= calc_entropy,
                GiniIndex=calc_gini_index ,
                Enthropy=calc_classification_error )[criretion]


class Node:
    def __init__(self, value, feature, node_type, chields=[], balance=[0,0]):
        self.feature = feature
        self.value=value
        self.node_type=node_type
        self.chields=chields
        self.balance = balance

    def predict(self,x):
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


    def _split_tree(self, X, Y, depht):
        if len(X) == 0:
             node = Node('Error', 'Error', -1, [],[0,0])
             return node
        elif len(Y.unique()) != 2:
            node = Node(Y.values[0], 'leaf', 'leaf', [],[sum(Y==0),sum(Y==1)])
            return node
        elif self.depht == depht or X.shape[1] == 1:
            return Node(np.argmax([np.sum(Y == 0), np.sum(Y == 1)]), 'leaf', 'leaf', [],[sum(Y==0),sum(Y==1)])
        info = self._find_cut(X, Y)
        feature = info['feature']
        value = info['value']
        left_x = X[X[feature] <= value]
        if self.dropping == True:
            left_x =left_x.drop(columns=feature)
        left_y = Y[left_x.index]
        right_x = X[X[feature] > value]
        if self.dropping == True:
            right_x = right_x.drop(columns=feature)
        right_y =Y[right_x.index]
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
        result = self._min_loss_cut(X[1], Y)
        for feature in tqdm(X.columns):
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



if __name__ == '__main__':
    from sklearn.model_selection import train_test_split
    a = loadmat(r'D:\Ale\Documents\Technion\ML\MachineLearning\Data\BreastCancerData.mat',appendmat=False)
    x = pd.DataFrame(a['X'].T)
    y = pd.Series(a['y'].reshape(-1))
    results = pd.DataFrame(columns = ['GiniIndex','ClassificationError', 'Enthropy'], index= [10,50,99])
    for col in results.columns:
        for row in results.index:
            train_x , test_x, train_y, test_y = train_test_split(x,y,train_size=row,test_size=300)
            tree = Tree(10, col)
            tree.fit(train_x,train_y)
            results.loc[row,col]=tree.score(test_x, test_y)

