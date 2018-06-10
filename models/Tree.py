import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat


class Node:
    def __init__(self, criteria, value, tipe,childs):
        self.criteria=criteria
        self.value=value
        self.tipe=tipe
        self.childs=childs


class Tree:
    def __init__(self, depht):
        self.depht = depht
        self.criteria = 'gini'
        self.root = None

    def fit(self, X, Y):
        # self.root is
        self.root = self.slit_tree(X, Y)

    def slit_tree(self, X, Y):
        if len(X) == 0:
            node = Node('final', Y[0], 'final' ,None)
        info = self.find_cut(X,Y)
        feature = info['feature']
        value = info['value']
        criteria =info['criterion']
        if criteria == 'le':
            left_X = X[X[feature] <= value]
            left_X =left_X.drop(columns=feature)
            left_Y = Y[left_X.index]
            right_X = X[X[feature] > value].drop(columns=feature)
            right_Y =Y[right_X.index]
            node = Node('le',value,'Node',[self.slit_tree(left_X,left_Y),self.slit_tree(right_X,right_Y)])

        else:
            left_X = X[X[feature] > value].drop(columns=feature)
            left_Y = Y[left_X.index]
            right_X = X[X[feature] <= value].drop(columns=feature)
            right_Y = Y[right_X.index]
            node = Node('g', value, 'Node', [self.slit_tree(left_X,left_Y),self.slit_tree(right_X,right_Y)])

        return node

    def find_cut(self, X, Y):
        result = self.gini_index(X[1], Y)
        for feature in X.columns:
            feature_r  = self.gini_index(X[feature], Y)

            if (feature_r['score'] <= result['score']):
                result = feature_r
                result['feature']= feature
        return result

    def gini_index(self, feature, Y):
        classes = Y.unique()
        fea_order = feature.sort_values().unique()
        gini = 1
        desition = ['le', 'val']
        for val in fea_order[:-1] :
            score_right_le = self.score(Y[feature <= val],0,Y.size)
            score_left_le = self.score(Y[feature <= val],1,Y.size)
            score_right_g = self.score(Y[feature > val], 1,Y.size)
            score_left_g = self.score(Y[feature > val], 0,Y.size)
            gini_lq = 1 - score_left_le - score_right_le
            gini_g = 1 - score_left_g - score_right_g

            if gini_lq < gini_g:
                if gini_lq < gini:
                    gini = gini_lq
                    desition[0] = 'le'
                    desition[1] = val

            else:
                if gini_g < gini:
                    gini = gini_g
                    desition[0] = 'g'
                    desition[1] = val
        return {'score':gini , 'criterion':desition[0],'value':desition[1]}

    def score(self, Y, category, toal):
        if self.criteria == 'gini':
            some = Y.values
            group_prob = (np.sum(some== category)/toal)**2
        else:
            group_prob =0
        return group_prob


if __name__ == '__main__':
    from sklearn.model_selection import train_test_split
    a = loadmat(r'C:\Users\amoscoso\Documents\Technion\MachineLearning\data\BreastCancerData.mat',appendmat=False)
    x = pd.DataFrame(a['X'].T)
    y = pd.Series(a['y'].reshape(-1))
    train_x , test_x, train_y, test_y = train_test_split(x,y,train_size=20)
    tree = Tree(10)
    tree.fit(train_x,train_y)
