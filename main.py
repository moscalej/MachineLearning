from models.Tree import Tree
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.io import loadmat
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier

#%%
a = loadmat(r'D:\Ale\Documents\Technion\ML\MachineLearning\Data\BreastCancerData.mat',appendmat=False)
x = pd.DataFrame(a['X'].T)
y = pd.Series(a['y'].reshape(-1))
    #%%
results = pd.DataFrame(columns = ['GiniIndex','ClassificationError', 'Enthropy'], index= [10,50,70])
for col in results.columns:
    for row in results.index:
        train_x , test_x, train_y, test_y = train_test_split(x,y,train_size=row,test_size=300)
        tree = Tree(10, col)
        tree.fit(train_x,train_y)
        results.loc[row,col]=tree.score(test_x, test_y)


#%%
value =[]
x_val = np.arange(0.5,0.7,0.12)
for i in tqdm(x_val):
    print(i)
    train_x , test_x, train_y, test_y = train_test_split(x,y,train_size=i,test_size=0.2)
    tree = Tree(10, 'Enthropy')
    tree.fit(train_x,train_y)
    value.append(tree.score(test_x, test_y))

plt.plot(x_val,value)
plt.title("Destion Tree presition as a function of Sample size")

#%%

