import pandas as pd
from sklearn.datasets import load_iris
from matplotlib import pyplot as plt
# %matplotlib
# %matplotlib inline sets the backend of matplotlib to the 'inline' backend:
# With this backend, the output of plotting commands is displayed inline within frontends like the Jupyter notebook,
# directly below the code cell that produced it.
# The resulting plots will then also be stored in the notebook document.
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['flower_name'] = df.target.apply(lambda x: iris.target_names[x])
# df.apply(): if you want to generate a new column from the already existing column u can use apply()
#              use (lambda x: sth to plug in x) -> x can be plugged in


df0 = df[df.target == 0]
df1 = df[df.target == 1]
df2 = df[df.target == 2]

X = df.drop(['target', 'flower_name'], axis = 'column')
y = df.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
# We are getting train_set, test_set for X, y

model = SVC()
model.fit(X_train, y_train)
# fit:
model.score(X_test, y_test)
# We are getting the accuracy of the model

def print_svm():
    len(X_train)
    len(X_test)


def print_dataframe():
    print("df[df.target == 2].head()")
    print(df[df.target == 2].head())

    print("df[df.target == 2].head()")
    print(df[df.target == 2].head())

    print("df1.head()")
    print(df1.head())

def sepal_plot():
    plt.xlabel('sepal length (cm)')
    plt.ylabel('sepal width (cm)')
    plt.scatter(df1['sepal length (cm)'], df1['sepal width (cm)'], color='yellow', marker='+')
    plt.scatter(df2['sepal length (cm)'], df2['sepal width (cm)'], color='blue', marker='+')

def petal_plot():
    plt.xlabel('petal length (cm)')
    plt.ylabel('petal width (cm)')
    plt.scatter(df1['petal length (cm)'], df1['petal width (cm)'], color='yellow', marker='+')
    plt.scatter(df2['petal length (cm)'], df2['petal width (cm)'], color='blue', marker='+')




