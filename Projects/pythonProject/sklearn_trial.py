# How to Build a Machine Learning Classifier in Python with Scikit-learn

# Step 1 — Importing Scikit-learn
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB # We will focus on a simple algorithm that usaully performs well in binary classification
from sklearn.metrix import accuary_score

# Step 2 — Importing Scikit-learn’s Dataset
# Load dataset
data = load_breast_cancer()

# The data variable represents a Python object that works like a dictionary.
# The important dictionary keys to consider are the classification label names (target_names),
# the actual labels (target), the attribute/feature names (feature_names), and the attributes (data).

# Step 3 — Organizing Data into Sets
# Organize our data
label_names = data['target_names']
labels = data['target']
feature_names = data['feature_names']
features = data['data']

# print(label_names)
# print(labels[0])
# print(feature_names[0])
# print(features[0])

# Split our data
train, test, train_labels, test_labels = train_test_split(features
                                                          , labels
                                                          , test_size=0.33
                                                          , random_state=42)


# Step 4 — Building and Evaluating the Model
# The function randomly splits the data using the test_size parameter.
# In this example, we now have a test set (test) that represents 33% of the original dataset.
# The remaining data (train) then makes up the training data.
# We also have the respective labels for both the train/test variables, i.e. train_labels and test_labels.

# Building and Evaluating the Model
gnb = GaussianNB()
model = gnb.fit(train, train_labels)

# After we train the model, we can then use the trained model to make predictions on our test set,
# which we do using the predict() function. The predict() function returns an array of predictions for each data instance in the test set.
# We can then print our predictions to get a sense of what the model determined.

preds = gnb.predict(test)
print(preds)

# Step 5 — Evaluating the Model’s Accuracy
# Evaluating the Model’s Accuracy
# Using the array of true class labels, we can evaluate the accuracy of our model’s predicted values by
# comparing the two arrays (test_labels vs. preds). We will use the sklearn function accuracy_score()
# to determine the accuracy of our machine learning classifier.



def sklearn_print():
    print();
