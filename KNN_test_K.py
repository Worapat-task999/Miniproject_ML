import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

name = ['faculty', 'price', 'purpose1', 'purpose2', 'purpose3','freaquency', 'gender', 'class label']
feature_name = ['faculty', 'price', 'purpose1', 'purpose2', 'purpose3','freaquency', 'gender']
class_label = ['class label']
data = pd.read_csv(r".\\dataset\\dataset_encoded_balance.csv", header=0, names=name)

x = data[feature_name]
y = data[class_label]
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,random_state=0)

# test k
k_range = range(1,101)

# array acc
scores = []

print((x_train.shape[0], x_test.shape[0]))

for k in k_range:
    model = KNeighborsClassifier(weights='distance',n_neighbors=k)
    model.fit(x_train, np.ravel(y_train))
    y_pred = model.predict(x_test)
    accuracy=accuracy_score(y_test, y_pred)
    scores.append(accuracy)
    print('value K :'+str(k))
    print("Accuracy: {:.2f}%".format(accuracy*100))

plt.plot(k_range, scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Testing Accuracy')