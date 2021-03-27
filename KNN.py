import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import accuracy_score
import numpy as np
from mlxtend.plotting import plot_decision_regions



name = ['faculty', 'price', 'purpose1', 'purpose2', 'purpose3','freaquency', 'gender', 'class label']
feature_name = ['faculty', 'price', 'purpose1', 'purpose2', 'purpose3','freaquency', 'gender']
class_label = ['class label']
data = pd.read_csv(r".\\dataset\\dataset_encoded_balance.csv", header=0, names=name)

x = data[feature_name]
y = data[class_label]
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,random_state=0)

# Create KNN classifier
model = KNeighborsClassifier(weights='distance',n_neighbors = 3)
print((x_train.shape[0], x_test.shape[0]))

model = model.fit(x_train, np.ravel(y_train))
y_pred = model.predict(x_test)
# print(y_pred)
plot_confusion_matrix(model, x_test, y_test)





# accuracy
accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
print("Accuracy: {:.2f}%".format(accuracy*100))