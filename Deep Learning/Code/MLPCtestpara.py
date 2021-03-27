import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import accuracy_score

name = ['faculty', 'price', 'purpose1', 'purpose2', 'purpose3','freaquency', 'gender', 'class label']
feature_name = ['faculty', 'price', 'purpose1', 'purpose2', 'purpose3','freaquency', 'gender']
class_label = ['class label']
data = pd.read_csv(r".\\dataset\\dataset_encoded_balance.csv", header=0, names=name)

# test รอบ
k_range = range(1,11)

# array acc
scores = []

x = data[feature_name]
y = data[class_label]
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,random_state=0)
print((x_train.shape[0], x_test.shape[0]))

for k in k_range:
    model = MLPClassifier(alpha=0.01,hidden_layer_sizes=(500,), learning_rate='adaptive', max_iter=1000)
    model = model.fit(x_train,y_train.values.ravel())
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
    scores.append(accuracy)
    plot_confusion_matrix(model, x_test, y_test)
    print('รอบที่ :'+str(k))
    print("Accuracy: {:.2f}%".format(accuracy*100))