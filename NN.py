import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.neural_network import MLPClassifier

x= pd.read_excel('FailLreducida500.xlsx')
y = x.loc[:,['Tag']].values
x = x.drop('Unnamed: 0', 1)
x = x.drop('Tag', 1)

x2= pd.read_excel('NWreducida500.xlsx')
y2 = x2.loc[:,['Tag']].values
x2 = x2.drop('Unnamed: 0', 1)
x2 = x2.drop('Tag', 1)

t= StandardScaler()
t.fit(x)
x= t.transform(x)
x2= t.transform(x2)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

mlp = MLPClassifier(hidden_layer_sizes=(400,200,50), activation='relu', solver='adam', max_iter=500) #5
mlp.fit(x_train,y_train)

predict_train = mlp.predict(x_train)
predict_test = mlp.predict(x2)

from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_train,predict_train))
print(classification_report(y_train,predict_train))

print(confusion_matrix(y2,predict_test))
print(classification_report(y2,predict_test))

class_names = [1, 2, 3, 4]
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
cnf_matrix = metrics.confusion_matrix(y2, predict_test)
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

df_aux = pd.DataFrame(predict_test)

## save to xlsx file

df_aux.to_excel('pred.xlsx')
