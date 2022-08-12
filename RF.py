import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestRegressor


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

rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
rf.fit(x_train, y_train.ravel())

y_pred = rf.predict(x2)
y_pred2 = [round(num) for num in y_pred]

accuracy = metrics.accuracy_score(y2, y_pred2)
print('Accuracy (Polynomial Kernel): ', "%.2f" % (accuracy*100))


class_names = [1, 2, 3, 4]
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
cnf_matrix = metrics.confusion_matrix(y2, y_pred2)
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()
