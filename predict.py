import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv('music.csv')
x = df.drop(columns=['genre'])
y = df['genre']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


model = DecisionTreeClassifier()
model.fit(x_train, y_train)

predict = model.predict(x_test)
print(predict)

score = accuracy_score(y_test,predict)
print(score)
