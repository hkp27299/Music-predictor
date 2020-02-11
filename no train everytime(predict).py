#import pandas as pd
#from sklearn.tree import DecisionTreeClassifier
from sklearn.externals import joblib

# df = pd.read_csv('music.csv')
# x = df.drop(columns=['genre'])
# y = df['genre']
#
# model = DecisionTreeClassifier()
# model.fit(x, y)
#
# joblib.dump(model, 'music_predict.joblib')

model = joblib.load('music_predict.joblib')

predict = model.predict([ [21, 1], [22, 0 ] ])
print(predict)
