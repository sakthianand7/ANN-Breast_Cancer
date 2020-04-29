import pandas  as pd
dataset=pd.read_csv('breast_cancer_dataset.csv')
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values
from sklearn.preprocessing import LabelEncoder
label_y=LabelEncoder()
y=label_y.fit_transform(y)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1)
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
X_train=sc_x.fit_transform(X_train)
X_test=sc_x.transform(X_test)
import keras 
from keras.models import Sequential
from keras.layers import Dense
classifier=Sequential()
classifier.add(Dense(input_dim=9,output_dim=5,activation='relu',init='uniform'))
classifier.add(Dense(output_dim=5,activation='relu',init='uniform'))
classifier.add(Dense(output_dim=1,activation='sigmoid',init='uniform'))
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
classifier.fit(X_train,y_train,epochs=50)
ypred=classifier.predict(X_test)
ypred=(ypred>0.5)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,ypred)