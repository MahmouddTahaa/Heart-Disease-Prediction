import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import tensorflow as tf

df = pd.read_csv("./datasets/preprocessed_data.csv")

X = df.drop(columns=['HeartDisease'],axis=1)
Y = df['HeartDisease']

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.3)

model = Sequential()
model.add(layers.Dense(20,activation='relu',input_shape=(x_train.shape[1],)))
model.add(layers.Dense(1,activation=tf.keras.activations.hard_sigmoid))

model.compile(optimizer="adam" , loss="binary_crossentropy", metrics=['accuracy'])
model.fit(x_train ,y_train ,epochs=50, verbose=1)

model.save('./models/nn.h5')

