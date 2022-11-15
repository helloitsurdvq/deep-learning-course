#%%1 load library
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.datasets import mnist
from keras.utils import np_utils

# %%2 load dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_val, y_val = X_train[50000:60000,:], y_train[50000:60000]
X_train, y_train = X_train[:50000,:], y_train[:50000]
print(X_train.shape)

# %%3 reshape the data dung kich thuoc ma keras yeu cau 
X_train = X_train.reshape(X_train.shape[0], 28, 28 ,1)
X_val = X_val.reshape(X_val.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

# %%4 one hot encoding label thanh 1 vector 
Y_train = np_utils.to_categorical(y_train, 10)
Y_val = np_utils.to_categorical(y_val, 10)
Y_test = np_utils.to_categorical(y_test, 10)
print("intial data: ", y_train[0])
print("data after encoding: ", Y_train[0])

# %%5 model definition
model = Sequential()
#them conv layer with 32 kernels, kich thuoc kernel 3*3, chi ro input_shape cho layer dau tien 
model.add(Conv2D(32, (3, 3), activation='sigmoid', input_shape=(28, 28, 1))) 
model.add(Conv2D(32, (3, 3), activation='sigmoid'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128, activation='sigmoid')) #them fully connected layer voi 128 nodes with sigmoid function
model.add(Dense(10, activation='softmax')) # output layer with 10 nodes and using softmax function

# %%6 compile model , using loss function 
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# %%7 train model and data
H = model.fit(X_train, Y_train, validation_data=(X_val, Y_val),
              batch_size=32, epochs=10, verbose=1)

# %%8 visualize loss, accuracy training and validation dataset
fig = plt.figure()
numOfEpoch = 10
plt.plot(np.arange(0, numOfEpoch), H.history['loss'], label='training loss')
plt.plot(np.arange(0, numOfEpoch), H.history['val_loss'], label='validation loss')
plt.plot(np.arange(0, numOfEpoch), H.history['acc'], label='accuracy')
plt.plot(np.arange(0, numOfEpoch), H.history['val_acc'], label='validation accuracy')
plt.title('Accuracy and Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss|Accuracy')
plt.legend()


# %%9 evaluate model
score = model.evaluate(X_test, Y_test, verbose=0)
print(score)

# %%10 prediction
# for i in range (0, 5): 
#     plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
#     y_predict = model.predict(X_test[i].reshape(1, 28, 28, 1))
#     print("Predict value: ", np.argmax(y_predict))
plt.imshow(X_test[2].reshape(28, 28), cmap='gray')
y_predict = model.predict(X_test[2].reshape(1, 28, 28, 1))
print("Predict value: ", np.argmax(y_predict))
# %%

