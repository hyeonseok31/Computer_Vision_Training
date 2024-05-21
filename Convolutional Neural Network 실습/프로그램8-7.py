from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, Rescaling
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.utils import image_dataset_from_directory
import pathlib

data_path = pathlib.Path('Stanford_dogs/Images')

train_ds = image_dataset_from_directory(data_path, validation_split=0.2, subset='training', seed=123, image_size=(224, 224), batch_size=16)
test_ds = image_dataset_from_directory(data_path, validation_split=0.2, subset='validation', seed=123, image_size=(224, 224), batch_size=16)

base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
cnn = Sequential()
cnn.add(Rescaling(1.0 / 255.0))  # 입력 텐서를 [0, 1]로 정규화
cnn.add(base_model)
cnn.add(Flatten())  # 백본 출력 텐서를 1차원으로 flattening
cnn.add(Dense(1024, activation='relu'))
cnn.add(Dropout(0.75))
cnn.add(Dense(units=120, activation='softmax'))  # 결과 출력을 위한 완전연결층, 120개의 클래스

cnn.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=0.000001), metrics=['accuracy'])
hist = cnn.fit(train_ds, epochs=200, validation_data=test_ds, verbose=2)

print('정확률=', cnn.evaluate(test_ds, verbose=0)[1] * 100)

cnn.save('cnn_for_stanford_dogs.h5')  # 미세 조정된 모델을 파일에 저장

import pickle
f = open('dog_species_names.txt', 'wb')
pickle.dump(train_ds.class_names, f)
f.close()

import matplotlib.pyplot as plt

plt.plot(hist.history['accuracy'])  # ④
plt.plot(hist.history['val_accuracy'])
plt.title('Accuracy graph')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])
plt.grid()
plt.show()

plt.plot(hist.history['loss'])  # ⑤
plt.plot(hist.history['val_loss'])
plt.title('Loss graph')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])
plt.grid()
plt.show()