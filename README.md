# Лабораторная работа #2
## Решение задачи классификации изображений из набора данных [Oregon Wildlife](https://www.kaggle.com/virtualdvid/oregon-wildlife)  с использованием нейронных сетей глубокого обучения и техники обучения Transfer Learning

## [EfficientNet-B0](https://www.tensorflow.org/api_docs/python/tf/keras/applications/EfficientNetB0) architecture
![image](https://user-images.githubusercontent.com/80168174/110480321-6aae0900-80f7-11eb-82e6-f389f93c3966.png)


## Train 1
### Нейронная сеть [EfficientNet-B0](https://www.tensorflow.org/api_docs/python/tf/keras/applications/EfficientNetB0)  (случайное начальное приближение), датасет [Oregon Wildlife](https://www.kaggle.com/virtualdvid/oregon-wildlife).

```
BATCH_SIZE = 16

def build_model():
  inputs = tf.keras.Input(shape=(RESIZE_TO, RESIZE_TO, 3))
  outputs = EfficientNetB0(weights=None, classes=NUM_CLASSES)(inputs)
  return tf.keras.Model(inputs=inputs, outputs=outputs)
```
#### Оранживая - обучающая выборка, Синия - валидационная выборка (на всех графиках в данном отчете)
#### [TensorBoard](https://tensorboard.dev/experiment/4EoeVqP1TLq6X8EG6GhRgw/#scalars) 
#### epoch_categorical_accuracy
<img src="https://raw.githubusercontent.com/NikitaShulgan/Laba2/main/for_Readme/epoch_categorical_accuracy_EfficientNet-B0_weights_None.svg">

#### epoch_loss
<img src="https://raw.githubusercontent.com/NikitaShulgan/Laba2/main/for_Readme/epoch_loss_EfficientNet-B0_weights_None.svg">


## Train 2
### Нейронная сеть [EfficientNet-B0](https://www.tensorflow.org/api_docs/python/tf/keras/applications/EfficientNetB0)  (предобученная на ImageNet), датасет [Oregon Wildlife](https://www.kaggle.com/virtualdvid/oregon-wildlife).

```
BATCH_SIZE = 16

def build_model():
  inputs = tf.keras.Input(shape=(RESIZE_TO, RESIZE_TO, 3))
  x = EfficientNetB0(include_top=False, weights='imagenet', classes=NUM_CLASSES)(inputs)
  x = tf.keras.layers.Flatten()(x)
  x = tf.keras.layers.Dense(1280)(x)
  outputs = tf.keras.layers.Dense(NUM_CLASSES, input_shape=(7, 7), activation = tf.keras.activations.softmax)(x)
  return tf.keras.Model(inputs=inputs, outputs=outputs)
```
#### [TensorBoard](https://tensorboard.dev/experiment/jbmjL062Ra6PiakXOYlaKA/#scalars) 
#### epoch_categorical_accuracy
<img src="https://raw.githubusercontent.com/NikitaShulgan/Laba2/main/for_Readme/epoch_categorical_accuracy_EfficientNet-B0_weights_ImageNet.svg">

#### epoch_loss
<img src="https://raw.githubusercontent.com/NikitaShulgan/Laba2/main/for_Readme/epoch_loss_EfficientNet-B0_weights_ImageNet.svg">

## Train 3
##### log file owl-1615469804.7533162
### Нейронная сеть [EfficientNet-B0](https://www.tensorflow.org/api_docs/python/tf/keras/applications/EfficientNetB0)  (предобученная на ImageNet), датасет [Oregon Wildlife](https://www.kaggle.com/virtualdvid/oregon-wildlife).
#### В сравнении с [Train 2](https://github.com/NikitaShulgan/Laba2#train-2) были удалены:
#### 1. Cлой 
```
x = tf.keras.layers.Dense(1280)(x) 
```
#### 2. Аргумент ```input_shape=(7, 7)``` из ```tf.keras.layers.Dense```
#### В итоге получилось:
```
BATCH_SIZE = 16

def build_model():
  inputs = tf.keras.Input(shape=(RESIZE_TO, RESIZE_TO, 3))
  x = EfficientNetB0(include_top=False, weights='imagenet', classes=NUM_CLASSES)(inputs)
  x = tf.keras.layers.Flatten()(x)
  outputs = tf.keras.layers.Dense(NUM_CLASSES, activation = tf.keras.activations.softmax)(x)
  return tf.keras.Model(inputs=inputs, outputs=outputs)
```
### Модель нейронной сети Train 3
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(None, 224, 224, 3)]     0
_________________________________________________________________
efficientnetb0 (Functional)  (None, None, None, 1280)  4049571
_________________________________________________________________
flatten (Flatten)            (None, 62720)             0
_________________________________________________________________
dense (Dense)                (None, 20)                1254420
=================================================================
Total params: 5,303,991
Trainable params: 5,261,968
Non-trainable params: 42,023
```
#### [TensorBoard](https://tensorboard.dev/experiment/V0weQQ7rRPiqQmT9t6gBJA/#scalars) 
#### epoch_categorical_accuracy
<img src="https://raw.githubusercontent.com/NikitaShulgan/Laba2/main/for_Readme/Train_3_epoch_categorical_accuracy.svg">

#### epoch_loss
<img src="https://raw.githubusercontent.com/NikitaShulgan/Laba2/main/for_Readme/Train_3_epoch_loss.svg">

## Train 4 
##### log file owl-1615473209.896111
### Нейронная сеть [EfficientNet-B0](https://www.tensorflow.org/api_docs/python/tf/keras/applications/EfficientNetB0)  (предобученная на ImageNet), датасет [Oregon Wildlife](https://www.kaggle.com/virtualdvid/oregon-wildlife).
#### В сравнении с [Train 3](https://github.com/NikitaShulgan/Laba2#train-3) были изменены:
#### 1. BATCH_SIZE. Он увеличился с 16 до 64.
#### 2. Изменена функция активации в ```tf.keras.layers.Dense```. Была ```Softmax```, стала ```ReLU```.
#### В итоге получилось:
```
BATCH_SIZE = 64

def build_model():
  inputs = tf.keras.Input(shape=(RESIZE_TO, RESIZE_TO, 3))
  x = EfficientNetB0(include_top=False, weights='imagenet', classes=NUM_CLASSES)(inputs)
  x = tf.keras.layers.Flatten()(x)
  outputs = tf.keras.layers.Dense(NUM_CLASSES, activation = tf.keras.activations.relu)(x)
  return tf.keras.Model(inputs=inputs, outputs=outputs)
```
#### Модель нейронной сети Train 4 
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(None, 224, 224, 3)]     0
_________________________________________________________________
efficientnetb0 (Functional)  (None, None, None, 1280)  4049571
_________________________________________________________________
flatten (Flatten)            (None, 62720)             0
_________________________________________________________________
dense (Dense)                (None, 20)                1254420
=================================================================
Total params: 5,303,991
Trainable params: 5,261,968
Non-trainable params: 42,023
_________________________________________________________________
```
#### [TensorBoard](https://tensorboard.dev/experiment/kBQ9MhjJRuewgV982QFekA/#scalars) 
#### epoch_categorical_accuracy
<img src="https://raw.githubusercontent.com/NikitaShulgan/Laba2/main/for_Readme/Train_4_epoch_categorical_accuracy.svg">

#### epoch_loss
<img src="https://raw.githubusercontent.com/NikitaShulgan/Laba2/main/for_Readme/Train_4_epoch_loss.svg">

## Train 5 
##### log file owl-1615475964.8956075
### Нейронная сеть [EfficientNet-B0](https://www.tensorflow.org/api_docs/python/tf/keras/applications/EfficientNetB0)  (предобученная на ImageNet), датасет [Oregon Wildlife](https://www.kaggle.com/virtualdvid/oregon-wildlife).
```
BATCH_SIZE = 64

def build_model():
  inputs = tf.keras.Input(shape=(RESIZE_TO, RESIZE_TO, 3))
  x = EfficientNetB0(include_top=False, weights='imagenet', classes=NUM_CLASSES)(inputs)
  x = tf.keras.layers.GlobalAveragePooling2D()(x)
  outputs = tf.keras.layers.Dense(NUM_CLASSES, activation = tf.keras.activations.relu)(x)
  return tf.keras.Model(inputs=inputs, outputs=outputs)
```
#### Модель нейронной сети Train 5 owl-1615475964.8956075
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(None, 224, 224, 3)]     0
_________________________________________________________________
efficientnetb0 (Functional)  (None, None, None, 1280)  4049571
_________________________________________________________________
global_average_pooling2d (Gl (None, 1280)              0
_________________________________________________________________
dense (Dense)                (None, 20)                25620
=================================================================
Total params: 4,075,191
Trainable params: 4,033,168
Non-trainable params: 42,023
_________________________________________________________________
```
#### [TensorBoard](https://tensorboard.dev/experiment/6gGBdAtZSp6Ovqd2XCDq9g/#scalars) 
#### epoch_categorical_accuracy
<img src="https://raw.githubusercontent.com/NikitaShulgan/Laba2/main/for_Readme/Train_5_epoch_categorical_accuracy.svg">

#### epoch_loss
<img src="https://raw.githubusercontent.com/NikitaShulgan/Laba2/main/for_Readme/Train_5_epoch_loss.svg">

## Train 6 
##### log file owl-1615481288.6346006
### Нейронная сеть [EfficientNet-B0](https://www.tensorflow.org/api_docs/python/tf/keras/applications/EfficientNetB0)  (предобученная на ImageNet), датасет [Oregon Wildlife](https://www.kaggle.com/virtualdvid/oregon-wildlife).
```
BATCH_SIZE = 64

def build_model():
  inputs = tf.keras.Input(shape=(RESIZE_TO, RESIZE_TO, 3))
  x = EfficientNetB0(include_top=False, weights='imagenet', classes=NUM_CLASSES)(inputs)
  x = tf.keras.layers.GlobalMaxPool2D()(x)
  outputs = tf.keras.layers.Dense(NUM_CLASSES, activation = tf.keras.activations.relu)(x)
  return tf.keras.Model(inputs=inputs, outputs=outputs)
```
#### Модель нейронной сети Train 6
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(None, 224, 224, 3)]     0
_________________________________________________________________
efficientnetb0 (Functional)  (None, None, None, 1280)  4049571
_________________________________________________________________
global_max_pooling2d (Global (None, 1280)              0
_________________________________________________________________
dense (Dense)                (None, 20)                25620
=================================================================
Total params: 4,075,191
Trainable params: 4,033,168
Non-trainable params: 42,023
_________________________________________________________________
```
#### [TensorBoard](https://tensorboard.dev/experiment/zbYSmjCpRounPWXTmOGeww/#scalars) 
#### epoch_categorical_accuracy
<img src="https://raw.githubusercontent.com/NikitaShulgan/Laba2/main/for_Readme/Train_6_epoch_categorical_accuracy.svg">

#### epoch_loss
<img src="https://raw.githubusercontent.com/NikitaShulgan/Laba2/main/for_Readme/Train_6_epoch_loss.svg">

## Train 7 
##### log file owl-1615484396.3451083
### Нейронная сеть [EfficientNet-B0](https://www.tensorflow.org/api_docs/python/tf/keras/applications/EfficientNetB0)  (предобученная на ImageNet), датасет [Oregon Wildlife](https://www.kaggle.com/virtualdvid/oregon-wildlife).
```
BATCH_SIZE = 64

def build_model():
  inputs = tf.keras.Input(shape=(RESIZE_TO, RESIZE_TO, 3))
  x = EfficientNetB0(include_top=False, weights='imagenet', classes=NUM_CLASSES)(inputs)
  x = tf.keras.layers.GlobalMaxPool2D()(x)
  x = tf.keras.layers.Dense(1280)(x)
  x = tf.keras.layers.Dense(640)(x)
  x = tf.keras.layers.Dense(20)(x)
  outputs = tf.keras.layers.Dense(NUM_CLASSES, activation = tf.keras.activations.relu)(x)
  return tf.keras.Model(inputs=inputs, outputs=outputs)
```
#### Модель нейронной сети Train 7
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(None, 224, 224, 3)]     0
_________________________________________________________________
efficientnetb0 (Functional)  (None, None, None, 1280)  4049571
_________________________________________________________________
global_max_pooling2d (Global (None, 1280)              0
_________________________________________________________________
dense (Dense)                (None, 1280)              1639680
_________________________________________________________________
dense_1 (Dense)              (None, 640)               819840
_________________________________________________________________
dense_2 (Dense)              (None, 20)                12820
_________________________________________________________________
dense_3 (Dense)              (None, 20)                420
=================================================================
Total params: 6,522,331
Trainable params: 6,480,308
Non-trainable params: 42,023
_________________________________________________________________
```
#### [TensorBoard](https://tensorboard.dev/experiment/6j0NyyEiQYGoqBgDyIaR8Q/#scalars) 
#### epoch_categorical_accuracy
<img src="https://raw.githubusercontent.com/NikitaShulgan/Laba2/main/for_Readme/Train_7_epoch_categorical_accuracy.svg">

#### epoch_loss
<img src="https://raw.githubusercontent.com/NikitaShulgan/Laba2/main/for_Readme/Train_7_epoch_loss.svg">

## Train 8 
##### log file owl-1615492561.5753999
### Нейронная сеть [EfficientNet-B0](https://www.tensorflow.org/api_docs/python/tf/keras/applications/EfficientNetB0)  (предобученная на ImageNet), датасет [Oregon Wildlife](https://www.kaggle.com/virtualdvid/oregon-wildlife).
```
BATCH_SIZE = 64

def build_model():
  inputs = tf.keras.Input(shape=(RESIZE_TO, RESIZE_TO, 3))
  x = EfficientNetB0(include_top=False, weights='imagenet', classes=NUM_CLASSES)(inputs)
  x = tf.keras.layers.GlobalMaxPool2D()(x)
  x = tf.keras.layers.Dense(1280)(x)
  x = tf.keras.layers.Dense(640)(x)
  x = tf.keras.layers.Dense(320)(x)
  x = tf.keras.layers.Dense(160)(x)
  x = tf.keras.layers.Dense(80)(x)
  x = tf.keras.layers.Dense(40)(x)
  x = tf.keras.layers.Dense(20)(x)
  outputs = tf.keras.layers.Dense(NUM_CLASSES, activation = tf.keras.activations.relu)(x)
  return tf.keras.Model(inputs=inputs, outputs=outputs)
```
#### Модель нейронной сети Train 8
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(None, 224, 224, 3)]     0
_________________________________________________________________
efficientnetb0 (Functional)  (None, None, None, 1280)  4049571
_________________________________________________________________
global_max_pooling2d (Global (None, 1280)              0
_________________________________________________________________
dense (Dense)                (None, 1280)              1639680
_________________________________________________________________
dense_1 (Dense)              (None, 640)               819840
_________________________________________________________________
dense_2 (Dense)              (None, 320)               205120
_________________________________________________________________
dense_3 (Dense)              (None, 160)               51360
_________________________________________________________________
dense_4 (Dense)              (None, 80)                12880
_________________________________________________________________
dense_5 (Dense)              (None, 40)                3240
_________________________________________________________________
dense_6 (Dense)              (None, 20)                820
_________________________________________________________________
dense_7 (Dense)              (None, 20)                420
=================================================================
Total params: 6,782,931
Trainable params: 6,740,908
Non-trainable params: 42,023
_________________________________________________________________
```
#### [TensorBoard](https://tensorboard.dev/experiment/qLNm24DGR4ubd7B3KP2FEA/#scalars) 
#### epoch_categorical_accuracy
<img src="https://raw.githubusercontent.com/NikitaShulgan/Laba2/main/for_Readme/Train_8_epoch_categorical_accuracy.svg">

#### epoch_loss
<img src="https://raw.githubusercontent.com/NikitaShulgan/Laba2/main/for_Readme/Train_8_epoch_loss.svg">

## Train 9
##### log file owl-1615498611.943456
### Нейронная сеть [EfficientNet-B0](https://www.tensorflow.org/api_docs/python/tf/keras/applications/EfficientNetB0)  (предобученная на ImageNet), датасет [Oregon Wildlife](https://www.kaggle.com/virtualdvid/oregon-wildlife).
```
BATCH_SIZE = 64

def build_model():
  inputs = tf.keras.Input(shape=(RESIZE_TO, RESIZE_TO, 3))
  x = EfficientNetB0(include_top=False, weights="imagenet", pooling='avg', classes=NUM_CLASSES, classifier_activation="relu")(inputs)
  outputs = tf.keras.layers.Dense(NUM_CLASSES, activation = tf.keras.activations.relu)(x)
  return tf.keras.Model(inputs=inputs, outputs=outputs)
```
#### Модель нейронной сети Train 9
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(None, 224, 224, 3)]     0
_________________________________________________________________
efficientnetb0 (Functional)  (None, 1280)              4049571
_________________________________________________________________
dense (Dense)                (None, 20)                25620
=================================================================
Total params: 4,075,191
Trainable params: 4,033,168
Non-trainable params: 42,023
_________________________________________________________________
```
#### [TensorBoard](https://tensorboard.dev/experiment/0rGz8j9ER228nywO4ZcBIQ/#scalars) 
#### epoch_categorical_accuracy
<img src="https://raw.githubusercontent.com/NikitaShulgan/Laba2/main/for_Readme/Train_9_epoch_categorical_accuracy.svg">

#### epoch_loss
<img src="https://raw.githubusercontent.com/NikitaShulgan/Laba2/main/for_Readme/Train_9_epoch_loss.svg">

## Train 10
##### log file owl-1615501328.662753
### Нейронная сеть [EfficientNet-B0](https://www.tensorflow.org/api_docs/python/tf/keras/applications/EfficientNetB0)  (предобученная на ImageNet), датасет [Oregon Wildlife](https://www.kaggle.com/virtualdvid/oregon-wildlife).
```
BATCH_SIZE = 64

def build_model():
  inputs = tf.keras.Input(shape=(RESIZE_TO, RESIZE_TO, 3))
  x = EfficientNetB0(include_top=False, weights="imagenet", pooling='max', classes=NUM_CLASSES, classifier_activation="relu")(inputs)
  outputs = tf.keras.layers.Dense(NUM_CLASSES, activation = tf.keras.activations.relu)(x)
  return tf.keras.Model(inputs=inputs, outputs=outputs)
```
#### Модель нейронной сети Train 10
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(None, 224, 224, 3)]     0
_________________________________________________________________
efficientnetb0 (Functional)  (None, 1280)              4049571
_________________________________________________________________
dense (Dense)                (None, 20)                25620
=================================================================
Total params: 4,075,191
Trainable params: 4,033,168
Non-trainable params: 42,023
_________________________________________________________________
```
#### [TensorBoard](https://tensorboard.dev/experiment/EEPmrPHpSxKUoNI3AvCd2g/#scalars) 
#### epoch_categorical_accuracy
<img src="https://raw.githubusercontent.com/NikitaShulgan/Laba2/main/for_Readme/Train_10_epoch_categorical_accuracy.svg">

#### epoch_loss
<img src="https://raw.githubusercontent.com/NikitaShulgan/Laba2/main/for_Readme/Train_10_epoch_loss.svg">

## Train 11
##### log file owl-1615504511.8922284
### Нейронная сеть [EfficientNet-B0](https://www.tensorflow.org/api_docs/python/tf/keras/applications/EfficientNetB0)  (предобученная на ImageNet), датасет [Oregon Wildlife](https://www.kaggle.com/virtualdvid/oregon-wildlife).
```
BATCH_SIZE = 64

def build_model():
  inputs = tf.keras.Input(shape=(RESIZE_TO, RESIZE_TO, 3))
  x = EfficientNetB0(include_top=False, weights="imagenet", pooling='max', classes=NUM_CLASSES, classifier_activation="softmax")(inputs)
  outputs = tf.keras.layers.Dense(NUM_CLASSES, activation = tf.keras.activations.relu)(x)
  return tf.keras.Model(inputs=inputs, outputs=outputs)
```
#### Модель нейронной сети Train 11
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(None, 224, 224, 3)]     0
_________________________________________________________________
efficientnetb0 (Functional)  (None, 1280)              4049571
_________________________________________________________________
dense (Dense)                (None, 20)                25620
=================================================================
Total params: 4,075,191
Trainable params: 4,033,168
Non-trainable params: 42,023
_________________________________________________________________
```
#### [TensorBoard](https://tensorboard.dev/experiment/Eqp73r0eTpeWmJVNE2kQEg/#scalars) 
#### epoch_categorical_accuracy
<img src="https://raw.githubusercontent.com/NikitaShulgan/Laba2/main/for_Readme/Train_11_epoch_categorical_accuracy.svg">

#### epoch_loss
<img src="https://raw.githubusercontent.com/NikitaShulgan/Laba2/main/for_Readme/Train_11_epoch_loss.svg">

## Train 12
##### log file owl-1615533588.6146936
### Нейронная сеть [EfficientNet-B0](https://www.tensorflow.org/api_docs/python/tf/keras/applications/EfficientNetB0)  (предобученная на ImageNet), датасет [Oregon Wildlife](https://www.kaggle.com/virtualdvid/oregon-wildlife).
```
BATCH_SIZE = 64

def build_model():
  inputs = tf.keras.Input(shape=(RESIZE_TO, RESIZE_TO, 3))
  x = EfficientNetB0(include_top=False, weights="imagenet",input_shape=(RESIZE_TO, RESIZE_TO, 3), pooling='max', classes=NUM_CLASSES, classifier_activation="softmax")(inputs)
  x = tf.keras.layers.Dense(640, activation = tf.keras.activations.relu)(x)
  x = tf.keras.layers.Dense(640, activation = tf.keras.activations.relu)(x)
  outputs = tf.keras.layers.Dense(NUM_CLASSES, activation = tf.keras.activations.sigmoid)(x)
  return tf.keras.Model(inputs=inputs, outputs=outputs)
```
#### Модель нейронной сети Train 12
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(None, 224, 224, 3)]     0
_________________________________________________________________
efficientnetb0 (Functional)  (None, 1280)              4049571
_________________________________________________________________
dense (Dense)                (None, 640)               819840
_________________________________________________________________
dense_1 (Dense)              (None, 640)               410240
_________________________________________________________________
dense_2 (Dense)              (None, 20)                12820
=================================================================
Total params: 5,292,471
Trainable params: 5,250,448
Non-trainable params: 42,023
_________________________________________________________________
```
#### [TensorBoard](https://tensorboard.dev/experiment/LefdC788QnCbNT7PXGp1fg/#scalars) 
#### epoch_categorical_accuracy
<img src="https://raw.githubusercontent.com/NikitaShulgan/Laba2/main/for_Readme/Train_12_epoch_categorical_accuracy.svg">

#### epoch_loss
<img src="https://raw.githubusercontent.com/NikitaShulgan/Laba2/main/for_Readme/Train_12_epoch_loss.svg">

## Train 13
##### log file owl-1615536518.2523909
### Нейронная сеть [EfficientNet-B0](https://www.tensorflow.org/api_docs/python/tf/keras/applications/EfficientNetB0)  (предобученная на ImageNet), датасет [Oregon Wildlife](https://www.kaggle.com/virtualdvid/oregon-wildlife).
```
BATCH_SIZE = 64

def build_model():
  inputs = tf.keras.Input(shape=(RESIZE_TO, RESIZE_TO, 3))
  x = EfficientNetB0(include_top=False, weights="imagenet", pooling='max', classes=NUM_CLASSES, classifier_activation="softmax")(inputs)
  x = tf.keras.layers.Dense(128, activation = tf.keras.activations.relu)(x)
  x = tf.keras.layers.Dense(128, activation = tf.keras.activations.relu)(x)
  outputs = tf.keras.layers.Dense(NUM_CLASSES, activation = tf.keras.activations.sigmoid)(x)
  return tf.keras.Model(inputs=inputs, outputs=outputs)
```
#### Модель нейронной сети Train 13
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(None, 224, 224, 3)]     0
_________________________________________________________________
efficientnetb0 (Functional)  (None, 1280)              4049571
_________________________________________________________________
dense (Dense)                (None, 128)               163968
_________________________________________________________________
dense_1 (Dense)              (None, 128)               16512
_________________________________________________________________
dense_2 (Dense)              (None, 20)                2580
=================================================================
Total params: 4,232,631
Trainable params: 4,190,608
Non-trainable params: 42,023
_________________________________________________________________
```
#### [TensorBoard]() 
#### epoch_categorical_accuracy
<img src="https://raw.githubusercontent.com/NikitaShulgan/Laba2/main/for_Readme/Train_13_epoch_categorical_accuracy.svg">

#### epoch_loss
<img src="https://raw.githubusercontent.com/NikitaShulgan/Laba2/main/for_Readme/Train_13_epoch_loss.svg">

## Анализ полученных результатов
[Train 1](https://github.com/NikitaShulgan/Laba2#train-1) и [Train 2](https://github.com/NikitaShulgan/Laba2#train-2) ничем не лучше метода "Пальцем в небо" (у нас 20 видов картинок, т.е. вероятность угадать 5%), что мы можем видеть на графиках.
#### Links
https://sci-hub.se/10.1007/s13748-019-00203-0
