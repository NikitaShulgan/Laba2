# Лабораторная работа #2
## Решение задачи классификации изображений из набора данных [Oregon Wildlife](https://www.kaggle.com/virtualdvid/oregon-wildlife)  с использованием нейронных сетей глубокого обучения и техники обучения Transfer Learning

## EfficientNet-B0 architecture
![image](https://user-images.githubusercontent.com/80168174/110480321-6aae0900-80f7-11eb-82e6-f389f93c3966.png)


## Train 1
### Нейронная сеть EfficientNet-B0 (случайное начальное приближение), датасет [Oregon Wildlife](https://www.kaggle.com/virtualdvid/oregon-wildlife).

```
BATCH_SIZE = 16

def build_model():
  inputs = tf.keras.Input(shape=(RESIZE_TO, RESIZE_TO, 3))
  outputs = EfficientNetB0(weights=None, classes=NUM_CLASSES)(inputs)
  return tf.keras.Model(inputs=inputs, outputs=outputs)
```
#### Оранживая - обучающая выборка, Синия - валидационная выборка (на всех графиках в данном отчете)
#### https://tensorboard.dev/experiment/4EoeVqP1TLq6X8EG6GhRgw/#scalars
#### epoch_categorical_accuracy
<img src="https://raw.githubusercontent.com/NikitaShulgan/Laba2/main/for_Readme/epoch_categorical_accuracy_EfficientNet-B0_weights_None.svg">

#### epoch_loss
<img src="https://raw.githubusercontent.com/NikitaShulgan/Laba2/main/for_Readme/epoch_loss_EfficientNet-B0_weights_None.svg">


## Train 2
### Нейронная сеть EfficientNet-B0 (продобученная на ImageNet), датасет [Oregon Wildlife](https://www.kaggle.com/virtualdvid/oregon-wildlife).

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
#### https://tensorboard.dev/experiment/jbmjL062Ra6PiakXOYlaKA/#scalars
#### epoch_categorical_accuracy
<img src="https://raw.githubusercontent.com/NikitaShulgan/Laba2/main/for_Readme/epoch_categorical_accuracy_EfficientNet-B0_weights_ImageNet.svg">

#### epoch_loss
<img src="https://raw.githubusercontent.com/NikitaShulgan/Laba2/main/for_Readme/epoch_loss_EfficientNet-B0_weights_ImageNet.svg">

## Train 3 owl-1615469804.7533162
### Нейронная сеть EfficientNet-B0 (продобученная на ImageNet), датасет [Oregon Wildlife](https://www.kaggle.com/virtualdvid/oregon-wildlife).
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
#### https://tensorboard.dev/experiment/V0weQQ7rRPiqQmT9t6gBJA/#scalars
#### epoch_categorical_accuracy
<img src="https://raw.githubusercontent.com/NikitaShulgan/Laba2/main/for_Readme/Train_3_epoch_categorical_accuracy.svg">

#### epoch_loss
<img src="https://raw.githubusercontent.com/NikitaShulgan/Laba2/main/for_Readme/Train_3_epoch_loss.svg">

## Train 4 owl-1615473209.896111
### Нейронная сеть EfficientNet-B0 (продобученная на ImageNet), датасет [Oregon Wildlife](https://www.kaggle.com/virtualdvid/oregon-wildlife).
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
#### https://tensorboard.dev/experiment/kBQ9MhjJRuewgV982QFekA/#scalars
#### epoch_categorical_accuracy
<img src="https://raw.githubusercontent.com/NikitaShulgan/Laba2/main/for_Readme/Train_4_epoch_categorical_accuracy.svg">

#### epoch_loss
<img src="https://raw.githubusercontent.com/NikitaShulgan/Laba2/main/for_Readme/Train_4_epoch_loss.svg">

## Train 5
### Нейронная сеть EfficientNet-B0 (продобученная на ImageNet), датасет [Oregon Wildlife](https://www.kaggle.com/virtualdvid/oregon-wildlife).
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
#### 
#### epoch_categorical_accuracy
<img src="">

#### epoch_loss
<img src="">


## Анализ полученных результатов
[Train 1](https://github.com/NikitaShulgan/Laba2#train-1) и [Train 2](https://github.com/NikitaShulgan/Laba2#train-2) ничем не лучше метода "Пальцем в небо" (у нас 20 видов картинок, т.е. вероятность угадать 5%), что мы можем видеть на графиках.
#### Links
https://sci-hub.se/10.1007/s13748-019-00203-0
