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
#### [TensorBoard](https://tensorboard.dev/experiment/TQznWDFRQWybHyEkfXlaww/#scalars) 
#### epoch_categorical_accuracy
<img src="https://raw.githubusercontent.com/NikitaShulgan/Laba2/main/for_Readme/Train_13_epoch_categorical_accuracy.svg">

#### epoch_loss
<img src="https://raw.githubusercontent.com/NikitaShulgan/Laba2/main/for_Readme/Train_13_epoch_loss.svg">

## Train 14
##### log file owl-1615795574.7657037
### Нейронная сеть [EfficientNet-B0](https://www.tensorflow.org/api_docs/python/tf/keras/applications/EfficientNetB0)  (предобученная на ImageNet), датасет [Oregon Wildlife](https://www.kaggle.com/virtualdvid/oregon-wildlife).
```
BATCH_SIZE = 64

def build_model():
  inputs = tf.keras.Input(shape=(RESIZE_TO, RESIZE_TO, 3))
  x = EfficientNetB0(include_top=False, weights="imagenet", pooling='max', classes=NUM_CLASSES)(inputs)
  x.trainsble = False
  outputs = tf.keras.layers.Dense(NUM_CLASSES, activation = tf.keras.activations.relu)(x)
  return tf.keras.Model(inputs=inputs, outputs=outputs)
```
#### Модель нейронной сети Train 14
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
#### [TensorBoard](https://tensorboard.dev/experiment/kfo8JOvoS5KDaxO6S7C6rg/#scalars) 
#### epoch_categorical_accuracy
<img src="https://raw.githubusercontent.com/NikitaShulgan/Laba2/main/for_Readme/Train_14_epoch_categorical_accuracy.svg">

#### epoch_loss
<img src="https://raw.githubusercontent.com/NikitaShulgan/Laba2/main/for_Readme/Train_14_epoch_loss.svg">

## Train 15
##### log file owl-1615798466.590331
### Нейронная сеть [EfficientNet-B0](https://www.tensorflow.org/api_docs/python/tf/keras/applications/EfficientNetB0)  (предобученная на ImageNet), датасет [Oregon Wildlife](https://www.kaggle.com/virtualdvid/oregon-wildlife).
```
BATCH_SIZE = 32

def build_model():
  inputs = tf.keras.Input(shape=(RESIZE_TO, RESIZE_TO, 3))
  x = EfficientNetB0(include_top=False, weights="imagenet", pooling='avg', classes=NUM_CLASSES)(inputs)
  x.trainsble = False
  outputs = tf.keras.layers.Dense(NUM_CLASSES, activation = tf.keras.activations.softmax)(x)
  return tf.keras.Model(inputs=inputs, outputs=outputs)
```
#### Модель нейронной сети Train 15
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
#### [TensorBoard](https://tensorboard.dev/experiment/z71n8271R6qYtw6W4bd6EA/#scalars) 
#### epoch_categorical_accuracy
<img src="https://raw.githubusercontent.com/NikitaShulgan/Laba2/main/for_Readme/Train_15_epoch_categorical_accuracy.svg">

#### epoch_loss
<img src="https://raw.githubusercontent.com/NikitaShulgan/Laba2/main/for_Readme/Train_15_epoch_loss.svg">

## Train 16
##### log file owl-1615836017.2279189
### Нейронная сеть [EfficientNet-B0](https://www.tensorflow.org/api_docs/python/tf/keras/applications/EfficientNetB0)  (предобученная на ImageNet), датасет [Oregon Wildlife](https://www.kaggle.com/virtualdvid/oregon-wildlife).
```
BATCH_SIZE = 32

def build_model():
  inputs = tf.keras.Input(shape=(RESIZE_TO, RESIZE_TO, 3))
  x = EfficientNetB0(include_top=False, weights="imagenet")(inputs)
  x.trainable = False
  #model.trainable = False
  x = layers.GlobalAveragePooling2D()(x)
  outputs = tf.keras.layers.Dense(NUM_CLASSES, tf.keras.activations.softmax)(x)
  return tf.keras.Model(inputs=inputs, outputs=outputs)
  
  optimizer=tf.optimizers.Adam(lr=0.0001),
```
#### Модель нейронной сети Train 16
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
#### [TensorBoard](https://tensorboard.dev/experiment/Lc31h3UrTbWTGfaqqrypAQ/#scalars) 
#### epoch_categorical_accuracy
<img src="https://raw.githubusercontent.com/NikitaShulgan/Laba2/main/for_Readme/Train_16_epoch_categorical_accuracy.svg">

#### epoch_loss
<img src="https://raw.githubusercontent.com/NikitaShulgan/Laba2/main/for_Readme/Train_16_epoch_loss.svg">

## Train 17
##### log file owl-1615846371.8349254
### Нейронная сеть [EfficientNet-B0](https://www.tensorflow.org/api_docs/python/tf/keras/applications/EfficientNetB0)  (случайное начальное приближение), датасет [Oregon Wildlife](https://www.kaggle.com/virtualdvid/oregon-wildlife).
```
BATCH_SIZE = 8

def build_model():
  inputs = tf.keras.Input(shape=(RESIZE_TO, RESIZE_TO, 3))
  outputs = EfficientNetB0(weights=None, classes=NUM_CLASSES)(inputs)
  return tf.keras.Model(inputs=inputs, outputs=outputs)
  
  optimizer=tf.optimizers.Adam(lr=0.0001),
```
#### Модель нейронной сети Train 17
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(None, 224, 224, 3)]     0
_________________________________________________________________
efficientnetb0 (Functional)  (None, 20)                4075191
=================================================================
Total params: 4,075,191
Trainable params: 4,033,168
Non-trainable params: 42,023
_________________________________________________________________
```
#### [TensorBoard](https://tensorboard.dev/experiment/ioKg3gTRS5KwU8Gj3SJKgg/#scalars) 
#### epoch_categorical_accuracy
<img src="https://raw.githubusercontent.com/NikitaShulgan/Laba2/main/for_Readme/Train_17_epoch_categorical_accuracy.svg">

#### epoch_loss
<img src="https://raw.githubusercontent.com/NikitaShulgan/Laba2/main/for_Readme/Train_17_epoch_loss.svg">

## Train 1_1
##### log file owl-1615908752.3159616
### Нейронная сеть [EfficientNet-B0](https://www.tensorflow.org/api_docs/python/tf/keras/applications/EfficientNetB0)  (случайное начальное приближение), датасет [Oregon Wildlife](https://www.kaggle.com/virtualdvid/oregon-wildlife).
```
BATCH_SIZE = 32

def build_model():
  inputs = tf.keras.Input(shape=(RESIZE_TO, RESIZE_TO, 3))
  x = EfficientNetB0(include_top=False, input_tensor=inputs, weights="imagenet")
  x.trainable = False
  x = layers.GlobalAveragePooling2D()(x.output)
  outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
  return tf.keras.Model(inputs=inputs, outputs=outputs)  
```
#### Модель нейронной сети Train 1_1
```
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
input_1 (InputLayer)            [(None, 224, 224, 3) 0
__________________________________________________________________________________________________
rescaling (Rescaling)           (None, 224, 224, 3)  0           input_1[0][0]
__________________________________________________________________________________________________
normalization (Normalization)   (None, 224, 224, 3)  7           rescaling[0][0]
__________________________________________________________________________________________________
stem_conv_pad (ZeroPadding2D)   (None, 225, 225, 3)  0           normalization[0][0]
__________________________________________________________________________________________________
stem_conv (Conv2D)              (None, 112, 112, 32) 864         stem_conv_pad[0][0]
__________________________________________________________________________________________________
stem_bn (BatchNormalization)    (None, 112, 112, 32) 128         stem_conv[0][0]
__________________________________________________________________________________________________
stem_activation (Activation)    (None, 112, 112, 32) 0           stem_bn[0][0]
__________________________________________________________________________________________________
block1a_dwconv (DepthwiseConv2D (None, 112, 112, 32) 288         stem_activation[0][0]
__________________________________________________________________________________________________
block1a_bn (BatchNormalization) (None, 112, 112, 32) 128         block1a_dwconv[0][0]
__________________________________________________________________________________________________
block1a_activation (Activation) (None, 112, 112, 32) 0           block1a_bn[0][0]
__________________________________________________________________________________________________
block1a_se_squeeze (GlobalAvera (None, 32)           0           block1a_activation[0][0]
__________________________________________________________________________________________________
block1a_se_reshape (Reshape)    (None, 1, 1, 32)     0           block1a_se_squeeze[0][0]
__________________________________________________________________________________________________
block1a_se_reduce (Conv2D)      (None, 1, 1, 8)      264         block1a_se_reshape[0][0]
__________________________________________________________________________________________________
block1a_se_expand (Conv2D)      (None, 1, 1, 32)     288         block1a_se_reduce[0][0]
__________________________________________________________________________________________________
block1a_se_excite (Multiply)    (None, 112, 112, 32) 0           block1a_activation[0][0]
                                                                 block1a_se_expand[0][0]
__________________________________________________________________________________________________
block1a_project_conv (Conv2D)   (None, 112, 112, 16) 512         block1a_se_excite[0][0]
__________________________________________________________________________________________________
block1a_project_bn (BatchNormal (None, 112, 112, 16) 64          block1a_project_conv[0][0]
__________________________________________________________________________________________________
block2a_expand_conv (Conv2D)    (None, 112, 112, 96) 1536        block1a_project_bn[0][0]
__________________________________________________________________________________________________
block2a_expand_bn (BatchNormali (None, 112, 112, 96) 384         block2a_expand_conv[0][0]
__________________________________________________________________________________________________
block2a_expand_activation (Acti (None, 112, 112, 96) 0           block2a_expand_bn[0][0]
__________________________________________________________________________________________________
block2a_dwconv_pad (ZeroPadding (None, 113, 113, 96) 0           block2a_expand_activation[0][0]
__________________________________________________________________________________________________
block2a_dwconv (DepthwiseConv2D (None, 56, 56, 96)   864         block2a_dwconv_pad[0][0]
__________________________________________________________________________________________________
block2a_bn (BatchNormalization) (None, 56, 56, 96)   384         block2a_dwconv[0][0]
__________________________________________________________________________________________________
block2a_activation (Activation) (None, 56, 56, 96)   0           block2a_bn[0][0]
__________________________________________________________________________________________________
block2a_se_squeeze (GlobalAvera (None, 96)           0           block2a_activation[0][0]
__________________________________________________________________________________________________
block2a_se_reshape (Reshape)    (None, 1, 1, 96)     0           block2a_se_squeeze[0][0]
__________________________________________________________________________________________________
block2a_se_reduce (Conv2D)      (None, 1, 1, 4)      388         block2a_se_reshape[0][0]
__________________________________________________________________________________________________
block2a_se_expand (Conv2D)      (None, 1, 1, 96)     480         block2a_se_reduce[0][0]
__________________________________________________________________________________________________
block2a_se_excite (Multiply)    (None, 56, 56, 96)   0           block2a_activation[0][0]
                                                                 block2a_se_expand[0][0]
__________________________________________________________________________________________________
block2a_project_conv (Conv2D)   (None, 56, 56, 24)   2304        block2a_se_excite[0][0]
__________________________________________________________________________________________________
block2a_project_bn (BatchNormal (None, 56, 56, 24)   96          block2a_project_conv[0][0]
__________________________________________________________________________________________________
block2b_expand_conv (Conv2D)    (None, 56, 56, 144)  3456        block2a_project_bn[0][0]
__________________________________________________________________________________________________
block2b_expand_bn (BatchNormali (None, 56, 56, 144)  576         block2b_expand_conv[0][0]
__________________________________________________________________________________________________
block2b_expand_activation (Acti (None, 56, 56, 144)  0           block2b_expand_bn[0][0]
__________________________________________________________________________________________________
block2b_dwconv (DepthwiseConv2D (None, 56, 56, 144)  1296        block2b_expand_activation[0][0]
__________________________________________________________________________________________________
block2b_bn (BatchNormalization) (None, 56, 56, 144)  576         block2b_dwconv[0][0]
__________________________________________________________________________________________________
block2b_activation (Activation) (None, 56, 56, 144)  0           block2b_bn[0][0]
__________________________________________________________________________________________________
block2b_se_squeeze (GlobalAvera (None, 144)          0           block2b_activation[0][0]
__________________________________________________________________________________________________
block2b_se_reshape (Reshape)    (None, 1, 1, 144)    0           block2b_se_squeeze[0][0]
__________________________________________________________________________________________________
block2b_se_reduce (Conv2D)      (None, 1, 1, 6)      870         block2b_se_reshape[0][0]
__________________________________________________________________________________________________
block2b_se_expand (Conv2D)      (None, 1, 1, 144)    1008        block2b_se_reduce[0][0]
__________________________________________________________________________________________________
block2b_se_excite (Multiply)    (None, 56, 56, 144)  0           block2b_activation[0][0]
                                                                 block2b_se_expand[0][0]
__________________________________________________________________________________________________
block2b_project_conv (Conv2D)   (None, 56, 56, 24)   3456        block2b_se_excite[0][0]
__________________________________________________________________________________________________
block2b_project_bn (BatchNormal (None, 56, 56, 24)   96          block2b_project_conv[0][0]
__________________________________________________________________________________________________
block2b_drop (Dropout)          (None, 56, 56, 24)   0           block2b_project_bn[0][0]
__________________________________________________________________________________________________
block2b_add (Add)               (None, 56, 56, 24)   0           block2b_drop[0][0]
                                                                 block2a_project_bn[0][0]
__________________________________________________________________________________________________
block3a_expand_conv (Conv2D)    (None, 56, 56, 144)  3456        block2b_add[0][0]
__________________________________________________________________________________________________
block3a_expand_bn (BatchNormali (None, 56, 56, 144)  576         block3a_expand_conv[0][0]
__________________________________________________________________________________________________
block3a_expand_activation (Acti (None, 56, 56, 144)  0           block3a_expand_bn[0][0]
__________________________________________________________________________________________________
block3a_dwconv_pad (ZeroPadding (None, 59, 59, 144)  0           block3a_expand_activation[0][0]
__________________________________________________________________________________________________
block3a_dwconv (DepthwiseConv2D (None, 28, 28, 144)  3600        block3a_dwconv_pad[0][0]
__________________________________________________________________________________________________
block3a_bn (BatchNormalization) (None, 28, 28, 144)  576         block3a_dwconv[0][0]
__________________________________________________________________________________________________
block3a_activation (Activation) (None, 28, 28, 144)  0           block3a_bn[0][0]
__________________________________________________________________________________________________
block3a_se_squeeze (GlobalAvera (None, 144)          0           block3a_activation[0][0]
__________________________________________________________________________________________________
block3a_se_reshape (Reshape)    (None, 1, 1, 144)    0           block3a_se_squeeze[0][0]
__________________________________________________________________________________________________
block3a_se_reduce (Conv2D)      (None, 1, 1, 6)      870         block3a_se_reshape[0][0]
__________________________________________________________________________________________________
block3a_se_expand (Conv2D)      (None, 1, 1, 144)    1008        block3a_se_reduce[0][0]
__________________________________________________________________________________________________
block3a_se_excite (Multiply)    (None, 28, 28, 144)  0           block3a_activation[0][0]
                                                                 block3a_se_expand[0][0]
__________________________________________________________________________________________________
block3a_project_conv (Conv2D)   (None, 28, 28, 40)   5760        block3a_se_excite[0][0]
__________________________________________________________________________________________________
block3a_project_bn (BatchNormal (None, 28, 28, 40)   160         block3a_project_conv[0][0]
__________________________________________________________________________________________________
block3b_expand_conv (Conv2D)    (None, 28, 28, 240)  9600        block3a_project_bn[0][0]
__________________________________________________________________________________________________
block3b_expand_bn (BatchNormali (None, 28, 28, 240)  960         block3b_expand_conv[0][0]
__________________________________________________________________________________________________
block3b_expand_activation (Acti (None, 28, 28, 240)  0           block3b_expand_bn[0][0]
__________________________________________________________________________________________________
block3b_dwconv (DepthwiseConv2D (None, 28, 28, 240)  6000        block3b_expand_activation[0][0]
__________________________________________________________________________________________________
block3b_bn (BatchNormalization) (None, 28, 28, 240)  960         block3b_dwconv[0][0]
__________________________________________________________________________________________________
block3b_activation (Activation) (None, 28, 28, 240)  0           block3b_bn[0][0]
__________________________________________________________________________________________________
block3b_se_squeeze (GlobalAvera (None, 240)          0           block3b_activation[0][0]
__________________________________________________________________________________________________
block3b_se_reshape (Reshape)    (None, 1, 1, 240)    0           block3b_se_squeeze[0][0]
__________________________________________________________________________________________________
block3b_se_reduce (Conv2D)      (None, 1, 1, 10)     2410        block3b_se_reshape[0][0]
__________________________________________________________________________________________________
block3b_se_expand (Conv2D)      (None, 1, 1, 240)    2640        block3b_se_reduce[0][0]
__________________________________________________________________________________________________
block3b_se_excite (Multiply)    (None, 28, 28, 240)  0           block3b_activation[0][0]
                                                                 block3b_se_expand[0][0]
__________________________________________________________________________________________________
block3b_project_conv (Conv2D)   (None, 28, 28, 40)   9600        block3b_se_excite[0][0]
__________________________________________________________________________________________________
block3b_project_bn (BatchNormal (None, 28, 28, 40)   160         block3b_project_conv[0][0]
__________________________________________________________________________________________________
block3b_drop (Dropout)          (None, 28, 28, 40)   0           block3b_project_bn[0][0]
__________________________________________________________________________________________________
block3b_add (Add)               (None, 28, 28, 40)   0           block3b_drop[0][0]
                                                                 block3a_project_bn[0][0]
__________________________________________________________________________________________________
block4a_expand_conv (Conv2D)    (None, 28, 28, 240)  9600        block3b_add[0][0]
__________________________________________________________________________________________________
block4a_expand_bn (BatchNormali (None, 28, 28, 240)  960         block4a_expand_conv[0][0]
__________________________________________________________________________________________________
block4a_expand_activation (Acti (None, 28, 28, 240)  0           block4a_expand_bn[0][0]
__________________________________________________________________________________________________
block4a_dwconv_pad (ZeroPadding (None, 29, 29, 240)  0           block4a_expand_activation[0][0]
__________________________________________________________________________________________________
block4a_dwconv (DepthwiseConv2D (None, 14, 14, 240)  2160        block4a_dwconv_pad[0][0]
__________________________________________________________________________________________________
block4a_bn (BatchNormalization) (None, 14, 14, 240)  960         block4a_dwconv[0][0]
__________________________________________________________________________________________________
block4a_activation (Activation) (None, 14, 14, 240)  0           block4a_bn[0][0]
__________________________________________________________________________________________________
block4a_se_squeeze (GlobalAvera (None, 240)          0           block4a_activation[0][0]
__________________________________________________________________________________________________
block4a_se_reshape (Reshape)    (None, 1, 1, 240)    0           block4a_se_squeeze[0][0]
__________________________________________________________________________________________________
block4a_se_reduce (Conv2D)      (None, 1, 1, 10)     2410        block4a_se_reshape[0][0]
__________________________________________________________________________________________________
block4a_se_expand (Conv2D)      (None, 1, 1, 240)    2640        block4a_se_reduce[0][0]
__________________________________________________________________________________________________
block4a_se_excite (Multiply)    (None, 14, 14, 240)  0           block4a_activation[0][0]
                                                                 block4a_se_expand[0][0]
__________________________________________________________________________________________________
block4a_project_conv (Conv2D)   (None, 14, 14, 80)   19200       block4a_se_excite[0][0]
__________________________________________________________________________________________________
block4a_project_bn (BatchNormal (None, 14, 14, 80)   320         block4a_project_conv[0][0]
__________________________________________________________________________________________________
block4b_expand_conv (Conv2D)    (None, 14, 14, 480)  38400       block4a_project_bn[0][0]
__________________________________________________________________________________________________
block4b_expand_bn (BatchNormali (None, 14, 14, 480)  1920        block4b_expand_conv[0][0]
__________________________________________________________________________________________________
block4b_expand_activation (Acti (None, 14, 14, 480)  0           block4b_expand_bn[0][0]
__________________________________________________________________________________________________
block4b_dwconv (DepthwiseConv2D (None, 14, 14, 480)  4320        block4b_expand_activation[0][0]
__________________________________________________________________________________________________
block4b_bn (BatchNormalization) (None, 14, 14, 480)  1920        block4b_dwconv[0][0]
__________________________________________________________________________________________________
block4b_activation (Activation) (None, 14, 14, 480)  0           block4b_bn[0][0]
__________________________________________________________________________________________________
block4b_se_squeeze (GlobalAvera (None, 480)          0           block4b_activation[0][0]
__________________________________________________________________________________________________
block4b_se_reshape (Reshape)    (None, 1, 1, 480)    0           block4b_se_squeeze[0][0]
__________________________________________________________________________________________________
block4b_se_reduce (Conv2D)      (None, 1, 1, 20)     9620        block4b_se_reshape[0][0]
__________________________________________________________________________________________________
block4b_se_expand (Conv2D)      (None, 1, 1, 480)    10080       block4b_se_reduce[0][0]
__________________________________________________________________________________________________
block4b_se_excite (Multiply)    (None, 14, 14, 480)  0           block4b_activation[0][0]
                                                                 block4b_se_expand[0][0]
__________________________________________________________________________________________________
block4b_project_conv (Conv2D)   (None, 14, 14, 80)   38400       block4b_se_excite[0][0]
__________________________________________________________________________________________________
block4b_project_bn (BatchNormal (None, 14, 14, 80)   320         block4b_project_conv[0][0]
__________________________________________________________________________________________________
block4b_drop (Dropout)          (None, 14, 14, 80)   0           block4b_project_bn[0][0]
__________________________________________________________________________________________________
block4b_add (Add)               (None, 14, 14, 80)   0           block4b_drop[0][0]
                                                                 block4a_project_bn[0][0]
__________________________________________________________________________________________________
block4c_expand_conv (Conv2D)    (None, 14, 14, 480)  38400       block4b_add[0][0]
__________________________________________________________________________________________________
block4c_expand_bn (BatchNormali (None, 14, 14, 480)  1920        block4c_expand_conv[0][0]
__________________________________________________________________________________________________
block4c_expand_activation (Acti (None, 14, 14, 480)  0           block4c_expand_bn[0][0]
__________________________________________________________________________________________________
block4c_dwconv (DepthwiseConv2D (None, 14, 14, 480)  4320        block4c_expand_activation[0][0]
__________________________________________________________________________________________________
block4c_bn (BatchNormalization) (None, 14, 14, 480)  1920        block4c_dwconv[0][0]
__________________________________________________________________________________________________
block4c_activation (Activation) (None, 14, 14, 480)  0           block4c_bn[0][0]
__________________________________________________________________________________________________
block4c_se_squeeze (GlobalAvera (None, 480)          0           block4c_activation[0][0]
__________________________________________________________________________________________________
block4c_se_reshape (Reshape)    (None, 1, 1, 480)    0           block4c_se_squeeze[0][0]
__________________________________________________________________________________________________
block4c_se_reduce (Conv2D)      (None, 1, 1, 20)     9620        block4c_se_reshape[0][0]
__________________________________________________________________________________________________
block4c_se_expand (Conv2D)      (None, 1, 1, 480)    10080       block4c_se_reduce[0][0]
__________________________________________________________________________________________________
block4c_se_excite (Multiply)    (None, 14, 14, 480)  0           block4c_activation[0][0]
                                                                 block4c_se_expand[0][0]
__________________________________________________________________________________________________
block4c_project_conv (Conv2D)   (None, 14, 14, 80)   38400       block4c_se_excite[0][0]
__________________________________________________________________________________________________
block4c_project_bn (BatchNormal (None, 14, 14, 80)   320         block4c_project_conv[0][0]
__________________________________________________________________________________________________
block4c_drop (Dropout)          (None, 14, 14, 80)   0           block4c_project_bn[0][0]
__________________________________________________________________________________________________
block4c_add (Add)               (None, 14, 14, 80)   0           block4c_drop[0][0]
                                                                 block4b_add[0][0]
__________________________________________________________________________________________________
block5a_expand_conv (Conv2D)    (None, 14, 14, 480)  38400       block4c_add[0][0]
__________________________________________________________________________________________________
block5a_expand_bn (BatchNormali (None, 14, 14, 480)  1920        block5a_expand_conv[0][0]
__________________________________________________________________________________________________
block5a_expand_activation (Acti (None, 14, 14, 480)  0           block5a_expand_bn[0][0]
__________________________________________________________________________________________________
block5a_dwconv (DepthwiseConv2D (None, 14, 14, 480)  12000       block5a_expand_activation[0][0]
__________________________________________________________________________________________________
block5a_bn (BatchNormalization) (None, 14, 14, 480)  1920        block5a_dwconv[0][0]
__________________________________________________________________________________________________
block5a_activation (Activation) (None, 14, 14, 480)  0           block5a_bn[0][0]
__________________________________________________________________________________________________
block5a_se_squeeze (GlobalAvera (None, 480)          0           block5a_activation[0][0]
__________________________________________________________________________________________________
block5a_se_reshape (Reshape)    (None, 1, 1, 480)    0           block5a_se_squeeze[0][0]
__________________________________________________________________________________________________
block5a_se_reduce (Conv2D)      (None, 1, 1, 20)     9620        block5a_se_reshape[0][0]
__________________________________________________________________________________________________
block5a_se_expand (Conv2D)      (None, 1, 1, 480)    10080       block5a_se_reduce[0][0]
__________________________________________________________________________________________________
block5a_se_excite (Multiply)    (None, 14, 14, 480)  0           block5a_activation[0][0]
                                                                 block5a_se_expand[0][0]
__________________________________________________________________________________________________
block5a_project_conv (Conv2D)   (None, 14, 14, 112)  53760       block5a_se_excite[0][0]
__________________________________________________________________________________________________
block5a_project_bn (BatchNormal (None, 14, 14, 112)  448         block5a_project_conv[0][0]
__________________________________________________________________________________________________
block5b_expand_conv (Conv2D)    (None, 14, 14, 672)  75264       block5a_project_bn[0][0]
__________________________________________________________________________________________________
block5b_expand_bn (BatchNormali (None, 14, 14, 672)  2688        block5b_expand_conv[0][0]
__________________________________________________________________________________________________
block5b_expand_activation (Acti (None, 14, 14, 672)  0           block5b_expand_bn[0][0]
__________________________________________________________________________________________________
block5b_dwconv (DepthwiseConv2D (None, 14, 14, 672)  16800       block5b_expand_activation[0][0]
__________________________________________________________________________________________________
block5b_bn (BatchNormalization) (None, 14, 14, 672)  2688        block5b_dwconv[0][0]
__________________________________________________________________________________________________
block5b_activation (Activation) (None, 14, 14, 672)  0           block5b_bn[0][0]
__________________________________________________________________________________________________
block5b_se_squeeze (GlobalAvera (None, 672)          0           block5b_activation[0][0]
__________________________________________________________________________________________________
block5b_se_reshape (Reshape)    (None, 1, 1, 672)    0           block5b_se_squeeze[0][0]
__________________________________________________________________________________________________
block5b_se_reduce (Conv2D)      (None, 1, 1, 28)     18844       block5b_se_reshape[0][0]
__________________________________________________________________________________________________
block5b_se_expand (Conv2D)      (None, 1, 1, 672)    19488       block5b_se_reduce[0][0]
__________________________________________________________________________________________________
block5b_se_excite (Multiply)    (None, 14, 14, 672)  0           block5b_activation[0][0]
                                                                 block5b_se_expand[0][0]
__________________________________________________________________________________________________
block5b_project_conv (Conv2D)   (None, 14, 14, 112)  75264       block5b_se_excite[0][0]
__________________________________________________________________________________________________
block5b_project_bn (BatchNormal (None, 14, 14, 112)  448         block5b_project_conv[0][0]
__________________________________________________________________________________________________
block5b_drop (Dropout)          (None, 14, 14, 112)  0           block5b_project_bn[0][0]
__________________________________________________________________________________________________
block5b_add (Add)               (None, 14, 14, 112)  0           block5b_drop[0][0]
                                                                 block5a_project_bn[0][0]
__________________________________________________________________________________________________
block5c_expand_conv (Conv2D)    (None, 14, 14, 672)  75264       block5b_add[0][0]
__________________________________________________________________________________________________
block5c_expand_bn (BatchNormali (None, 14, 14, 672)  2688        block5c_expand_conv[0][0]
__________________________________________________________________________________________________
block5c_expand_activation (Acti (None, 14, 14, 672)  0           block5c_expand_bn[0][0]
__________________________________________________________________________________________________
block5c_dwconv (DepthwiseConv2D (None, 14, 14, 672)  16800       block5c_expand_activation[0][0]
__________________________________________________________________________________________________
block5c_bn (BatchNormalization) (None, 14, 14, 672)  2688        block5c_dwconv[0][0]
__________________________________________________________________________________________________
block5c_activation (Activation) (None, 14, 14, 672)  0           block5c_bn[0][0]
__________________________________________________________________________________________________
block5c_se_squeeze (GlobalAvera (None, 672)          0           block5c_activation[0][0]
__________________________________________________________________________________________________
block5c_se_reshape (Reshape)    (None, 1, 1, 672)    0           block5c_se_squeeze[0][0]
__________________________________________________________________________________________________
block5c_se_reduce (Conv2D)      (None, 1, 1, 28)     18844       block5c_se_reshape[0][0]
__________________________________________________________________________________________________
block5c_se_expand (Conv2D)      (None, 1, 1, 672)    19488       block5c_se_reduce[0][0]
__________________________________________________________________________________________________
block5c_se_excite (Multiply)    (None, 14, 14, 672)  0           block5c_activation[0][0]
                                                                 block5c_se_expand[0][0]
__________________________________________________________________________________________________
block5c_project_conv (Conv2D)   (None, 14, 14, 112)  75264       block5c_se_excite[0][0]
__________________________________________________________________________________________________
block5c_project_bn (BatchNormal (None, 14, 14, 112)  448         block5c_project_conv[0][0]
__________________________________________________________________________________________________
block5c_drop (Dropout)          (None, 14, 14, 112)  0           block5c_project_bn[0][0]
__________________________________________________________________________________________________
block5c_add (Add)               (None, 14, 14, 112)  0           block5c_drop[0][0]
                                                                 block5b_add[0][0]
__________________________________________________________________________________________________
block6a_expand_conv (Conv2D)    (None, 14, 14, 672)  75264       block5c_add[0][0]
__________________________________________________________________________________________________
block6a_expand_bn (BatchNormali (None, 14, 14, 672)  2688        block6a_expand_conv[0][0]
__________________________________________________________________________________________________
block6a_expand_activation (Acti (None, 14, 14, 672)  0           block6a_expand_bn[0][0]
__________________________________________________________________________________________________
block6a_dwconv_pad (ZeroPadding (None, 17, 17, 672)  0           block6a_expand_activation[0][0]
__________________________________________________________________________________________________
block6a_dwconv (DepthwiseConv2D (None, 7, 7, 672)    16800       block6a_dwconv_pad[0][0]
__________________________________________________________________________________________________
block6a_bn (BatchNormalization) (None, 7, 7, 672)    2688        block6a_dwconv[0][0]
__________________________________________________________________________________________________
block6a_activation (Activation) (None, 7, 7, 672)    0           block6a_bn[0][0]
__________________________________________________________________________________________________
block6a_se_squeeze (GlobalAvera (None, 672)          0           block6a_activation[0][0]
__________________________________________________________________________________________________
block6a_se_reshape (Reshape)    (None, 1, 1, 672)    0           block6a_se_squeeze[0][0]
__________________________________________________________________________________________________
block6a_se_reduce (Conv2D)      (None, 1, 1, 28)     18844       block6a_se_reshape[0][0]
__________________________________________________________________________________________________
block6a_se_expand (Conv2D)      (None, 1, 1, 672)    19488       block6a_se_reduce[0][0]
__________________________________________________________________________________________________
block6a_se_excite (Multiply)    (None, 7, 7, 672)    0           block6a_activation[0][0]
                                                                 block6a_se_expand[0][0]
__________________________________________________________________________________________________
block6a_project_conv (Conv2D)   (None, 7, 7, 192)    129024      block6a_se_excite[0][0]
__________________________________________________________________________________________________
block6a_project_bn (BatchNormal (None, 7, 7, 192)    768         block6a_project_conv[0][0]
__________________________________________________________________________________________________
block6b_expand_conv (Conv2D)    (None, 7, 7, 1152)   221184      block6a_project_bn[0][0]
__________________________________________________________________________________________________
block6b_expand_bn (BatchNormali (None, 7, 7, 1152)   4608        block6b_expand_conv[0][0]
__________________________________________________________________________________________________
block6b_expand_activation (Acti (None, 7, 7, 1152)   0           block6b_expand_bn[0][0]
__________________________________________________________________________________________________
block6b_dwconv (DepthwiseConv2D (None, 7, 7, 1152)   28800       block6b_expand_activation[0][0]
__________________________________________________________________________________________________
block6b_bn (BatchNormalization) (None, 7, 7, 1152)   4608        block6b_dwconv[0][0]
__________________________________________________________________________________________________
block6b_activation (Activation) (None, 7, 7, 1152)   0           block6b_bn[0][0]
__________________________________________________________________________________________________
block6b_se_squeeze (GlobalAvera (None, 1152)         0           block6b_activation[0][0]
__________________________________________________________________________________________________
block6b_se_reshape (Reshape)    (None, 1, 1, 1152)   0           block6b_se_squeeze[0][0]
__________________________________________________________________________________________________
block6b_se_reduce (Conv2D)      (None, 1, 1, 48)     55344       block6b_se_reshape[0][0]
__________________________________________________________________________________________________
block6b_se_expand (Conv2D)      (None, 1, 1, 1152)   56448       block6b_se_reduce[0][0]
__________________________________________________________________________________________________
block6b_se_excite (Multiply)    (None, 7, 7, 1152)   0           block6b_activation[0][0]
                                                                 block6b_se_expand[0][0]
__________________________________________________________________________________________________
block6b_project_conv (Conv2D)   (None, 7, 7, 192)    221184      block6b_se_excite[0][0]
__________________________________________________________________________________________________
block6b_project_bn (BatchNormal (None, 7, 7, 192)    768         block6b_project_conv[0][0]
__________________________________________________________________________________________________
block6b_drop (Dropout)          (None, 7, 7, 192)    0           block6b_project_bn[0][0]
__________________________________________________________________________________________________
block6b_add (Add)               (None, 7, 7, 192)    0           block6b_drop[0][0]
                                                                 block6a_project_bn[0][0]
__________________________________________________________________________________________________
block6c_expand_conv (Conv2D)    (None, 7, 7, 1152)   221184      block6b_add[0][0]
__________________________________________________________________________________________________
block6c_expand_bn (BatchNormali (None, 7, 7, 1152)   4608        block6c_expand_conv[0][0]
__________________________________________________________________________________________________
block6c_expand_activation (Acti (None, 7, 7, 1152)   0           block6c_expand_bn[0][0]
__________________________________________________________________________________________________
block6c_dwconv (DepthwiseConv2D (None, 7, 7, 1152)   28800       block6c_expand_activation[0][0]
__________________________________________________________________________________________________
block6c_bn (BatchNormalization) (None, 7, 7, 1152)   4608        block6c_dwconv[0][0]
__________________________________________________________________________________________________
block6c_activation (Activation) (None, 7, 7, 1152)   0           block6c_bn[0][0]
__________________________________________________________________________________________________
block6c_se_squeeze (GlobalAvera (None, 1152)         0           block6c_activation[0][0]
__________________________________________________________________________________________________
block6c_se_reshape (Reshape)    (None, 1, 1, 1152)   0           block6c_se_squeeze[0][0]
__________________________________________________________________________________________________
block6c_se_reduce (Conv2D)      (None, 1, 1, 48)     55344       block6c_se_reshape[0][0]
__________________________________________________________________________________________________
block6c_se_expand (Conv2D)      (None, 1, 1, 1152)   56448       block6c_se_reduce[0][0]
__________________________________________________________________________________________________
block6c_se_excite (Multiply)    (None, 7, 7, 1152)   0           block6c_activation[0][0]
                                                                 block6c_se_expand[0][0]
__________________________________________________________________________________________________
block6c_project_conv (Conv2D)   (None, 7, 7, 192)    221184      block6c_se_excite[0][0]
__________________________________________________________________________________________________
block6c_project_bn (BatchNormal (None, 7, 7, 192)    768         block6c_project_conv[0][0]
__________________________________________________________________________________________________
block6c_drop (Dropout)          (None, 7, 7, 192)    0           block6c_project_bn[0][0]
__________________________________________________________________________________________________
block6c_add (Add)               (None, 7, 7, 192)    0           block6c_drop[0][0]
                                                                 block6b_add[0][0]
__________________________________________________________________________________________________
block6d_expand_conv (Conv2D)    (None, 7, 7, 1152)   221184      block6c_add[0][0]
__________________________________________________________________________________________________
block6d_expand_bn (BatchNormali (None, 7, 7, 1152)   4608        block6d_expand_conv[0][0]
__________________________________________________________________________________________________
block6d_expand_activation (Acti (None, 7, 7, 1152)   0           block6d_expand_bn[0][0]
__________________________________________________________________________________________________
block6d_dwconv (DepthwiseConv2D (None, 7, 7, 1152)   28800       block6d_expand_activation[0][0]
__________________________________________________________________________________________________
block6d_bn (BatchNormalization) (None, 7, 7, 1152)   4608        block6d_dwconv[0][0]
__________________________________________________________________________________________________
block6d_activation (Activation) (None, 7, 7, 1152)   0           block6d_bn[0][0]
__________________________________________________________________________________________________
block6d_se_squeeze (GlobalAvera (None, 1152)         0           block6d_activation[0][0]
__________________________________________________________________________________________________
block6d_se_reshape (Reshape)    (None, 1, 1, 1152)   0           block6d_se_squeeze[0][0]
__________________________________________________________________________________________________
block6d_se_reduce (Conv2D)      (None, 1, 1, 48)     55344       block6d_se_reshape[0][0]
__________________________________________________________________________________________________
block6d_se_expand (Conv2D)      (None, 1, 1, 1152)   56448       block6d_se_reduce[0][0]
__________________________________________________________________________________________________
block6d_se_excite (Multiply)    (None, 7, 7, 1152)   0           block6d_activation[0][0]
                                                                 block6d_se_expand[0][0]
__________________________________________________________________________________________________
block6d_project_conv (Conv2D)   (None, 7, 7, 192)    221184      block6d_se_excite[0][0]
__________________________________________________________________________________________________
block6d_project_bn (BatchNormal (None, 7, 7, 192)    768         block6d_project_conv[0][0]
__________________________________________________________________________________________________
block6d_drop (Dropout)          (None, 7, 7, 192)    0           block6d_project_bn[0][0]
__________________________________________________________________________________________________
block6d_add (Add)               (None, 7, 7, 192)    0           block6d_drop[0][0]
                                                                 block6c_add[0][0]
__________________________________________________________________________________________________
block7a_expand_conv (Conv2D)    (None, 7, 7, 1152)   221184      block6d_add[0][0]
__________________________________________________________________________________________________
block7a_expand_bn (BatchNormali (None, 7, 7, 1152)   4608        block7a_expand_conv[0][0]
__________________________________________________________________________________________________
block7a_expand_activation (Acti (None, 7, 7, 1152)   0           block7a_expand_bn[0][0]
__________________________________________________________________________________________________
block7a_dwconv (DepthwiseConv2D (None, 7, 7, 1152)   10368       block7a_expand_activation[0][0]
__________________________________________________________________________________________________
block7a_bn (BatchNormalization) (None, 7, 7, 1152)   4608        block7a_dwconv[0][0]
__________________________________________________________________________________________________
block7a_activation (Activation) (None, 7, 7, 1152)   0           block7a_bn[0][0]
__________________________________________________________________________________________________
block7a_se_squeeze (GlobalAvera (None, 1152)         0           block7a_activation[0][0]
__________________________________________________________________________________________________
block7a_se_reshape (Reshape)    (None, 1, 1, 1152)   0           block7a_se_squeeze[0][0]
__________________________________________________________________________________________________
block7a_se_reduce (Conv2D)      (None, 1, 1, 48)     55344       block7a_se_reshape[0][0]
__________________________________________________________________________________________________
block7a_se_expand (Conv2D)      (None, 1, 1, 1152)   56448       block7a_se_reduce[0][0]
__________________________________________________________________________________________________
block7a_se_excite (Multiply)    (None, 7, 7, 1152)   0           block7a_activation[0][0]
                                                                 block7a_se_expand[0][0]
__________________________________________________________________________________________________
block7a_project_conv (Conv2D)   (None, 7, 7, 320)    368640      block7a_se_excite[0][0]
__________________________________________________________________________________________________
block7a_project_bn (BatchNormal (None, 7, 7, 320)    1280        block7a_project_conv[0][0]
__________________________________________________________________________________________________
top_conv (Conv2D)               (None, 7, 7, 1280)   409600      block7a_project_bn[0][0]
__________________________________________________________________________________________________
top_bn (BatchNormalization)     (None, 7, 7, 1280)   5120        top_conv[0][0]
__________________________________________________________________________________________________
top_activation (Activation)     (None, 7, 7, 1280)   0           top_bn[0][0]
__________________________________________________________________________________________________
global_average_pooling2d (Globa (None, 1280)         0           top_activation[0][0]
__________________________________________________________________________________________________
dense (Dense)                   (None, 20)           25620       global_average_pooling2d[0][0]
==================================================================================================
Total params: 4,075,191
Trainable params: 25,620
Non-trainable params: 4,049,571
__________________________________________________________________________________________________

```
#### [TensorBoard](https://tensorboard.dev/experiment/hMNO868KRoeMHEuRqzImPA/#scalars) 
#### epoch_categorical_accuracy
<img src="https://raw.githubusercontent.com/NikitaShulgan/Laba2/main/for_Readme/Train_1_1_epoch_categorical_accuracy.svg">

#### epoch_loss
<img src="https://raw.githubusercontent.com/NikitaShulgan/Laba2/main/for_Readme/Train_1_1_epoch_loss.svg">


## Анализ полученных результатов
[Train 1](https://github.com/NikitaShulgan/Laba2#train-1), [Train 2](https://github.com/NikitaShulgan/Laba2#train-2), [Train 3](https://github.com/NikitaShulgan/Laba2#train-3), [Train 12](https://github.com/NikitaShulgan/Laba2#train-12), [Train 13](https://github.com/NikitaShulgan/Laba2#train-13) ничем не лучше метода "Пальцем в небо" (у нас 20 видов картинок, т.е. вероятность угадать 5%), что мы можем видеть на графиках. Убрав слой ``` Dense``` в [Train 3](https://github.com/NikitaShulgan/Laba2#train-3) я смог уменьшить epoch_loss в 1000000 раз. Эксперименты со слоями показали эффективность ```GlobalAveragePooling2D``` ([Train 5](https://github.com/NikitaShulgan/Laba2#train-5)) и ```GlobalMaxPool2D``` ([Train 6](https://github.com/NikitaShulgan/Laba2#train-6)). Точность около 15%. 
Добавление слоев ``` Dense``` ( [Train 7](https://github.com/NikitaShulgan/Laba2#train-7), [Train 8](https://github.com/NikitaShulgan/Laba2#train-8) ) не показали свою эффективность. В итоге за основу был взят [Train 5](https://github.com/NikitaShulgan/Laba2#train-5) и уменьшин [Learning rate](https://en.wikipedia.org/wiki/Learning_rate), на основании [статьи](https://towardsdatascience.com/a-comprehensive-hands-on-guide-to-transfer-learning-with-real-world-applications-in-deep-learning-212bf3b2f27a). [Результаты](https://github.com/NikitaShulgan/Laba2/blob/main/README.md#train-16) показали около 87%. Проблема предыдущих попыток была в том, что нейросеть [не успевала обучаться](https://miro.medium.com/max/2400/1*EP8stDFdu_OxZFGimCZRtQ.jpeg).
#### Links
##### https://sci-hub.se/10.1007/s13748-019-00203-0
##### https://towardsdatascience.com/a-comprehensive-hands-on-guide-to-transfer-learning-with-real-world-applications-in-deep-learning-212bf3b2f27a
##### https://habr.com/ru/post/469931/

