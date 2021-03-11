# Лабораторная работа #2
## Решение задачи классификации изображений из набора данных Oregon Wildlife с использованием нейронных сетей глубокого обучения и техники обучения Transfer Learning

#### EfficientNet-B0 architecture
![image](https://user-images.githubusercontent.com/80168174/110480321-6aae0900-80f7-11eb-82e6-f389f93c3966.png)


### Train 1
#### Нейронная сеть EfficientNet-B0 (случайное начальное приближение), датасет Oregon WildLife.
##### https://tensorboard.dev/experiment/4EoeVqP1TLq6X8EG6GhRgw/#scalars
```
BATCH_SIZE = 16


def build_model():
  inputs = tf.keras.Input(shape=(RESIZE_TO, RESIZE_TO, 3))
  outputs = EfficientNetB0(weights=None, classes=NUM_CLASSES)(inputs)
  return tf.keras.Model(inputs=inputs, outputs=outputs)
```

### Train 2
#### Нейронная сеть EfficientNet-B0 (продобученная на ImageNet), датасет Oregon WildLife.
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

#### Links
https://sci-hub.se/10.1007/s13748-019-00203-0
