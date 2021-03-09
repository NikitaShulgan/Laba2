# Лабораторная работа #2
## Решение задачи классификации изображений из набора данных Oregon Wildlife с использованием нейронных сетей глубокого обучения и техники обучения Transfer Learning
### Train 1
#### Нейронная сеть EfficientNet-B0 (случайное начальное приближение), датасет Oregon WildLife.
```
BATCH_SIZE = 16


def build_model():
  inputs = tf.keras.Input(shape=(RESIZE_TO, RESIZE_TO, 3))
  outputs = EfficientNetB0(weights=None, classes=NUM_CLASSES)(inputs)
  return tf.keras.Model(inputs=inputs, outputs=outputs)
```


#### Links
