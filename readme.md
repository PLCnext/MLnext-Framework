# MLNext Framework

*MLNext Framework* is an open source framework for hardware independent execution of
machine learning using *Python* and *Docker*.
It provides machine learning utilities for Tensorflow and Keras.
The corresponding *Python* package is called *mlnext*.

## Installation

Install this package using `pip`:

```bash
pip install mlnext --index-url https://pypi:ZS2HLWUqbgmjfURn6U_7@gitlab.phoenixcontact.com/api/v4/projects/771/packages/pypi/simple --trusted-host gitlab.phoenixcontact.com
```

## Modules

The *MLnext Framework* consists of 5 modules:

```python
import mlnext.data as data          # for data loading and manipulation
import mlnext.io as io              # for loading and saving files
import mlnext.pipeline as pipeline  # for data preprocessing
import mlnext.plot as plot          # for data visualization
import mlnext.score as score        # for model evaluation
```

## Example

```python
import tensorflow.keras as keras
import numpy as np
from numpy.random import random

import mlnext

# Create a model
model = keras.Sequential([
    keras.layers.Input(2),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(2, activation='softmax')
])

# Generate data for example
X_train, X_test = random((100, 2)), random((100, 2))
y_train, y_test = random((100, 1)) > 0.5, random((100, 1)) > 0.2

# Plot signals
mlnext.setup_plot()
mlnext.plot_signals(x_pred=X_train, y=y_train)
mlnext.plot_signals(x_pred=X_test, y=y_test)
```

![png](img/output_0_1.png)


![png](img/output_0_3.png)


```python
# Compile and train
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics='categorical_accuracy')
history = model.fit(X_train, y_train, epochs=10)

# Visualize training
mlnext.plot_history(history.history)
```

    Epoch 1/10
    4/4 [==============================] - 0s 4ms/step - loss: 0.6968 - categorical_accuracy: 0.8800
    Epoch 2/10
    4/4 [==============================] - 0s 3ms/step - loss: 0.6909 - categorical_accuracy: 0.0400
    Epoch 3/10
    4/4 [==============================] - 0s 3ms/step - loss: 0.6879 - categorical_accuracy: 0.0000e+00
    Epoch 4/10
    4/4 [==============================] - 0s 3ms/step - loss: 0.6878 - categorical_accuracy: 0.0000e+00
    Epoch 5/10
    4/4 [==============================] - 0s 3ms/step - loss: 0.6862 - categorical_accuracy: 0.0000e+00
    Epoch 6/10
    4/4 [==============================] - 0s 3ms/step - loss: 0.6856 - categorical_accuracy: 0.0000e+00
    Epoch 7/10
    4/4 [==============================] - 0s 3ms/step - loss: 0.6851 - categorical_accuracy: 0.0000e+00
    Epoch 8/10
    4/4 [==============================] - 0s 3ms/step - loss: 0.6848 - categorical_accuracy: 0.0000e+00
    Epoch 9/10
    4/4 [==============================] - 0s 3ms/step - loss: 0.6846 - categorical_accuracy: 0.0000e+00
    Epoch 10/10
    4/4 [==============================] - 0s 3ms/step - loss: 0.6844 - categorical_accuracy: 0.0000e+00




![png](img/output_1_1.png)



```python
# Predict labels
y_train_pred = mlnext.eval_softmax(model.predict(X_train))
y_test_pred = mlnext.eval_softmax(model.predict(X_test))


# Evaluate model
print(mlnext.eval_metrics(y_train, y_train_pred))
print(mlnext.eval_metrics(y_test, y_test_pred))
print(mlnext.eval_metrics_all([y_train, y_test], [y_train_pred, y_test_pred]))
```

    {'accuracy': 0.56, 'precision': 0.56, 'recall': 1.0, 'f1': 0.717948717948718, 'AUC': 0.5}
    {'accuracy': 0.82, 'precision': 0.82, 'recall': 1.0, 'f1': 0.9010989010989011, 'AUC': 0.5}
    {'accuracy': 0.69, 'precision': 0.69, 'recall': 1.0, 'f1': 0.8165680473372781, 'AUC': 0.5}
