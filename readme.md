# MLNext

*MLNext* is an open source framework for hardware independent execution of
machine learning using *Python* and *Docker*.
It provides machine learning utilities for Tensorflow and Keras.

## Installation

Install this package using `pip`:

```bash
pip install mlnext --index-url https://pypi:ZS2HLWUqbgmjfURn6U_7@gitlab.phoenixcontact.com/api/v4/projects/771/packages/pypi/simple --trusted-host gitlab.phoenixcontact.com
```

Alternatively, build the `Docker` image:

```bash
docker build -t mlnext:latest .
```

TODO: build from image

## Example

This example works as is.

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
mlnext.plot_signals(x_pred=X_train, y=y_train)
mlnext.plot_signals(x_pred=X_test, y=y_test)

# Compile and train
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics='categorical_accuracy')
history = model.fit(X_train, y_train, epochs=10)

# Visualize training
mlnext.setup_plot()
mlnext.plot_history(history.history)

# Predict labels
y_train_pred = mlnext.eval_softmax(model.predict(X_train))
y_test_pred = mlnext.eval_softmax(model.predict(X_test))


# Evaluate model
print(mlnext.eval_metrics(y_train, y_train_pred))
print(mlnext.eval_metrics(y_test, y_test_pred))
print(mlnext.eval_metrics_all([y_train, y_test], [y_train_pred, y_test_pred]))
```
