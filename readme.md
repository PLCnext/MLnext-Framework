# MLNext Framework

*MLNext Framework* is an open source framework for hardware independent execution of
machine learning using *Python* and *Docker*.
It provides machine learning utilities for Tensorflow and Keras.
The corresponding *Python* package is called *mlnext*

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
mlnext.setup_plot()
mlnext.plot_signals(x_pred=X_train, y=y_train)
mlnext.plot_signals(x_pred=X_test, y=y_test)

# Compile and train
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics='categorical_accuracy')
history = model.fit(X_train, y_train, epochs=10)

# Visualize training
mlnext.plot_history(history.history)

# Predict labels
y_train_pred = mlnext.eval_softmax(model.predict(X_train))
y_test_pred = mlnext.eval_softmax(model.predict(X_test))


# Evaluate model
print(mlnext.eval_metrics(y_train, y_train_pred))
print(mlnext.eval_metrics(y_test, y_test_pred))
print(mlnext.eval_metrics_all([y_train, y_test], [y_train_pred, y_test_pred]))
```


## Example 2

# VAE_Proba_Evaluation

## Setup


```python
%load_ext autoreload
%autoreload 2
```


```python
# change cwd to root in order to import cauldron
import os
from pyprojroot import here
os.chdir(here('.'))
```


```python
# Imports

import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt

import mlnext.data as data
import mlnext.plot as plot
import mlnext.score as score
import mlnext.io as io
```


```python
def norm_log_likelihood(x: tf.Tensor,
                        mean: tf.Tensor,
                        log_var: tf.Tensor) -> tf.Tensor:
    """Calculates the log likelihood that `x` was drawn from
        a normal gaussian distribution defined by `mean` and `log_var`.

    ..math::

        Gaussian normal distribution with mean \\mu and variance \\sigma:
        f(x) = \\frac{1}{\\sqrt{2\\pi\\sigma^2}}\\exp{-\\frac{1}{2}
        (\\frac{x-\\mu}{\\sigma})^2}

        Log likelihood:
        log(p(x | \\mu, \\sigma)) = -0.5 (\\log(2\\pi) + (x-mu)^2/\\sigma^2 +
        \\log(\\sigma^2))

    Args:
        x (tf.Tensor): Sample.
        mean (tf.Tensor): Mean of the gaussian normal distribution.
        log_var (tf.Tensor): Log variance of the gaussian normal distribution.

    Returns:
        tf.Tensor: Returns the log likelihood.
    """
    x = tf.cast(x, tf.float32)
    mean = tf.cast(mean, tf.float32)
    log_var = tf.cast(log_var, tf.float32)

    c = tf.math.log(2. * np.pi) * tf.ones(tf.shape(x), tf.float32)
    b = log_var
    a = tf.square(x - mean) / (tf.exp(log_var) + 1e-10)

    return 0.5 * (c + b + a)
```


```python
plot.setup_plot()

devices = tf.config.list_physical_devices('GPU')
if len(devices) > 0:
    tf.config.experimental.set_memory_growth(devices[0], enable=True)
```

## Load Model


```python
date = '2021_07_05__15_28_24'
folder = here(f'./examples/logs/{date}')
name = 'vae'

model = keras.models.load_model(f'{folder}/{name}')
```


```python
model.summary()
```

    Model: "vae"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    encoder (Functional)         [(None, 20, 6), (None, 20 4780
    _________________________________________________________________
    decoder (Functional)         [(None, 20, 17), (None, 2 4228
    =================================================================
    Total params: 9,010
    Trainable params: 9,008
    Non-trainable params: 2
    _________________________________________________________________


## Load Data


```python
# get window size from model
timesteps = model.layers[0].input_shape[1]
print(f'timesteps: {timesteps}')

X_train = data.load_data_3d(path=here('./examples/data/01_UPS_Train_ZV.csv'),
                            timesteps=timesteps)
y_train = data.load_data(
    path=here('./examples/data/01_UPS_Train_Label.csv'))

X_val = data.load_data_3d(path=here('./examples/data/01_UPS_Validation_ZV.csv'),
                          timesteps=timesteps)
y_val = data.load_data(
    path=here('./examples/data/01_UPS_Validation_Label.csv'))

X_test = data.load_data_3d(path=here('./examples/data/01_UPS_Test_ZV.csv'),
                           timesteps=timesteps)
y_test = data.load_data(
    path=here('./examples/data/01_UPS_Test_Label.csv'))
```

    timesteps: 20
    Loaded 01_UPS_Train_ZV.csv with 10492 rows and 17 columns.
    Dropped 12 rows. New shape: (524, 20, 17).
    Loaded 01_UPS_Train_Label.csv with 10492 rows and 1 columns.
    Loaded 01_UPS_Validation_ZV.csv with 1166 rows and 17 columns.
    Dropped 6 rows. New shape: (58, 20, 17).
    Loaded 01_UPS_Validation_Label.csv with 1166 rows and 1 columns.
    Loaded 01_UPS_Test_ZV.csv with 500 rows and 17 columns.
    Dropped 0 rows. New shape: (25, 20, 17).
    Loaded 01_UPS_Test_Label.csv with 500 rows and 1 columns.


# Training History


```python
history = io.load_json(f'{folder}/history.json')
plot.plot_history(history, filter=['likelihood', 'kld'])
```



![png](img/output_13_0.png)




```python
keras.utils.plot_model(model.encoder, show_shapes=True, to_file="encoder.png")
```





![png](img/output_14_0.png)





```python
keras.utils.plot_model(model.decoder, show_shapes=True, to_file="decoder.png")
```





![png](img/output_15_0.png)




# Validation

## Signal Reconstruction


```python
val_pred = model.predict({'continuous': X_val})
```


```python
plot.plot_signals_norm(x_pred=val_pred.norm_mean, y=y_val, norm_mean=val_pred.norm_mean,
                  norm_std=np.exp(0.5*val_pred.norm_log_var), x=X_val)
```



![png](img/output_19_0.png)



## Reconstruction Loss


```python
recon_error_val = score.l2_norm(X_val, val_pred.norm_mean)
threshold = score.get_threshold(recon_error_val, p=99.9)
plot.plot_error(recon_error_val, y=y_val.to_numpy(), threshold=threshold)
```



![png](img/output_21_0.png)




```python
y_val_pred = score.apply_threshold(x=recon_error_val, threshold=threshold)
score.eval_metrics(y_val, y_val_pred)
```

    E:\anaconda\envs\py38gpu\lib\site-packages\sklearn\metrics\_classification.py:1245: UndefinedMetricWarning: Recall is ill-defined and being set to 0.0 due to no true samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))





    {'accuracy': 0.9982758620689656, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}




```python
X_val_ll = norm_log_likelihood(
    X_val, val_pred.norm_mean, val_pred.norm_log_var)
X_val_ll = np.mean(X_val_ll, axis=-1).reshape(-1, 1)
threshold_ll = score.get_threshold(X_val_ll, 99.9)
plt.yscale('symlog')
plot.plot_error(X_val_ll, y=y_val.to_numpy(), threshold=threshold_ll)
```



![png](img/output_23_0.png)




```python
y_val_ll_pred = score.apply_threshold(X_val_ll, threshold=threshold_ll)
score.eval_metrics(y_val, y_val_ll_pred)
```

    E:\anaconda\envs\py38gpu\lib\site-packages\sklearn\metrics\_classification.py:1245: UndefinedMetricWarning: Recall is ill-defined and being set to 0.0 due to no true samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))





    {'accuracy': 0.9982758620689656, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}



# Training

## Signal Reconstruction


```python
train_pred = model.predict({'continuous': X_train})
```


```python
plot.plot_signals_norm(x_pred=train_pred.norm_mean, y=y_train, norm_mean=train_pred.norm_mean,
                  norm_std=np.exp(0.5*train_pred.norm_log_var), x=X_train)
```



![png](img/output_28_0.png)



## Reconstruction Loss


```python
recon_error_train = score.l2_norm(X_train, train_pred.norm_mean)
plot.plot_error(recon_error_train, y=y_train.to_numpy(),threshold=threshold)
```



![png](img/output_30_0.png)




```python
y_train_pred = score.apply_threshold(recon_error_train, threshold=threshold)
score.eval_metrics(y_train, y_train_pred)
```

    E:\anaconda\envs\py38gpu\lib\site-packages\sklearn\metrics\_classification.py:1245: UndefinedMetricWarning: Recall is ill-defined and being set to 0.0 due to no true samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))





    {'accuracy': 0.9991412213740458, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}




```python
X_train_ll = norm_log_likelihood(
    X_train, train_pred.norm_mean, train_pred.norm_log_var)
X_train_ll = np.mean(X_train_ll, axis=-1).reshape(-1, 1)
plt.yscale('symlog')
plot.plot_error(X_train_ll, y=y_train.to_numpy(), threshold=threshold_ll)
```



![png](img/output_32_0.png)




```python
y_train_ll_pred = score.apply_threshold(x=X_train_ll, threshold=threshold_ll)
score.eval_metrics(y_train, y_train_ll_pred)
```

    E:\anaconda\envs\py38gpu\lib\site-packages\sklearn\metrics\_classification.py:1245: UndefinedMetricWarning: Recall is ill-defined and being set to 0.0 due to no true samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))





    {'accuracy': 0.9995229007633588, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}



# Test

## Signal Reconstruction


```python
test_pred = model.predict({'continuous': X_test})
```


```python
plot.plot_signals_norm(x_pred=test_pred.norm_mean, y=y_test, norm_mean=test_pred.norm_mean,
                  norm_std=np.exp(0.5*test_pred.norm_log_var), x=X_test)
```



![png](img/output_37_0.png)



## Reconstruction Loss


```python
recon_error_test = score.l2_norm(X_test, test_pred.norm_mean)
plot.plot_error(recon_error_test, y=y_test.to_numpy(), threshold=threshold)
```



![png](img/output_39_0.png)




```python
y_test_pred = score.apply_threshold(recon_error_test, threshold=threshold)
score.eval_metrics(y_test, y_test_pred)
```




    {'accuracy': 0.694,
     'precision': 0.9652777777777778,
     'recall': 0.4843205574912892,
     'f1': 0.6450116009280742,
     'AUC': 0.7304231895437667}




```python
X_test_ll = norm_log_likelihood(
    X_test, test_pred.norm_mean, test_pred.norm_log_var)
X_test_ll = np.mean(X_test_ll, axis=-1).reshape(-1, 1)
plt.yscale('symlog')
plot.plot_error(X_test_ll, y=y_test.to_numpy(), threshold=threshold_ll)
```



![png](img/output_41_0.png)




```python
y_test_ll_pred = score.apply_threshold(X_test_ll, threshold=threshold_ll)
score.eval_metrics(y_test, y_test_ll_pred)
```




    {'accuracy': 0.912,
     'precision': 1.0,
     'recall': 0.8466898954703833,
     'f1': 0.9169811320754718,
     'AUC': 0.9233449477351916}



## Results


```python
score.eval_metrics_all([y_test, y_train, y_val], [
                       y_test_pred, y_train_pred, y_val_pred])
```




    {'accuracy': 0.986490939044481,
     'precision': 0.896774193548387,
     'recall': 0.4843205574912892,
     'f1': 0.6289592760180995,
     'AUC': 0.7414853441299355}




```python
score.eval_metrics_all([y_test, y_train, y_val], [
                       y_test_ll_pred, y_train_ll_pred, y_val_ll_pred])
```




    {'accuracy': 0.9957990115321252,
     'precision': 0.972,
     'recall': 0.8466898954703833,
     'f1': 0.9050279329608939,
     'AUC': 0.9230496638408189}
