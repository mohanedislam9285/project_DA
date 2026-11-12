from __future__ import annotations
import numpy as np

def build_cnn_lstm(lookback:int,n_features:int,config:dict|None=None):
    try: import tensorflow as tf
    except ImportError as exc: raise RuntimeError("CNN-LSTM requires the optional TensorFlow dependency; install the [deep] extra") from exc
    c={"filters":32,"kernel_size":3,"pool_size":2,"dropout":.2,"lstm_units":64,"dense_units":32,"learning_rate":1e-3,"loss":"mse",**(config or {})}
    if lookback<c["kernel_size"]: raise ValueError("lookback must be at least kernel_size")
    pool_size=c["pool_size"]
    if pool_size is not None and (isinstance(pool_size,bool) or not isinstance(pool_size,int) or pool_size<1):
        raise ValueError("pool_size must be a positive integer or None")
    tf.keras.utils.set_random_seed(int(c.get("seed",42)))
    inputs=tf.keras.Input(shape=(lookback,n_features),name="daily_features")
    x=tf.keras.layers.Conv1D(c["filters"],c["kernel_size"],padding="causal",name="causal_conv")(inputs)
    x=tf.keras.layers.LayerNormalization()(x); x=tf.keras.layers.Activation("relu")(x); x=tf.keras.layers.Dropout(c["dropout"])(x)
    if pool_size is not None and pool_size>1 and lookback>=pool_size:
        x=tf.keras.layers.MaxPooling1D(pool_size)(x)
    x=tf.keras.layers.LSTM(c["lstm_units"],name="lstm")(x)
    x=tf.keras.layers.Dense(c["dense_units"],activation="relu")(x); outputs=tf.keras.layers.Dense(1,name="pm25_prediction")(x)
    model=tf.keras.Model(inputs,outputs,name="cnn_lstm_pm25")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=c["learning_rate"]),loss=c["loss"],metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")])
    return model

def build_lstm(lookback:int,n_features:int,config:dict|None=None):
    """Standalone recurrent comparator with the same input/output contract as CNN-LSTM."""
    try: import tensorflow as tf
    except ImportError as exc: raise RuntimeError("LSTM requires the optional TensorFlow dependency; install the [deep] extra") from exc
    c={"lstm_units":64,"dense_units":32,"dropout":.2,"learning_rate":1e-3,"loss":"mse",**(config or {})}
    tf.keras.utils.set_random_seed(int(c.get("seed",42)))
    inputs=tf.keras.Input(shape=(lookback,n_features),name="daily_features")
    x=tf.keras.layers.LSTM(c["lstm_units"],name="lstm")(inputs)
    x=tf.keras.layers.Dropout(c["dropout"])(x)
    x=tf.keras.layers.Dense(c["dense_units"],activation="relu")(x)
    outputs=tf.keras.layers.Dense(1,name="pm25_prediction")(x)
    model=tf.keras.Model(inputs,outputs,name="lstm_pm25")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=c["learning_rate"]),loss=c["loss"],metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")])
    return model

def fit_cnn_lstm(model,x_train,y_train,x_inner,y_inner,config):
    import tensorflow as tf
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=int(config.get("patience",10)),restore_best_weights=True)]
    return model.fit(x_train,y_train,validation_data=(x_inner,y_inner),epochs=int(config.get("epochs",100)),batch_size=int(config.get("batch_size",64)),callbacks=callbacks,verbose=0)
