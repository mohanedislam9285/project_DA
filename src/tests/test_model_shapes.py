import pytest
from pm25_project.models.cnn_lstm import build_cnn_lstm

def test_assigned_cnn_lstm_input_and_output_shape():
    try: import tensorflow  # noqa
    except ImportError:
        with pytest.raises(RuntimeError,match="optional TensorFlow"): build_cnn_lstm(14,4)
    else:
        model=build_cnn_lstm(14,4); assert model.input_shape==(None,14,4); assert model.output_shape==(None,1)

def test_cnn_lstm_can_preserve_all_time_steps_without_pooling():
    import tensorflow as tf

    for pool_size in (1,None):
        model=build_cnn_lstm(7,3,{"pool_size":pool_size})
        assert not any(isinstance(layer,tf.keras.layers.MaxPooling1D) for layer in model.layers)
