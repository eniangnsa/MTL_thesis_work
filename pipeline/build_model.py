from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
import tensorflow as tf 


def build_mtl_model(input_shape=(224, 224, 3), num_anomaly_types=14):
    """
    Build MTL model with shared ResNet50 backbone and task-specific heads for images.
    """
    inputs = tf.keras.Input(shape=input_shape)
    
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False
    
    x = base_model(inputs)
    pooled = layers.GlobalAveragePooling2D()(x)
    shared_dense = layers.Dense(512, activation='relu')(pooled)
    
    # Task-specific heads
    anomaly_output = layers.Dense(1, activation='sigmoid', name='general_anomaly')(shared_dense)
    violence_output = layers.Dense(1, activation='sigmoid', name='violence')(shared_dense)
    property_output = layers.Dense(1, activation='sigmoid', name='property_crime')(shared_dense)
    type_output = layers.Dense(num_anomaly_types, activation='softmax', name='anomaly_type')(shared_dense)
    
    model = models.Model(inputs, [
        anomaly_output, violence_output, property_output, type_output
    ])
    
    return model