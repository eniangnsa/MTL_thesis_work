import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from sklearn.metrics import accuracy_score, roc_auc_score

def ablation_study(X_train, y_train, X_val, y_val, tasks_to_include, input_shape=(224, 224, 3), num_anomaly_types=14):
    """
    Train model with a subset of tasks for ablation study.
    """
    inputs = tf.keras.Input(shape=input_shape)
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False
    
    x = base_model(inputs)
    pooled = layers.GlobalAveragePooling2D()(x)
    shared_dense = layers.Dense(512, activation='relu')(pooled)
    
    outputs = []
    loss_dict = {}
    metrics_dict = {}
    
    for task in tasks_to_include:
        if task in ['general_anomaly', 'violence', 'property_crime']:
            output = layers.Dense(1, activation='sigmoid', name=task)(shared_dense)
            loss_dict[task] = 'binary_crossentropy'
            metrics_dict[task] = ['accuracy']
        elif task == 'anomaly_type':
            output = layers.Dense(num_anomaly_types, activation='softmax', name=task)(shared_dense)
            loss_dict[task] = 'sparse_categorical_crossentropy'
            metrics_dict[task] = ['accuracy']
        outputs.append(output)
    
    model = models.Model(inputs, outputs)
    model.compile(
        optimizer='adam',
        loss=loss_dict,
        metrics=metrics_dict
    )
    
    history = model.fit(
        X_train,
        [y_train[task] for task in tasks_to_include],
        validation_data=(X_val, [y_val[task] for task in tasks_to_include]),
        epochs=5,
        batch_size=32,
        verbose=1
    )
    
    val_preds = model.predict(X_val)
    general_idx = tasks_to_include.index('general_anomaly') if 'general_anomaly' in tasks_to_include else None
    if general_idx is not None:
        auc = roc_auc_score(y_val['general_anomaly'], val_preds[general_idx])
        return auc
    return None