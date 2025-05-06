from compute_relatedness import compute_gradient_alignment, compute_loss_correlation


def train_mtl_model(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=32):
    """
    Train MTL model and collect losses for correlation analysis.
    """
    task_names = ['general_anomaly', 'violence', 'property_crime', 'anomaly_type']
    losses_dict = {task: [] for task in task_names}
    
    model.compile(
        optimizer='adam',
        loss={
            'general_anomaly': 'binary_crossentropy',
            'violence': 'binary_crossentropy',
            'property_crime': 'binary_crossentropy',
            'anomaly_type': 'sparse_categorical_crossentropy'
        },
        loss_weights={
            'general_anomaly': 1.0,
            'violence': 0.5,
            'property_crime': 0.5,
            'anomaly_type': 1.0
        },
        metrics={
            'general_anomaly': ['accuracy', tf.keras.metrics.AUC(name='auc')],
            'violence': ['accuracy'],
            'property_crime': ['accuracy'],
            'anomaly_type': ['accuracy']
        }
    )
    
    history = model.fit(
        X_train,
        [y_train[task] for task in task_names],
        validation_data=(X_val, [y_val[task] for task in task_names]),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    
    for task in task_names:
        losses_dict[task] = history.history[f'{task}_loss']
    
    grad_similarities = compute_gradient_alignment(model, X_val[:batch_size], y_val, task_names)
    loss_correlations = compute_loss_correlation(losses_dict, task_names)
    
    return history, grad_similarities, loss_correlations