import tensorflow as tf
from scipy.stats import pearsonr

def compute_gradient_alignment(model, data, labels, task_names):
    """
    Compute cosine similarity between gradients of tasks.
    """
    gradients = {}
    for task in task_names:
        with tf.GradientTape() as tape:
            predictions = model(data)
            task_idx = task_names.index(task)
            loss = tf.keras.losses.binary_crossentropy(labels[task], predictions[task_idx])
            if task == 'anomaly_type':
                loss = tf.keras.losses.sparse_categorical_crossentropy(labels[task], predictions[task_idx])
        grads = tape.gradient(loss, model.trainable_variables)
        gradients[task] = grads
    
    similarities = {}
    for task1 in task_names:
        for task2 in task_names:
            if task1 >= task2:
                continue
            g1 = tf.concat([tf.reshape(g, [-1]) for g in gradients[task1] if g is not None], axis=0)
            g2 = tf.concat([tf.reshape(g, [-1]) for g in gradients[task2] if g is not None], axis=0)
            cos_sim = tf.reduce_sum(g1 * g2) / (tf.norm(g1) * tf.norm(g2))
            similarities[f'{task1}_vs_{task2}'] = cos_sim.numpy()
    
    return similarities


def compute_loss_correlation(losses_dict, task_names):
    """
    Compute Pearson correlation between task losses.
    
    Args:
        losses_dict (dict): Dictionary with task names as keys and lists of loss values as values.
        task_names (list): List of task names.
    
    Returns:
        dict: Dictionary with task pair names as keys and Pearson correlation coefficients as values.
    """
    correlations = {}
    for task1 in task_names:
        for task2 in task_names:
            if task1 >= task2:
                continue
            corr, _ = pearsonr(losses_dict[task1], losses_dict[task2])
            correlations[f'{task1}_vs_{task2}'] = corr
    return correlations