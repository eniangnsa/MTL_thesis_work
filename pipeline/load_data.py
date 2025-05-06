import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
import numpy as np
import os
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy.stats import pearsonr

def load_ucf_crime_data(data_dir, annotation_file, image_size=(224, 224)):
    """
    Load UCF-Crime dataset images and annotations from the generated annotation file.
    
    Args:
        data_dir: Path to the dataset root directory.
        annotation_file: Path to the annotation file (e.g., train_annotations.txt).
        image_size: Size to resize images.
    
    Returns:
        images: List of images (numpy arrays).
        labels: Dictionary of labels for each task.
    """
    # Read annotation file
    annotations = pd.read_csv(annotation_file, sep=' ', header=None, names=['image', 'label'])
    
    images = []
    labels = {
        'general_anomaly': [],
        'violence': [],
        'property_crime': [],
        'anomaly_type': []
    }
    
    # Define anomaly classes
    anomaly_classes = ['Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary', 'Explosion',
                       'Fighting', 'Robbery', 'Shooting', 'Stealing', 'Shoplifting',
                       'Vandalism', 'RoadAccident']
    violent_classes = ['Assault', 'Fighting', 'Shooting']
    property_classes = ['Burglary', 'Stealing', 'Shoplifting', 'Vandalism']
    
    for _, row in annotations.iterrows():
        image_path = os.path.join(data_dir, row['image'])
        label = row['label']
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            continue
        
        image = cv2.resize(image, image_size)
        image = image / 255.0  # Normalize
        images.append(image)
        
        # Assign labels for each task
        is_anomaly = 1 if label in anomaly_classes else 0
        is_violent = 1 if label in violent_classes else 0
        is_property = 1 if label in property_classes else 0
        anomaly_type = anomaly_classes.index(label) if label in anomaly_classes else len(anomaly_classes)
        
        labels['general_anomaly'].append(is_anomaly)
        labels['violence'].append(is_violent)
        labels['property_crime'].append(is_property)
        labels['anomaly_type'].append(anomaly_type)
    
    return np.array(images), labels