
from train_model import train_mtl_model
from build_model import build_mtl_model
from load_data import load_ucf_crime_data
from ablation_study import ablation_study

def main():
    # Paths to dataset
    data_dir = '/home/user/ucf_crime_dataset'  # Update to your dataset path
    train_annotation_file = 'train_annotations.txt'
    test_annotation_file = 'test_annotations.txt'
    
    # Load data
    X_train, y_train = load_ucf_crime_data(data_dir, train_annotation_file)
    X_val, y_val = load_ucf_crime_data(data_dir, test_annotation_file)
    
    # Build and train MTL model
    model = build_mtl_model()
    history, grad_similarities, loss_correlations = train_mtl_model(model, X_train, y_train, X_val, y_val)
    
    # Print task relationship results
    print("Gradient Alignment (Cosine Similarity):")
    for pair, sim in grad_similarities.items():
        print(f"{pair}: {sim:.4f}")
    
    print("\nLoss Correlation (Pearson):")
    for pair, corr in loss_correlations.items():
        print(f"{pair}: {corr:.4f}")
    
    # Ablation study
    task_combinations = [
        ['general_anomaly'],
        ['general_anomaly', 'violence', 'property_crime'],
        ['general_anomaly', 'anomaly_type'],
        ['general_anomaly', 'violence', 'property_crime', 'anomaly_type']
    ]
    
    print("\nAblation Study Results (AUC for General Anomaly Detection):")
    for tasks in task_combinations:
        auc = ablation_study(X_train, y_train, X_val, y_val, tasks)
        if auc is not None:
            print(f"Tasks: {tasks}, AUC: {auc:.4f}")

if __name__ == "__main__":
    main()