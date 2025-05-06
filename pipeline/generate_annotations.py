import os
import glob

def generate_annotation_file(dataset_root, split, output_file):
    """
    Generate an annotation file for a given dataset split (train or test) with .png images.
    
    Args:
        dataset_root (str): Path to the dataset root directory (e.g., '/home/user/ucf_crime_dataset').
        split (str): Dataset split ('train' or 'test').
        output_file (str): Path to save the annotation file (e.g., 'train_annotations.txt').
    """
    split_dir = os.path.join(dataset_root, split)
    if not os.path.exists(split_dir):
        raise ValueError(f"Directory {split_dir} does not exist.")
    
    annotations = []
    for class_name in os.listdir(split_dir):
        class_dir = os.path.join(split_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        
        # Find all .png images in the class subfolder
        image_files = glob.glob(os.path.join(class_dir, "*.png"))
        for image_path in image_files:
            relative_path = os.path.relpath(image_path, dataset_root)
            annotation = f"{relative_path} {class_name}"
            annotations.append(annotation)
    
    with open(output_file, 'w') as f:
        for annotation in annotations:
            f.write(annotation + '\n')
    
    print(f"Generated annotation file: {output_file} with {len(annotations)} entries.")

def main():
    dataset_root = '/home/user/ucf_crime_dataset'  # Update to your dataset path
    train_output = 'train_annotations.txt'
    test_output = 'test_annotations.txt'
    
    generate_annotation_file(dataset_root, 'train', train_output)
    generate_annotation_file(dataset_root, 'test', test_output)

if __name__ == "__main__":
    main()