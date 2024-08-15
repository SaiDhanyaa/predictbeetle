# Split train and test images using scikit-learn's train_test_split function
from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil

def main():
    # original_image_dir = Path('/fs/scratch/PAS2684/yoohj0416/data-archive/2018-NEON-beetles/group_images')
    original_image_dir = Path('/fs/scratch/PAS2684/yoohj0416/data-archive/2018-NEON-beetles/group_images_filtered')
    save_dir = Path('/fs/scratch/PAS2684/yoohj0416/data-archive/2018-NEON-beetles/predictbeetle')
    save_dir.mkdir(exist_ok=True)

    # Get all image paths
    image_paths = list(original_image_dir.rglob('*.jpg'))

    # Split train and test images
    train_paths, test_paths = train_test_split(image_paths, test_size=0.3, random_state=42)

    # Create train and test directories
    train_dir_name = 'group_images_filtered_train'
    test_dir_name = 'group_images_filtered_test'
    train_dir = save_dir.joinpath(train_dir_name)
    test_dir = save_dir.joinpath(test_dir_name)
    train_dir.mkdir(exist_ok=True)
    test_dir.mkdir(exist_ok=True)

    # Save train and test images
    for path in train_paths:
        shutil.copy(str(path), str(train_dir.joinpath(path.name)))
    for path in test_paths:
        shutil.copy(str(path), str(test_dir.joinpath(path.name)))

                    
if __name__ == '__main__':
    main()