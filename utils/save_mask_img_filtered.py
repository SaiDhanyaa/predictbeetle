import cv2
from pathlib import Path

def main():
    mask_image_dir = Path('/fs/scratch/PAS2684/yoohj0416/data-archive/2018-NEON-beetles/group_images_masks')
    filter_img_dir = Path('/fs/scratch/PAS2684/yoohj0416/data-archive/2018-NEON-beetles/group_images_filtered')
    save_image_dir = Path('/fs/scratch/PAS2684/yoohj0416/data-archive/2018-NEON-beetles/predictbeetle/filtered_masks_converted')
    
    save_image_dir.mkdir(exist_ok=True)

    filter_list = [img_path.stem for img_path in filter_img_dir.rglob('*.jpg')]

    # Iterate through all mask images
    for mask_img_path in mask_image_dir.rglob('*.png'):

        mask_img_stem = mask_img_path.stem.replace('_mask', '')
        if mask_img_stem not in filter_list:
            continue

        # Read the image as binary
        mask_img = cv2.imread(str(mask_img_path), cv2.IMREAD_GRAYSCALE)

        # Covert binary image to 0-255 scale
        mask_img = mask_img * 255

        # Image path to save
        save_path = save_image_dir.joinpath(mask_img_path.name)

        # Save image
        cv2.imwrite(str(save_path), mask_img)

if __name__ == '__main__':
    main()