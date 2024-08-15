import cv2
from pathlib import Path

def main():
    mask_image_dir = '/fs/scratch/PAS2684/yoohj0416/data-archive/2018-NEON-beetles/group_images_masks'

    # Get the first image in the directory
    # mask_img_path = Path(mask_image_dir).rglob('*.png').__next__()
    # mask_img_path = Path('/fs/scratch/PAS2684/yoohj0416/data-archive/2018-NEON-beetles/group_images_masks/A00000033572_mask.png')
    # mask_img_path = Path('/fs/scratch/PAS2684/yoohj0416/data-archive/2018-NEON-beetles/group_images_masks/A00000033623_mask.png')
    mask_img_path = Path('/fs/scratch/PAS2684/yoohj0416/data-archive/2018-NEON-beetles/group_images_masks/A00000008916_mask.png')
    # Read the image as binary
    mask_img = cv2.imread(str(mask_img_path), cv2.IMREAD_GRAYSCALE)

    # Covert binary image to 0-255 scale
    mask_img = mask_img * 255

    # Image path to save
    save_dir = Path('/users/PAS2119/yoohj0416/predictbeetle/samples')
    save_path = save_dir.joinpath(mask_img_path.name)

    # Save image
    cv2.imwrite(str(save_path), mask_img)

if __name__ == '__main__':
    main()