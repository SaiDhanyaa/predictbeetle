from pathlib import Path
import cv2


def main():

    original_image_dir = Path('/fs/scratch/PAS2684/yoohj0416/data-archive/2018-NEON-beetles/group_images')
    mask_image_dir = Path('/fs/scratch/PAS2684/yoohj0416/data-archive/2018-NEON-beetles/group_images_masks')
    save_dir = Path('/users/PAS2119/yoohj0416/predictbeetle/samples')

    # Get the first image in the directory
    # mask_img_path = Path(mask_image_dir).rglob('*.png').__next__()
    # mask_img_path = Path('/fs/scratch/PAS2684/yoohj0416/data-archive/2018-NEON-beetles/group_images_masks/A00000040730_mask.png')
    # mask_img_path = Path('/fs/scratch/PAS2684/yoohj0416/data-archive/2018-NEON-beetles/group_images_masks/A00000040747_mask.png')
    # mask_img_path = Path('/fs/scratch/PAS2684/yoohj0416/data-archive/2018-NEON-beetles/group_images_masks/A00000051555_1_mask.png')
    # mask_img_path = Path('/fs/scratch/PAS2684/yoohj0416/data-archive/2018-NEON-beetles/group_images_masks/A00000003356_mask.png')
    mask_img_path = Path('/fs/scratch/PAS2684/yoohj0416/data-archive/2018-NEON-beetles/group_images_masks/A00000012429_mask.png')

    # Read the image as binary
    mask_img = cv2.imread(str(mask_img_path), cv2.IMREAD_GRAYSCALE)

    # Find multiple contours in the image
    contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out small contours (under 100 pixels)
    contours = [contour for contour in contours if cv2.contourArea(contour) > 3000]

    # Copy image for draw box
    # drawed_img = mask_img.copy()
    # drawed_img = cv2.cvtColor(drawed_img, cv2.COLOR_GRAY2BGR)

    # Load original image
    drawed_img = cv2.imread(str(original_image_dir.joinpath(mask_img_path.name.replace('_mask', '').replace('.png', '.jpg'))))

    # Crop image using all contours
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        cropped_img = mask_img[y:y+h, x:x+w]

        # Before, draw a rectangle for bounding box with red color
        cv2.rectangle(drawed_img, (x, y), (x+w, y+h), (0, 0, 255), 2)
    
    # Path to save image
    save_path = save_dir.joinpath(mask_img_path.name)

    # Save image
    cv2.imwrite(str(save_path), drawed_img)

if __name__ == '__main__':
    main()