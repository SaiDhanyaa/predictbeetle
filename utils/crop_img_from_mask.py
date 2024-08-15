from pathlib import Path
import cv2
from tqdm import tqdm
import pandas as pd
import ast
import csv


def lines_intersect(x1, y1, x2, y2, x3, y3, x4, y4):
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    A = (x1, y1)
    B = (x2, y2)
    C = (x3, y3)
    D = (x4, y4)
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

def main():

    original_image_dir = Path('/fs/scratch/PAS2684/yoohj0416/data-archive/2018-NEON-beetles/predictbeetle/group_images_filtered_train')
    mask_image_dir = Path('/fs/scratch/PAS2684/yoohj0416/data-archive/2018-NEON-beetles/group_images_masks')
    save_dir = Path('/fs/scratch/PAS2684/yoohj0416/data-archive/2018-NEON-beetles/predictbeetle/individual_images_train')
    save_dir.mkdir(exist_ok=True)

    # Path for measurement file
    measurement_file_path = Path('/fs/scratch/PAS2684/yoohj0416/data-archive/2018-NEON-beetles/BeetleMeasurements.csv')
    save_csv_path = Path('/fs/scratch/PAS2684/yoohj0416/data-archive/2018-NEON-beetles/predictbeetle/individual_images_train.csv')

    if save_csv_path.exists():
        raise ValueError(f'{save_csv_path} is already exists')
    with open(save_csv_path, 'w') as f:
        save_csv = csv.writer(f)
        save_csv.writerow(['BeetleID', 'dim_ori_img', 'coords_beetle_box', 'coords_len', 'coords_width'])

    # Read the measurement file
    df = pd.read_csv(measurement_file_path)

    area_threshold = 3000

    contour_cnt = 0
    ignore_cnt = 0
    # Iterate through all mask images
    for img_path in tqdm(original_image_dir.rglob('*.jpg'), total=len(list(original_image_dir.rglob('*.jpg')))):

        # Path for mask image
        mask_img_path = mask_image_dir.joinpath(img_path.name.replace('.jpg', '_mask.png'))
        if not mask_img_path.exists():
            raise FileNotFoundError(f'{mask_img_path} does not exist')

        # Read the image as binary
        mask_img = cv2.imread(str(mask_img_path), cv2.IMREAD_GRAYSCALE)

        # Find multiple contours in the image
        contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter out small contours (under threshold)
        contours = [contour for contour in contours if cv2.contourArea(contour) > area_threshold]

        # Get original image
        ori_img = cv2.imread(str(img_path))

        # Get the measurement for the image
        img_id = img_path.name
        img_measurements = df[df['pictureID'] == img_id]

        # Make dictionary for each contour with ID
        contour_dict = {}
        for i, contour in enumerate(contours):
            contour_dict[i] = []

        # Get column 'structure' and 'coords_pix_scaled_up' from the measurement
        img_measurements = img_measurements[['structure', 'coords_pix_scaled_up']]
        coords_len = (0, 0, 0, 0) # (x1, y1, x2, y2)
        coords_width = (0, 0, 0, 0) # (x1, y1, x2, y2)
        for i, row in img_measurements.iterrows():
            structure = row['structure']
            if structure == 'ElytraLength':
                coords_len = ast.literal_eval(row['coords_pix_scaled_up'])
            else:
                coords_width = ast.literal_eval(row['coords_pix_scaled_up'])

                # Check intersection between line of coords_len and line of coords_width
                x1, y1, x2, y2 = coords_len['x1'], coords_len['y1'], coords_len['x2'], coords_len['y2']
                x3, y3, x4, y4 = coords_width['x1'], coords_width['y1'], coords_width['x2'], coords_width['y2']

                x_len = x1 + (x2 - x1) / 2
                y_len = y1 + (y2 - y1) / 2
                x_width = x3 + (x4 - x3) / 2
                y_width = y3 + (y4 - y3) / 2

                # Check if (xn, yn) is in the bounding box of the contour
                is_in_bounding_box = False
                for i, contour in enumerate(contours):
                    x, y, w, h = cv2.boundingRect(contour)
                    # if x <= x1 <= x+w and y <= y1 <= y+h and x <= x2 <= x+w and y <= y2 <= y+h and x <= x3 <= x+w and y <= y3 <= y+h and x <= x4 <= x+w and y <= y4 <= y+h:
                    if x <= x_len <= x+w and y <= y_len <= y+h and x <= x_width <= x+w and y <= y_width <= y+h:
                        is_in_bounding_box = True
                        contour_dict[i].append((x1, y1, x2, y2, x3, y3, x4, y4))
                        break
                if not is_in_bounding_box:
                    continue
                
        
        # If length of contour_dict is 0, raise error
        for i, contour in enumerate(contours):
            if len(contour_dict[i]) == 1:
                contour_cnt += 1
                x1, y1, x2, y2, x3, y3, x4, y4 = contour_dict[i][0]
            elif len(contour_dict[i]) > 1:
                contour_cnt += 1
                # Find the highest in y1
                max_y1 = 0
                max_idx = 0
                for j, coord in enumerate(contour_dict[i]):
                    if coord[1] > max_y1:
                        max_y1 = coord[1]
                        max_idx = j
                x1, y1, x2, y2, x3, y3, x4, y4 = contour_dict[i][max_idx]
            else: # len(contour_dict[i]) == 0
                ignore_cnt += 1
                continue
            
            # Get box coordinates
            x, y, w, h = cv2.boundingRect(contour)
            x1_box, y1_box, x2_box, y2_box = x, y, x+w, y+h

            beetle_id = f'{img_path.stem}_{i:03d}.png'
            # Write to csv file
            with open(save_csv_path, 'a') as f:
                save_csv = csv.writer(f)
                save_csv.writerow([beetle_id, ori_img.shape, (x1_box, y1_box, x2_box, y2_box), (x1, y1, x2, y2,), (x3, y3, x4, y4,)])

            # Crop image using contour
            cropped_img = ori_img[y:y+h, x:x+w]

            # Path to save image
            save_path = save_dir.joinpath(beetle_id)

            # Save image
            cv2.imwrite(str(save_path), cropped_img)
        

if __name__ == '__main__':
    main()