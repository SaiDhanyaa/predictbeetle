from pathlib import Path
import pandas as pd
import csv
import ast

import cv2


def transform_coords_to_local(global_coords, beetle_box_coords):
    x1, y1, x2, y2 = beetle_box_coords
    x, y = global_coords
    return x - x1, y - y1

def main():

    image_base_dir = Path('/fs/scratch/PAS2684/yoohj0416/data-archive/2018-NEON-beetles/predictbeetle/individual_images')
    gt_file_path = Path('/fs/scratch/PAS2684/yoohj0416/data-archive/2018-NEON-beetles/predictbeetle/individual_images_train.csv')

    save_dir = Path('/users/PAS2119/yoohj0416/predictbeetle/results/gt')
    save_dir.mkdir(exist_ok=True)

    df = pd.read_csv(gt_file_path)

    for idx, row in df.iterrows():
        image_path = image_base_dir.joinpath(row["BeetleID"])
        if not image_path.exists():
            raise FileNotFoundError(f'{image_path} does not exist')
        
        img = cv2.imread(str(image_path))

        # dim_ori_img, coords_beetle_box, coords_len, coords_width
        # (h, w, c), (x1, y1, x2, y2), (x1, y1, x2, y2), (x1, y1, x2, y2)

        dim_ori_img = ast.literal_eval(row["dim_ori_img"])
        coords_beetle_box = ast.literal_eval(row["coords_beetle_box"])
        coords_len = ast.literal_eval(row["coords_len"])
        coords_width = ast.literal_eval(row["coords_width"])

        # Compute the local coordinates of coords_len, coords_width
        coords_len_local = (transform_coords_to_local(coords_len[:2], coords_beetle_box), transform_coords_to_local(coords_len[2:], coords_beetle_box))
        coords_width_local = (transform_coords_to_local(coords_width[:2], coords_beetle_box), transform_coords_to_local(coords_width[2:], coords_beetle_box))

        # Draw the lines on the image with green color
        img = cv2.line(img, coords_len_local[0], coords_len_local[1], (0, 255, 0), 2)
        img = cv2.line(img, coords_width_local[0], coords_width_local[1], (0, 255, 0), 2)

        # And draw small circles on the end points of the lines
        img = cv2.circle(img, coords_len_local[0], 5, (0, 255, 0), -1)
        img = cv2.circle(img, coords_len_local[1], 5, (0, 255, 0), -1)
        img = cv2.circle(img, coords_width_local[0], 5, (0, 255, 0), -1)
        img = cv2.circle(img, coords_width_local[1], 5, (0, 255, 0), -1)

        # Save the image
        save_path = save_dir.joinpath(image_path.name)
        cv2.imwrite(str(save_path), img)
        

if __name__ == '__main__':
    main()