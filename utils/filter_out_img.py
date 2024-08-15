from pathlib import Path
import shutil

def main():

    src_dir = Path('/fs/scratch/PAS2684/yoohj0416/data-archive/2018-NEON-beetles/group_images')
    dst_dir = Path('/fs/scratch/PAS2684/yoohj0416/data-archive/2018-NEON-beetles/group_images_filtered')
    dst_dir.mkdir(exist_ok=True)

    filter_list_path = Path('/fs/scratch/PAS2684/yoohj0416/data-archive/2018-NEON-beetles/predictbeetle/filter_images.txt')

    with open(filter_list_path, 'r') as f:
        filter_list = f.read().splitlines()

    # Replace _mask in the filter list with ''
    filter_list = [name.replace('_mask', '') for name in filter_list]

    for img_path in src_dir.rglob('*.jpg'):
        if img_path.stem in filter_list:
            continue
        shutil.copy(img_path, dst_dir.joinpath(img_path.name))


if __name__ == '__main__':
    main()