import os
import time
import glob
import random
import argparse
from datetime import datetime
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--images', dest='images', required=True, help="Glob of images to combine")
args = parser.parse_args()

def combine_images(output_path):
    image_paths = glob.glob(args.images)
    chosen_images = random.choices(image_paths, k=2)
    images = [Image.open(x) for x in chosen_images]
    widths, heights = zip(*(i.size for i in images))
    max_width = max(widths)
    total_height = sum(heights)
    new_im = Image.new('RGB', (max_width, total_height), 'WHITE')
    y_offset = 0
    for im in images:
        new_im.paste(im, (0, y_offset))
        y_offset += im.size[1] + 50
    new_im.save(output_path)
    return chosen_images

if __name__ == '__main__':
    temp_image = "/tmp/combined_image_temp.png"  # save to tmpfs dir
    final_image = "/tmp/combined_image.png"
    chosen_images = combine_images(temp_image)
    print(datetime.now(), "- Combining images:", ' & '.join(chosen_images))
    os.rename(temp_image, final_image)


