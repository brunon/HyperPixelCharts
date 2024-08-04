import os
import time
import glob
import random
import argparse
from datetime import datetime
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--images', dest='images', required=True, help="Glob of images to combine")
parser.add_argument('--offset', dest='offset', required=False, type=int, default=50, help="Vertical offset (spacing)")
args = parser.parse_args()

def combine_images(output_path, offset_height):
    image_paths = glob.glob(args.images)
    image1 = random.choice(image_paths)
    
    # remove image1 to avoid a duplicate
    image_paths.remove(image1)
    image2 = random.choice(image_paths)

    images = [Image.open(x) for x in [image1, image2]]
    widths, heights = zip(*(i.size for i in images))
    max_width = max(widths)
    total_height = sum(heights) + offset_height
    new_im = Image.new('RGB', (max_width, total_height), 'WHITE')
    y_offset = 0
    for im in images:
        new_im.paste(im, (0, y_offset))
        y_offset += im.size[1] + offset_height
    new_im.save(output_path)
    return [image1, image2]

if __name__ == '__main__':
    temp_image = "/tmp/combined_image_temp.png"  # save to tmpfs dir
    final_image = "/tmp/combined_image.png"
    chosen_images = combine_images(temp_image, args.offset)
    print(datetime.now(), "- Combining images:", ' & '.join(chosen_images))
    os.rename(temp_image, final_image)

