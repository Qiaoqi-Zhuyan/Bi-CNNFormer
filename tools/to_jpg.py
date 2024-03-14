from PIL import Image
from tqdm import tqdm
import os

def tif_to_jpeg(tif_path):
    image_files = [f for f in os.listdir(tif_path) if f.endswith('.tif')]
    for tif_file in image_files:
        tif_pic = os.path.join(tif_path, tif_file)
        img = Image.open(tif_path)
        output_filename = os.path.splitext(tif_file)[0] + ".jpg"
        img.save(output_filename, 'JPEG')

tif_path = "..\\dataset"

if __name__ == "__main__":
    tif_to_jpeg(tif_path)