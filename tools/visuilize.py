import numpy as np
from PIL import Image
import pywt
import argparse
import os
from tqdm import tqdm

import matplotlib.pyplot as plt

import numpy as np
from PIL import Image
import pywt
import argparse
import os
from tqdm import tqdm

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', default='E:\\project_TransUNet\\data\\Synapse\\train_npz')
    parser.add_argument('--L_path', default='E:\\ACDC\\trainL')
    parser.add_argument('--H_path', default='E:\\ACDC\\trainH')
    parser.add_argument('--wavelet_type', default='haar', help='haar, db2, bior1.5, bior2.4, coif1, dmey')
    parser.add_argument('--if_RGB', default=False)
    parser.add_argument('--numpy_file', default=True)
    parser.add_argument('--train_img', default='E:\\project_TransUNet\\data\\Synapse\\train_img')
    parser.add_argument('--train_label', default='E:\\project_TransUNet\\data\\Synapse\\train_label')
    args = parser.parse_args()

    #if not os.path.exists(args.L_path):
    #    os.mkdir(args.L_path)
    #if not os.path.exists(args.H_path):
    #    os.mkdir(args.H_path)

    if not os.path.exists(args.train_img):
        os.mkdir(args.train_img)
    if not os.path.exists(args.train_label):
        os.mkdir(args.train_label)


    image_files = [f for f in os.listdir(args.image_path) if f.endswith('.png') or f.endswith('.jpg') or f.endswith(".tif")]
    numpy_files = [f for f in os.listdir(args.image_path) if f.endswith('.npz')]

    for i in tqdm(numpy_files):
        image_path = os.path.join(args.image_path, i)
        train_img_path = os.path.join(args.train_img, os.path.splitext(i)[0])
        train_label_path = os.path.join(args.train_label, os.path.splitext(i)[0])
        #L_path = os.path.join(args.L_path, os.path.splitext(i)[0])
        #H_path = os.path.join(args.H_path, os.path.splitext(i)[0])

        if args.if_RGB:
            image = Image.open(image_path).convert('L')
            image = np.array(image)
        elif args.numpy_file:
            image = np.load(image_path)

        else:
            image = Image.open(image_path)
            image = np.array(image)

        train_img = image["image"] * 255
        train_img = Image.fromarray(train_img.astype(np.uint8))
        train_img.save(train_img_path + ".png")

        train_label = image["label"] * 255
        train_label = Image.fromarray(train_label.astype(np.uint8))
        train_label.save(train_label_path + ".png")





