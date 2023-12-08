import numpy as np
from PIL import Image
import pywt
import argparse
import os
from tqdm import tqdm

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', default='E:\\project_TransUNet\\data\\Synapse\\train_npz')
    parser.add_argument('--L_path', default='E:\\project_TransUNet\\data\\Synapse\\train_npzL')
    parser.add_argument('--H_path', default='E:\\project_TransUNet\\data\\Synapse\\train_npzH')
    parser.add_argument('--wavelet_type', default='db2', help='haar, db2, bior1.5, bior2.4, coif1, dmey')
    parser.add_argument('--if_RGB', default=False)
    parser.add_argument('--numpy_file', default=True)
    args = parser.parse_args()

    if not os.path.exists(args.L_path):
        os.mkdir(args.L_path)
    if not os.path.exists(args.H_path):
        os.mkdir(args.H_path)

    image_files = [f for f in os.listdir(args.image_path) if f.endswith('.png') or f.endswith('.jpg') or f.endswith(".tif")]
    numpy_files = [f for f in os.listdir(args.image_path) if f.endswith('.npz')]

    for i in tqdm(numpy_files):
        image_path = os.path.join(args.image_path, i)
        #L_path = os.path.join(args.L_path, i)
        #H_path = os.path.join(args.H_path, i)
        L_path = os.path.join(args.L_path, os.path.splitext(i)[0])
        H_path = os.path.join(args.H_path, os.path.splitext(i)[0])

        if args.if_RGB:
            image = Image.open(image_path).convert('L')
            image = np.array(image)
        elif args.numpy_file:
            image = np.load(image_path)["image"]
        else:
            image = Image.open(image_path)
            image = np.array(image)

        LL, (LH, HL, HH) = pywt.dwt2(image, args.wavelet_type)

        LL = (LL - LL.min()) / (LL.max() - LL.min()) * 255

        LL = Image.fromarray(LL.astype(np.uint8))
        LL.save(L_path + ".png")

        LH = (LH - LH.min()) / (LH.max() - LH.min()) * 255
        HL = (HL - HL.min()) / (HL.max() - HL.min()) * 255
        HH = (HH - HH.min()) / (HH.max() - HH.min()) * 255

        merge1 = HH + HL + LH
        merge1 = (merge1-merge1.min()) / (merge1.max()-merge1.min()) * 255

        merge1 = Image.fromarray(merge1.astype(np.uint8))
        merge1.save(H_path + ".png")