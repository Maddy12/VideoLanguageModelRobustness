import sys

import cv2
import numpy as np
import os
from argparse import ArgumentParser
from scipy.ndimage import zoom as scizoom
from tqdm import tqdm
import pandas as pd
import ctypes
import skimage
import random
from pathlib import Path
import pdb

from wand.image import Image as WandImage
from wand.api import library as wandlibrary
from PIL import Image
from io import BytesIO


def disk(radius, alias_blur=0.1, dtype=np.float32):
    if radius <= 8:
        L = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        L = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    X, Y = np.meshgrid(L, L)
    aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)

    # supersample disk to antialias
    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)


def defocus_blur(x, severity=1):
    c = [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)][severity - 1]

    x = (np.array(x) / 255.0).astype(np.float32)
    kernel = disk(radius=c[0], alias_blur=c[1])
    # *255
    channels = []
    for d in range(3):
        channels.append(cv2.filter2D(x[:, :, d], -1, kernel))
    channels = np.array(channels).transpose((1, 2, 0))  # 3x224x224 -> 224x224x3

    return Image.fromarray(np.uint8(np.clip(channels, 0, 1) * 255))


def jpeg_compression(x, severity=1):
    c = [25, 18, 15, 10, 7][severity - 1]

    output = BytesIO()
    x.save(output, 'JPEG', quality=c)
    x = Image.open(output)

    return x


def rotate(x, sev):
    deg = [7.5, 15, 30, 45, 90, 180][sev - 1]
    return x.rotate(deg)


def variable_rotate(x, sev):
    deg = np.random.randint(-6 * sev, 6 * sev + 1)
    return x.rotate(deg)


def translate(x, sev):
    deg = np.random.randint(-6 * sev, 6 * sev + 1)
    return x.rotate(deg)


def salt_blur(x, severity=1):
    im1 = np.array(x)
    mask = np.random.randint(0, 100, im1.shape)
    im2 = np.where(mask < severity * 10, 255, im1)
    return Image.fromarray(np.uint8(np.clip(im2, 0, 255)))


def pepper_blur(x, severity=1):
    im1 = np.array(x)
    mask = np.random.randint(0, 100, im1.shape)
    im2 = np.where(mask < severity * 10, 0, im1)
    return Image.fromarray(np.uint8(np.clip(im2, 0, 255)))


def impulse_noise(x, severity=2):
    c = [.03, .06, .09, 0.17, 0.27][severity - 1]
    temp = x
    x = skimage.util.random_noise(np.array(x) / 255., mode='s&p', amount=c)
    return Image.fromarray(np.uint8(np.clip(x, 0, 1) * 255))


def shot_noise(x, severity=4):
    c = [250, 100, 50, 30, 15][severity - 1]
    temp = x

    x = np.array(x) / 255.
    return Image.fromarray(np.uint8(np.clip(np.random.poisson(x * c) / c, 0, 1) * 255))


def gaussian_noise(x, severity=1):
    c = [.08, .12, 0.18, 0.26, 0.38][severity - 1]
    temp = x

    x = np.array(x) / 255.
    return Image.fromarray(np.uint8(np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255))


def speckle_noise(x, severity=1):
    c = [.15, .2, 0.25, 0.3, 0.35][severity - 1]

    x = np.array(x) / 255.
    return Image.fromarray(np.uint8(np.clip(x + x * np.random.normal(size=x.shape, scale=c), 0, 1) * 255))


# Tell Python about the C method
wandlibrary.MagickMotionBlurImage.argtypes = (ctypes.c_void_p,  # wand
                                              ctypes.c_double,  # radius
                                              ctypes.c_double,  # sigma
                                              ctypes.c_double)  # angle


# Extend wand.image.Image class to include method signature
class MotionImage(WandImage):
    def motion_blur(self, radius=0.0, sigma=0.0, angle=0.0):
        wandlibrary.MagickMotionBlurImage(self.wand, radius, sigma, angle)


def motion_blur(x, severity=1):
    c = [(10, 3), (15, 5), (15, 8), (15, 12), (20, 15)][severity - 1]

    output = BytesIO()
    x.save(output, format='PNG')
    x = MotionImage(blob=output.getvalue())

    x.motion_blur(radius=c[0], sigma=c[1], angle=np.random.uniform(-45, 45))

    x = cv2.imdecode(np.fromstring(x.make_blob(), np.uint8),
                     cv2.IMREAD_UNCHANGED)

    if x.shape != (256, 256):
        return Image.fromarray(np.uint8(np.clip(x[..., [2, 1, 0]], 0, 255)))  # BGR to RGB
    else:  # greyscale to RGB
        return Image.fromarray(np.uint8(np.clip(np.array([x, x, x]).transpose((1, 2, 0)), 0, 255)))


def clipped_zoom(img, zoom_factor):
    h = img.shape[0]
    w = img.shape[1]
    # ceil crop height(= crop width)
    ch = int(np.ceil(h / zoom_factor))
    ch2 = int(np.ceil(w / zoom_factor))

    top = (h - ch) // 2
    top2 = (w - ch2) // 2
    img = scizoom(img[top:top + ch, top2:top2 + ch2], (zoom_factor, zoom_factor, 1), order=1)
    # trim off any extra pixels
    trim_top = (img.shape[0] - h) // 2
    trim_top2 = (img.shape[1] - w) // 2
    return img[trim_top:trim_top + h, trim_top2:trim_top2 + w]


def zoom_blur(x, severity=1):
    c = [np.arange(1, 1.11, 0.01),
         np.arange(1, 1.16, 0.01),
         np.arange(1, 1.21, 0.02),
         np.arange(1, 1.26, 0.02),
         np.arange(1, 1.31, 0.03)][severity - 1]

    x = (np.array(x) / 255.).astype(np.float32)
    out = np.zeros_like(x)
    for zoom_factor in c:
        out += clipped_zoom(x, zoom_factor)

    x = (x + out) / (len(c) + 1)
    return Image.fromarray(np.uint8(np.clip(x, 0, 1) * 255))


def create_codec2(vidpath1, vidpath2, severity=1):
    os.system(
        'ffmpeg -i "{}" -c:v mpeg4 -q:v {} "{}"'.format(vidpath1, 15 * int(severity), vidpath2))


def create_codec1(vidpath1, vidpath2, severity=1):
    os.system(
        'ffmpeg -i "{}" -c:v mpeg2video -q:v {} -c:a mp2 -f vob -y "{}" '.format(vidpath1, 20 * int(severity), vidpath2))


if __name__ == '__main__':
    argparser = ArgumentParser(prog='noise_dataset',
                               description='Code to save noise dataset for multimodal datasets')

    # Required arguments
    argparser.add_argument('dataset', type=str, help="Either YouCook2 or MSRVTT")
    argparser.add_argument('data_root', type=str, help="Where the original videos are stored, lowest parent level.")
    argparser.add_argument('noisy_root', type=str, help="Where to store the noisy version of videos.")
    argparser.add_argument('type', type=str, help="The category of video perturbations")

    args = argparser.parse_args()
    # Methods we want to collect
    if args.type == 'temporal':
        methods = ["freeze", "box_jumble", "jumble", "sample", "reverse_sample", "freeze"]
    elif args.type == 'blur':
        methods = ['defocus_blur', 'motion_blur', 'zoom_blur']
    elif args.type == 'camera':
        methods = ['translate', 'rotate', 'var_rotate']
    elif args.type == 'noise':
        methods = ['impulse_noise', 'speckle_noise', 'gaussian_noise', 'shot_noise']
    elif args.type == 'digital':
        methods = ['jpeg',  'codec1', 'codec2']
    else:
        print("Passed incorrect type")
        exit()

    fn_dict = {"gaussian_noise": gaussian_noise, "shot_noise": shot_noise, "impulse_noise": impulse_noise,
               "speckle_noise": speckle_noise,  "defocus_blur": defocus_blur,  "jpeg": jpeg_compression,
               "static_rotate": rotate, "rotate": variable_rotate, "zoom_blur": zoom_blur, "motion_blur": motion_blur,
               'codec1': create_codec1, 'codec2': create_codec2}
    pbar0 = tqdm(methods, total=len(methods))


    # Iterate through the methods we want
    for method in pbar0:
        pbar0.set_postfix({'method': method})
        severity = range(1, 6)

        # Iterate through the severities we want
        pbar1 = tqdm(severity, total=len(severity), position=1, leave=False)
        for sev in pbar1:
            videos = pd.read_csv(f'datasets/{args.dataset}_videolist.csv')

            pbar1.set_postfix({'method': method, 'severity': sev})
            noisy_root = os.path.join(args.noisy_root, f"{method}_{sev}")
            if not os.path.exists(noisy_root):
                Path(noisy_root).mkdir(parents=True)
            pbar2 = tqdm(videos.iterrows(), total=len(videos), position=2, leave=False)
            # Iterate through the videos we want
            for idx, row in pbar2:

                frame_list = list()
                oldpath = row[0]
                newpath = oldpath.replace(args.data_root, noisy_root)
                newroot = '/'.join(newpath.split('/')[:-1])
                if not os.path.exists(newroot):
                    Path(newroot).mkdir(parents=True)

                pbar2.set_postfix({'method': method, 'severity': sev, 'video': row[0].replace(args.data_root, ''), 'new_video': newpath})

                if sys.getsizeof(oldpath) > 5e+10:
                    print(f"{oldpath} failed due to memory size greater than 50GB. Skipping...")
                    frame_list = list()
                    continue

                assert os.path.isfile(oldpath), f"File does not exist, please pass correct root dir. Path: {oldpath}"

                # Run MPEG1
                if method == "codec1":
                    create_codec1(oldpath, newpath, sev)
                    continue
                # Run MPEG2
                elif method == "codec2":
                    if 'mp4' in newpath:
                        newpath = newpath.replace('.mp4', '.avi')
                    else:
                        newpath = newpath.replace('.mkv', '.avi')
                    create_codec2(oldpath, newpath, sev)
                    continue

                # Load video
                try:
                    vidcap = cv2.VideoCapture(oldpath)
                except:
                    print(f"Video {oldpath} failed...skipping...")
                    continue

                fps = vidcap.get(cv2.CAP_PROP_FPS)

                # Read first frame
                success, image = vidcap.read()
                if not success:
                    print(f"Failed for {oldpath}. Skipping...")
                    continue
                height, width, layers = image.shape
                size = (width, height)

                # Generate CV2 writer
                if method == "translate":
                    out = cv2.VideoWriter(newpath, cv2.VideoWriter_fourcc(*'mp4v'), fps,
                                          (224, 224))
                else:
                    out = cv2.VideoWriter(newpath, cv2.VideoWriter_fourcc(*'mp4v'), fps,
                                          size)

                if method == "translate":
                    if width > height:
                        width2 = int((width * 256) / height)
                        height2 = 256
                    else:
                        width2 = 256
                        height2 = int((height * 256) / width)

                # Iterate through each frame, add noise, and write frame
                prev = image
                while success:
                    if method == "translate":
                        img2 = cv2.resize(image, (width2, height2))
                        img2 = np.array(img2)
                        x = np.random.randint(0, 8 * sev)
                        y = np.random.randint(0, 8 * sev)
                        noise = img2[x:x + 224, y:y + 224]
                        out.write(noise)

                    elif args.type == 'temporal':
                        frame_list.append(image)
                        pbar2.set_postfix({'method': method, 'severity': sev, 'video': row[0], 'size': sys.getsizeof(frame_list)*1e-9})
                    else:
                        noise = fn_dict[method](Image.fromarray(image), sev)
                        noise = np.array(noise)
                        out.write(noise)

                    prev = image
                    success, image = vidcap.read()

                if method == "jumble":
                    list2 = []
                    indices = list(range(0, len(frame_list)))
                    x = 0
                    seg = 2 ** (1 + sev)
                    while x < len(frame_list):
                        tmp = indices[x:x + seg]
                        random.shuffle(tmp)
                        list2.extend(tmp)
                        x = x + seg
                    for f in list2:
                        out.write(frame_list[f])

                if method == "box_jumble":
                    list2 = []
                    indices = list(range(0, len(frame_list)))
                    x = 0
                    seg = 2 ** (sev)
                    while x < len(frame_list):
                        tmp = indices[x:x + seg]
                        list2.append(tmp)
                        x = x + seg
                    random.shuffle(list2)
                    list2 = sum(list2, [])

                    for f in list2:

                        out.write(frame_list[f])

                if method == "sample" or method == 'reverse_sample':
                    sample_rate = [2, 4, 8, 16, 32][sev - 1]
                    n_frames = len(frame_list)
                    frame_list = frame_list[::sample_rate]

                    if method == 'reverse_sample':
                        frame_list.reverse()

                    for f in frame_list:
                        for _ in range(sample_rate):
                            out.write(f)

                if method == 'freeze':
                    total = len(frame_list)
                    # k = int([.4 * total, .2 * total, .1 * total, .05 * total, 1][sev - 1])
                    k = int([.4 * total, .2 * total, .1 * total, max(.05 * total, 2), max(.1 * total, 1)][sev - 1])
                    final = list()
                    indices = list(range(0, total))
                    subselect = random.sample(indices, k=k)
                    subselect.sort()
                    prev = 0
                    for idx, frame_ind in enumerate(subselect):
                        if idx + 1 < len(subselect):
                            # Add as many until the next one
                            b = (subselect[idx + 1]) - frame_ind
                        else:
                            # If at the end of the list, go until total
                            b = total - frame_ind
                        final.extend([frame_ind]*b)
                        # final.extend([frame_list[frame_ind]] * b)

                    for f in final:
                        out.write(frame_list[f])

                vidcap.release()
                out.release()
                pbar2.set_postfix({'method': method, 'severity': sev,
                                   'video': row[0].replace(args.data_root, ''),
                                   'new_video': newpath, 'status': "complete"})
