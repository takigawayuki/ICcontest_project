from torch.utils.data import *
from imutils import paths
import numpy as np
import random
import cv2
import os

CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
         '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
         '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
         '新',
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
         'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z', 'I', 'O', '-'
         ]

CHARS_DICT = {char:i for i, char in enumerate(CHARS)}


def imread_unicode(filename):
    data = np.fromfile(filename, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def parse_plate_name(filename):
    basename = os.path.basename(filename)
    imgname, _ = os.path.splitext(basename)
    return imgname.split("-")[0].split("_")[0]


def is_valid_plate_name(plate_name):
    if len(plate_name) not in (7, 8):
        return False

    for c in plate_name:
        if c not in CHARS_DICT:
            return False

    # Blue plates are 7 chars, keep them directly.
    if len(plate_name) == 7:
        return True

    # Green plates are 8 chars and should contain D/F in the 3rd or last position.
    return plate_name[2] in ("D", "F") or plate_name[-1] in ("D", "F")

class LPRDataLoader(Dataset):
    def __init__(self, img_dir, imgSize, lpr_max_len, PreprocFun=None):
        self.img_dir = img_dir
        self.img_paths = []
        invalid_img_paths = []
        for i in range(len(img_dir)):
            for image_path in paths.list_images(img_dir[i]):
                plate_name = parse_plate_name(image_path)
                if is_valid_plate_name(plate_name):
                    self.img_paths.append(image_path)
                else:
                    invalid_img_paths.append(image_path)
        random.shuffle(self.img_paths)
        if invalid_img_paths:
            print("[Info] Skip {} invalid plate labels.".format(len(invalid_img_paths)))
            for invalid_path in invalid_img_paths[:10]:
                print("[Info] Invalid label sample: {}".format(parse_plate_name(invalid_path)))
        self.img_size = imgSize
        self.lpr_max_len = lpr_max_len
        if PreprocFun is not None:
            self.PreprocFun = PreprocFun
        else:
            self.PreprocFun = self.transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        filename = self.img_paths[index]
        image = imread_unicode(filename)
        if image is None:
            raise FileNotFoundError("Failed to read or decode image: {}".format(filename))

        height, width, _ = image.shape
        if height != self.img_size[1] or width != self.img_size[0]:
            image = cv2.resize(image, self.img_size)
        image = self.PreprocFun(image)

        imgname = parse_plate_name(filename)
        label = list()
        for c in imgname:
            if c not in CHARS_DICT:
                raise KeyError("Unsupported label character '{}' from file '{}'".format(c, filename))
            label.append(CHARS_DICT[c])

        # return image, label, len(label)

        plate_len = len(label)

        # 👇 加一个标志（蓝牌/绿牌）
        plate_type = 0 if plate_len == 7 else 1

        return image, label, plate_len, plate_type

    def transform(self, img):
        img = img.astype('float32')
        img -= 127.5
        img *= 0.0078125
        img = np.transpose(img, (2, 0, 1))

        return img

