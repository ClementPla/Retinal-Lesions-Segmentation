from enum import auto, IntFlag, Flag

import albumentations as A
import cv2


class DA(IntFlag):
    NONE = auto()
    GEOMETRIC = auto()
    COLOR = auto()
    COLOR_V2 = auto()
    GEOMETRIC_SIMPLE = auto()
    FLIP = auto()

    @property
    def name(self):
        name = super(DA, self).name
        if name:
            return name
        else:
            return ', '.join([flag.name for flag in DA if flag in self])


class Dataset(Flag):
    IDRID = auto()
    RETINAL_LESIONS = auto()
    MESSIDOR = auto()
    FGADR = auto()
    KAGGLE_TEACHER = auto()
    DDR = auto()

    @property
    def name(cls):
        name = super(Dataset, cls).name
        if name:
            return name
        else:
            return [flag.name for flag in Dataset if flag in cls]

    @property
    def str_name(cls):
        name = cls.name
        if isinstance(name, list):
            return '_'.join(name)
        else:
            return name

    @property
    def suffix(cls):
        name = cls.name
        if isinstance(name, list):
            return ["_%s" % f.lower() for f in cls.name]
        else:
            return "_%s" % name.lower()

    @property
    def length(cls):
        name = cls.name
        if isinstance(name, list):
            return len(name)
        else:
            return 1


class Lesions(Flag):
    COTTON_WOOL_SPOT = auto()
    EXUDATES = auto()
    HEMORRHAGES = auto()
    MICROANEURYSMS = auto()

    @property
    def name(cls):
        name = super(Lesions, cls).name
        if name:
            return name
        else:
            return [flag.name for flag in Lesions if flag in cls]

    @property
    def length(cls):
        name = cls.name
        if isinstance(name, list):
            return len(name)
        else:
            return 1


DA_FUNCTIONS = {DA.NONE: [],
                DA.GEOMETRIC: [A.HorizontalFlip(p=0.5),
                               A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.10, rotate_limit=15,
                                                  border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0)],
                DA.COLOR: [A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
                           A.GaussianBlur(blur_limit=7, p=0.25), A.Sharpen(p=0.25),
                           A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=0, val_shift_limit=0)],
                DA.COLOR_V2: [A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.2),
                              A.OneOf([A.GaussianBlur(blur_limit=7, p=0.5), A.Sharpen(p=0.5)]),
                              A.OneOf([A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=15, val_shift_limit=0),
                                       A.RandomGamma()])],
                DA.GEOMETRIC_SIMPLE: [A.HorizontalFlip(p=0.5),
                                      A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=5,
                                                         border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0)],
                DA.FLIP: [A.HorizontalFlip(p=0.5)]
                }


def get_augmentation_functions(flags):
    f = []
    for k, v in DA_FUNCTIONS.items():
        if k in flags:
            f += v
    return f
