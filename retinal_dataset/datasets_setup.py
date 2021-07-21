import os

import albumentations as A
import cv2
import nntools as nt
import nntools.dataset as D
import numpy as np
import torch

from scripts.utils import Dataset


@D.nntools_wrapper
def process_masks(Exudates, Microaneurysms, Hemorrhages, Cotton_Wool_Spot):
    mask = np.stack((Cotton_Wool_Spot > 0, Exudates > 0, Hemorrhages > 0, Microaneurysms > 0), 2)
    return {'mask': mask.astype(np.uint8)}


@D.nntools_wrapper
def process_masks_kaggle(Exudates, Microaneurysms, Hemorrhages, Cotton_Wool_Spot):
    mask = np.stack((Cotton_Wool_Spot > 127, Exudates > 127, Hemorrhages > 127, Microaneurysms > 127), 2)
    return {'mask': mask.astype(np.uint8)}


@D.nntools_wrapper
def autocrop(image, mask):
    blur = 5
    threshold = 10
    threshold_img = cv2.blur(image, (blur, blur), borderType=cv2.BORDER_REPLICATE)
    if threshold_img.ndim == 3:
        threshold_img = np.mean(threshold_img, axis=2)
    not_null_pixels = np.nonzero(threshold_img > threshold)
    x_range = (np.min(not_null_pixels[1]), np.max(not_null_pixels[1]))
    y_range = (np.min(not_null_pixels[0]), np.max(not_null_pixels[0]))
    d = {'image': image[y_range[0]:y_range[1], x_range[0]:x_range[1]],
         'mask': mask[y_range[0]:y_range[1], x_range[0]:x_range[1]]}
    return d


@D.nntools_wrapper
def autocrop_image(image):
    blur = 5
    threshold = 10
    img = cv2.blur(image, (blur, blur), borderType=cv2.BORDER_REPLICATE)
    if img.ndim == 3:
        img = np.mean(img, axis=2)
    not_null_pixels = np.nonzero(img > threshold)
    try:
        x_range = (np.min(not_null_pixels[1]), np.max(not_null_pixels[1]))
        y_range = (np.min(not_null_pixels[0]), np.max(not_null_pixels[0]))
        d = {'image': image[y_range[0]:y_range[1], x_range[0]:x_range[1]]}
        return d
    except ValueError:
        return {'image': image}


def sort_func_idrid(x):
    return '_'.join(x.split('.')[0].split('_')[:2])


def get_idrid_dataset(root_img, root_mask, input_shape=(1024, 1024)):
    input_shape = tuple(input_shape)
    ma = os.path.join(root_mask, '1. Microaneurysms/')
    he = os.path.join(root_mask, '2. Haemorrhages/')
    ex = os.path.join(root_mask, '3. Hard Exudates/')
    se = os.path.join(root_mask, '4. Soft Exudates/')
    masks = {'Exudates': ex,
             'Microaneurysms': ma,
             'Hemorrhages': he,
             'Cotton_Wool_Spot': se}
    segmentDataset = D.SegmentationDataset(root_img, masks, input_shape, filling_strategy=nt.NN_FILL_UPSAMPLE,
                                           keep_size_ratio=True,
                                           extract_image_id_function=sort_func_idrid)
    composer = D.Composition()
    composer << process_masks << autocrop << A.LongestMaxSize(max_size=max(input_shape), always_apply=True) \
    << A.PadIfNeeded(min_height=input_shape[0], min_width=input_shape[1], border_mode=cv2.BORDER_CONSTANT,
                     value=0, mask_value=0, always_apply=True)
    segmentDataset.set_composition(composer)
    segmentDataset.tag = Dataset.IDRID
    return segmentDataset


def get_kaggle_dataset(root_img, input_shape):
    input_shape = tuple(input_shape)
    segmentDataset = D.SegmentationDataset(root_img, shape=input_shape, keep_size_ratio=True)
    composer = D.Composition()
    composer << autocrop_image << A.LongestMaxSize(max_size=max(input_shape), always_apply=True)
    composer << A.PadIfNeeded(min_height=input_shape[0], min_width=input_shape[1],
                              border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, always_apply=True)
    segmentDataset.set_composition(composer)
    return segmentDataset


def get_kaggle_segmented_dataset(root_img, root_mask, input_shape):
    input_shape = tuple(input_shape)
    ma = os.path.join(root_mask, 'Microaneurysms/')
    he = os.path.join(root_mask, 'Hemorrhages/')
    ex = os.path.join(root_mask, 'Exudates/')
    se = os.path.join(root_mask, 'Cotton_Wool_Spot/')
    masks = {'Exudates': ex,
             'Microaneurysms': ma,
             'Hemorrhages': he,
             'Cotton_Wool_Spot': se}
    max_shape = (max(input_shape), max(input_shape))
    segmentDataset = D.SegmentationDataset(root_img, masks, max_shape,
                                           keep_size_ratio=True,
                                           filling_strategy=nt.NN_FILL_UPSAMPLE)
    composer = D.Composition()
    composer << autocrop_image << A.LongestMaxSize(max_size=max(input_shape), always_apply=True)
    composer << A.PadIfNeeded(min_height=input_shape[0], min_width=input_shape[1],
                              border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, always_apply=True)
    composer << process_masks_kaggle
    segmentDataset.set_composition(composer)
    segmentDataset.tag = Dataset.KAGGLE_TEACHER
    return segmentDataset


def get_messidor_dataset(root_img, root_mask, input_shape=(1024, 1024)):
    input_shape = tuple(input_shape)
    ma = os.path.join(root_mask, 'Red/Microaneurysms/')
    he = os.path.join(root_mask, 'Red/Hemorrhages/')
    ex = os.path.join(root_mask, 'Bright/Exudates/')
    se = os.path.join(root_mask, 'Bright/Cotton Wool Spots/')
    masks = {'Exudates': ex,
             'Microaneurysms': ma,
             'Hemorrhages': he,
             'Cotton_Wool_Spot': se}

    max_shape = (max(input_shape), max(input_shape))
    segmentDataset = D.SegmentationDataset(root_img, masks, input_shape,
                                           keep_size_ratio=True,
                                           filling_strategy=nt.NN_FILL_UPSAMPLE)
    composer = D.Composition()
    composer << process_masks
    if max_shape != input_shape:
        composer << A.CenterCrop(*input_shape[::-1])
    composer << A.PadIfNeeded(min_height=input_shape[0], min_width=input_shape[1],
                              border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, always_apply=True)
    segmentDataset.set_composition(composer)
    segmentDataset.tag = Dataset.MESSIDOR
    return segmentDataset


def get_fgadr_dataset(root_img, root_mask, input_shape=(1024, 1024)):
    input_shape = tuple(input_shape)
    ma = os.path.join(root_mask, 'Microaneurysms_Masks/')
    he = os.path.join(root_mask, 'Hemohedge_Masks/')
    ex = os.path.join(root_mask, 'HardExudate_Masks/')
    se = os.path.join(root_mask, 'SoftExudate_Masks/')
    masks = {'Exudates': ex,
             'Microaneurysms': ma,
             'Hemorrhages': he,
             'Cotton_Wool_Spot': se,
             }

    max_shape = (max(input_shape), max(input_shape))
    segmentDataset = D.SegmentationDataset(root_img, masks, input_shape,
                                           keep_size_ratio=True,
                                           filling_strategy=nt.NN_FILL_UPSAMPLE)
    composer = D.Composition()
    composer << process_masks
    if max_shape != input_shape:
        composer << A.CenterCrop(*input_shape[::-1])
    composer << A.PadIfNeeded(min_height=input_shape[0], min_width=input_shape[1],
                              border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, always_apply=True)
    segmentDataset.set_composition(composer)
    segmentDataset.tag = Dataset.FGADR
    return segmentDataset


def get_retlesions_dataset(root_img, root_mask, input_shape=(1024, 1024)):
    input_shape = tuple(input_shape)
    ma = os.path.join(root_mask, 'microaneurysm/')
    he = os.path.join(root_mask, 'retinal_hemorrhage/')
    ex = os.path.join(root_mask, 'hard_exudate/')
    se = os.path.join(root_mask, 'cotton_wool_spots/')
    masks = {'Exudates': ex,
             'Microaneurysms': ma,
             'Hemorrhages': he,
             'Cotton_Wool_Spot': se}

    max_shape = (max(input_shape), max(input_shape))
    segmentDataset = D.SegmentationDataset(root_img, masks, input_shape,
                                           keep_size_ratio=True,
                                           filling_strategy=nt.NN_FILL_UPSAMPLE)
    composer = D.Composition()
    composer << process_masks << autocrop << A.LongestMaxSize(max_size=max(input_shape), always_apply=True)
    if max_shape != input_shape:
        composer << A.CenterCrop(*input_shape[::-1])
    composer << A.PadIfNeeded(min_height=input_shape[0], min_width=input_shape[1],
                              border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, always_apply=True)
    segmentDataset.set_composition(composer)
    segmentDataset.tag = Dataset.RETINAL_LESIONS
    return segmentDataset


def get_DDR_dataset(root_img, root_mask, input_shape=(1024, 1024)):
    input_shape = tuple(input_shape)
    ma = os.path.join(root_mask, 'MA/')
    he = os.path.join(root_mask, 'HE/')
    ex = os.path.join(root_mask, 'EX/')
    se = os.path.join(root_mask, 'SE/')
    masks = {'Exudates': ex,
             'Microaneurysms': ma,
             'Hemorrhages': he,
             'Cotton_Wool_Spot': se}

    max_shape = (max(input_shape), max(input_shape))
    segmentDataset = D.SegmentationDataset(root_img, masks, input_shape,
                                           keep_size_ratio=True,
                                           filling_strategy=nt.NN_FILL_UPSAMPLE)
    composer = D.Composition()
    composer << process_masks << autocrop << A.LongestMaxSize(max_size=max(input_shape), always_apply=True)
    if max_shape != input_shape:
        composer << A.CenterCrop(*input_shape[::-1], always_apply=True)
    composer << A.PadIfNeeded(min_height=input_shape[0], min_width=input_shape[1],
                              border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, always_apply=True)
    segmentDataset.set_composition(composer)
    segmentDataset.tag = Dataset.DDR
    return segmentDataset


def split_dataset(dataset, ratio_validation, seed):
    if ratio_validation:
        train_len = int(len(dataset) * (1 - ratio_validation))
        valid_len = len(dataset) - train_len
        train_dataset, valid_dataset = D.random_split(dataset, [train_len, valid_len],
                                                      generator=torch.Generator().manual_seed(seed))
        return train_dataset, valid_dataset
    else:
        return dataset, None


def get_datasets(roots_idrid=(None, None),
                 roots_messidor=(None, None),
                 roots_fgadr=(None, None),
                 root_retlesion=(None, None),
                 root_kaggle_segmented=(None, None),
                 root_ddr=(None, None),
                 shape=(1024, 1024),
                 split_ratio=0.15,
                 seed=1234):
    outputs = {'core': [], 'split': []}
    roots = [(roots_idrid, get_idrid_dataset), (roots_messidor, get_messidor_dataset),
             (roots_fgadr, get_fgadr_dataset), (root_retlesion, get_retlesions_dataset),
             (root_kaggle_segmented, get_kaggle_segmented_dataset),
             (root_ddr, get_DDR_dataset)]

    for r, func in roots:
        if all(r):
            dataset = func(root_img=r[0], root_mask=r[1], input_shape=shape)
            train, val = split_dataset(dataset, split_ratio, seed)
            outputs['core'].append(train)
            if val is not None:
                outputs['split'].append(val)

    return outputs


def get_datasets_from_config(c, sets, seed=1234, shape=None, split_ratio=0):
    img_idrid_root = c.get('img_idrid_url', None) if Dataset.IDRID in sets else None
    mask_idrid_root = c.get('mask_idrid_url', None) if Dataset.IDRID in sets else None

    img_messidor_root = c.get('img_messidor_url', None) if Dataset.MESSIDOR in sets else None
    mask_messidor_root = c.get('mask_messidor_url', None) if Dataset.MESSIDOR in sets else None

    img_fgadr_root = c.get('img_fgadr_url', None) if Dataset.FGADR in sets else None
    mask_fgadr_root = c.get('mask_fgadr_url', None) if Dataset.FGADR in sets else None

    img_retles_root = c.get('img_retles_url', None) if Dataset.RETINAL_LESIONS in sets else None
    mask_retles_root = c.get('mask_retles_url', None) if Dataset.RETINAL_LESIONS in sets else None

    img_ddr_root = c.get('img_ddr_url', None) if Dataset.DDR in sets else None
    mask_ddr_root = c.get('mask_ddr_url', None) if Dataset.DDR in sets else None

    img_kaggle_root = c.get('img_kaggle_url', None) if Dataset.KAGGLE_TEACHER in sets else None
    mask_kaggle_root = c.get('mask_kaggle_url', None) if Dataset.KAGGLE_TEACHER in sets else None

    if shape is None:
        shape = c['shape']
    shape = tuple(shape)
    return get_datasets(roots_idrid=(img_idrid_root, mask_idrid_root),
                        roots_fgadr=(img_fgadr_root, mask_fgadr_root),
                        roots_messidor=(img_messidor_root, mask_messidor_root),
                        root_retlesion=(img_retles_root, mask_retles_root),
                        root_kaggle_segmented=(img_kaggle_root, mask_kaggle_root),
                        root_ddr=(img_ddr_root, mask_ddr_root),
                        shape=shape,
                        split_ratio=split_ratio,
                        seed=seed)


def add_operations_to_dataset(datasets, aug=None):
    if aug is None:
        return
    if not isinstance(datasets, list):
        datasets = [datasets]
    for d in datasets:
        if isinstance(aug, list):
            d.composer.add(*aug)
        else:
            d.composer.add(aug)
