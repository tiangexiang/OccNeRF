#!/usr/bin/env python

import functools
import os.path
import random
import sys
import xml.etree.ElementTree
import numpy as np
import matplotlib.pyplot as plt
import skimage.data
import cv2
import PIL.Image


def main():
    """Demo of how to use the code"""
    
    # path = 'something/something/VOCtrainval_11-May-2012/VOCdevkit/VOC2012'
    path = sys.argv[1]

    print('Loading occluders from Pascal VOC dataset...')
    occluders = load_occluders(pascal_voc_root_path=path)
    print('Found {} suitable objects'.format(len(occluders)))

    original_im = cv2.resize(skimage.data.astronaut(), (256,256))
    fig, axarr = plt.subplots(3,3, figsize=(7,7))
    for ax in axarr.ravel():
        occluded_im = occlude_with_objects(original_im, occluders)
        ax.imshow(occluded_im, interpolation="none")
        ax.axis('off')

    fig.tight_layout(h_pad=0)
    # plt.savefig('examples.jpg', dpi=150, bbox_inches='tight')
    plt.show()


def load_occluders(pascal_voc_root_path):
    occluders = []
    structuring_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
    
    annotation_paths = list_filepaths(os.path.join(pascal_voc_root_path, 'Annotations'))
    for annotation_path in annotation_paths:
        xml_root = xml.etree.ElementTree.parse(annotation_path).getroot()
        is_segmented = (xml_root.find('segmented').text != '0')

        if not is_segmented:
            continue

        boxes = []
        for i_obj, obj in enumerate(xml_root.findall('object')):
            is_person = (obj.find('name').text == 'person')
            is_difficult = (obj.find('difficult').text != '0')
            is_truncated = (obj.find('truncated').text != '0')
            if not is_person and not is_difficult and not is_truncated:
                bndbox = obj.find('bndbox')
                box = [int(bndbox.find(s).text) for s in ['xmin', 'ymin', 'xmax', 'ymax']]
                boxes.append((i_obj, box))

        if not boxes:
            continue

        im_filename = xml_root.find('filename').text
        seg_filename = im_filename.replace('jpg', 'png')

        im_path = os.path.join(pascal_voc_root_path, 'JPEGImages', im_filename)
        seg_path = os.path.join(pascal_voc_root_path,'SegmentationObject', seg_filename)

        #im = mmcv.imread(im_path)
        #labels = mmcv.imread(seg_path)
        im = np.asarray(PIL.Image.open(im_path))
        labels = np.asarray(PIL.Image.open(seg_path))

        for i_obj, (xmin, ymin, xmax, ymax) in boxes:
            object_mask = (labels[ymin:ymax, xmin:xmax] == i_obj + 1).astype(np.uint8)*255
            object_image = im[ymin:ymax, xmin:xmax]
            if cv2.countNonZero(object_mask) < 500:
                # Ignore small objects
                continue

            # Reduce the opacity of the mask along the border for smoother blending
            eroded = cv2.erode(object_mask, structuring_element)
            object_mask[eroded < object_mask] = 192
            object_with_mask = np.concatenate([object_image, object_mask[..., np.newaxis]], axis=-1)
            
            # Downscale for efficiency
            object_with_mask = resize_by_factor(object_with_mask, 0.5).astype(np.float32)
            #print('!!',mean, std)
            # object_with_mask[:,:,:3] -= mean[::-1]
            # object_with_mask[:,:,:3] /= std[::-1]
            occluders.append(object_with_mask)

    return occluders


def occlude_with_objects(im, occluders, multiplier, joints, cross_prob=0., mask=True):
    """Returns an augmented version of `im`, containing some occluders from the Pascal VOC dataset."""

    if im.shape[0] <=3:
        result = im.copy().transpose(1, 2, 0)
    else:
        result = im.copy()

    for i in range(len(occluders)):
        occluder = occluders[i]
        occluder_size = occluder.shape[1] * occluder.shape[0]
        
        
        # human_size = areas[i]
        # #size_scale = human_size / occluder_size
        # #print('size scale', size_scale, human_size)
        # multiplier = np.random.uniform(0.15*human_size, 0.25*human_size)
        #multiplier = multiplier / occluder_size
 
        # if occluder.shape[0] * multiplier < 5 or occluder.shape[1] * multiplier < 5:
        #     continue
    
        #print(joints.shape)
        ojoint = joints
        
        # large occlusion across head, left arm, right arm, legs
        # if np.random.rand() < cross_prob:
        #     # pick a part
        #     part = np.random.randint(0, 4)
        #     if part == 0: # head
        #         joint = ojoint[[8, 9, 12, 13, 15], :]
        #     elif part == 1: # left
        #         joint = ojoint[[0, 1, 2, 6, 7, 8], :]
        #     elif part == 2: # right
        #         joint = ojoint[[3, 4, 5, 9, 10, 11], :]
        #     elif part == 3: # bottom
        #         joint = ojoint[[0, 1, 2, 3, 4, 5, 6, 11, 14], :]
        #     joint = joint[joint[:,-1] > 0,:]
        #     if joint.shape[0] < 1:
        #         continue
        #     joint = np.mean(joint, axis=0, keepdims=False)[:2] # 2,
        #     multiplier *= 3.
        # else:
        #     joint = ojoint
        #     joint = joint[joint[:,-1] > 0,:]
        #     if joint.shape[0] < 1:
        #         continue
        #     joint = random.choice(joint)[:2] # 2,
        #     x_jitter = np.random.uniform(-1 * occluder.shape[0] / 3., occluder.shape[0] / 3.)
        #     y_jitter = np.random.uniform(-1 * occluder.shape[1] / 3., occluder.shape[1] / 3.)
        #     joint[0] += x_jitter
        #     joint[1] += y_jitter

        occluder = resize_by_factor(occluder, multiplier)

        paste_over(im_src=occluder, im_dst=result, center=joints, mask=mask)

    return result[:,:,:3]







    # im_scale_factor = min(width_height) / 256
    # count = np.random.randint(1, 8)

    # for _ in range(count):
    #     occluder = random.choice(occluders)

    #     random_scale_factor = np.random.uniform(0.2, 1.0)
    #     scale_factor = random_scale_factor * im_scale_factor
    #     occluder = resize_by_factor(occluder, scale_factor)

    #     center = np.random.uniform([0,0], width_height)
    #     paste_over(im_src=occluder, im_dst=result, center=center)

    # return result


def paste_over(im_src, im_dst, center, mask=True):
    """Pastes `im_src` onto `im_dst` at a specified position, with alpha blending, in place.

    Locations outside the bounds of `im_dst` are handled as expected (only a part or none of
    `im_src` becomes visible).

    Args:
        im_src: The RGBA image to be pasted onto `im_dst`. Its size can be arbitrary.
        im_dst: The target image.
        alpha: A float (0.0-1.0) array of the same size as `im_src` controlling the alpha blending
            at each pixel. Large values mean more visibility for `im_src`.
        center: coordinates in `im_dst` where the center of `im_src` should be placed.
    """
    im_src = im_src.copy()

    if im_src.shape[-1] == 3:
        im_src = np.concatenate((im_src, im_src[:,:,[0]]), axis=-1)

    # im_src_a = im_src[:,:,[-1]]
    if mask:
        im_src[:,:,:3] *= 0
        im_src[:,:,:3] += im_src[:,:,[-1]]

        im_src[:,:,:3] = 255 - im_src[:,:,:3]

    width_height_src = np.asarray([im_src.shape[1], im_src.shape[0]])
    width_height_dst = np.asarray([im_dst.shape[1], im_dst.shape[0]])

    center = np.round(center).astype(np.int32)
    raw_start_dst = center - width_height_src // 2
    raw_end_dst = raw_start_dst + width_height_src

    start_dst = np.clip(raw_start_dst, 0, width_height_dst)
    end_dst = np.clip(raw_end_dst, 0, width_height_dst)
    region_dst = im_dst[start_dst[1]:end_dst[1], start_dst[0]:end_dst[0]]

    start_src = start_dst - raw_start_dst
    end_src = width_height_src + (end_dst - raw_end_dst)
    region_src = im_src[start_src[1]:end_src[1], start_src[0]:end_src[0]]
    color_src = region_src[..., 0:3]
    alpha = region_src[..., 3:].astype(np.float32)/255

    im_dst[start_dst[1]:end_dst[1], start_dst[0]:end_dst[0]] = (
            alpha * color_src + (1 - alpha) * region_dst)


def resize_by_factor(im, factor):
    """Returns a copy of `im` resized by `factor`, using bilinear interp for up and area interp
    for downscaling.
    """
    #print(im.shape)
    new_size = tuple(np.round(np.array([im.shape[1], im.shape[0]]) * factor).astype(int))
    interp = cv2.INTER_LINEAR if factor > 1.0 else cv2.INTER_AREA
    # try:
    #     result = cv2.resize(im, new_size, fx=factor, fy=factor, interpolation=interp)
    # except cv2.error:
    #     print(im.shape, new_size, factor)
    #     raise cv2.error
    return cv2.resize(im, new_size, fx=factor, fy=factor, interpolation=interp)#result


def list_filepaths(dirpath):
    names = os.listdir(dirpath)
    paths = [os.path.join(dirpath, name) for name in names]
    return sorted(filter(os.path.isfile, paths))


if __name__=='__main__':
    main()