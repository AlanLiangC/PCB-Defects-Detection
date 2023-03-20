# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import math
import os
import random
import shutil
import sys
import xml.etree.cElementTree as ET
import numpy as np


ROOT_PATH = '.'
DATASET_NAME = 'VOC_PCB'

NAME_LABEL_MAP = {
        'back_ground': 0,
        'missing_hole': 1,
        'mouse_bite': 2,
        'open_circuit': 3,
        'short': 4,
        'spur': 5,
        'spurious_copper': 6
    }

def mkdir(path):
  if not os.path.exists(path):
    os.makedirs(path)

def read_xml_gtbox_and_label(xml_path):
    """
    :param xml_path: the path of voc xml
    :return: a list contains gtboxes and labels, shape is [num_of_gtboxes, 9],
           and has [x1, y1, x2, y2, x3, y3, x4, y4, label] in a per row
    """

    tree = ET.parse(xml_path)
    root = tree.getroot()
    img_width = None
    img_height = None
    box_list = []
    for child_of_root in root:
        # if child_of_root.tag == 'filename':
        #     assert child_of_root.text == xml_path.split('/')[-1].split('.')[0] \
        #                                  + FLAGS.img_format, 'xml_name and img_name cannot match'

        if child_of_root.tag == 'size':
            for child_item in child_of_root:
                if child_item.tag == 'width':
                    img_width = int(child_item.text)
                if child_item.tag == 'height':
                    img_height = int(child_item.text)

        if child_of_root.tag == 'object':
            label = None
            for child_item in child_of_root:
                tmp_box = []
                if child_item.tag == 'name':
                    label = NAME_LABEL_MAP[child_item.text]
                if child_item.tag == 'bndbox':
                    tmp_box.append(label)
                    for node in child_item:
                        tmp_box.append(int(node.text))
                    assert label is not None, 'label is none, error'
                    
                    box_list.append(tmp_box)

    gtbox_label = np.array(box_list, dtype=np.float32)

    xmin, ymin, xmax, ymax = gtbox_label[:,1], gtbox_label[:,2], gtbox_label[:,3], gtbox_label[:,4]

    center_x = (xmin + xmax) / (2*img_width)
    center_y = (ymin + ymax) / (2*img_height)

    box_weight = (xmax - xmin)/ img_width
    box_height = (ymax - ymin)/ img_height

    gtbox_label[:,1] = center_x
    gtbox_label[:,2] = center_y
    gtbox_label[:,3] = box_weight
    gtbox_label[:,4] = box_height

    return gtbox_label


divide_rate = 0.8

image_path = os.path.join(ROOT_PATH, '{}/JPEGImages'.format(DATASET_NAME))
xml_path = os.path.join(ROOT_PATH, '{}/Annotations'.format(DATASET_NAME))

image_list = os.listdir(image_path)

image_name = [n.split('.')[0] for n in image_list]

random.shuffle(image_name)

train_image = image_name[:int(math.ceil(len(image_name)) * divide_rate)]
test_image = image_name[int(math.ceil(len(image_name)) * divide_rate):]

image_output_train = os.path.join(
    ROOT_PATH, '{}_train/images'.format(DATASET_NAME))
mkdir(image_output_train)
image_output_val = os.path.join(
    ROOT_PATH, '{}_val/images'.format(DATASET_NAME))
mkdir(image_output_val)

label_output_train = os.path.join(ROOT_PATH, '{}_train/labels'.format(DATASET_NAME))
mkdir(label_output_train)
label_output_val = os.path.join(ROOT_PATH, '{}_val/labels'.format(DATASET_NAME))
mkdir(label_output_val)



count = 0
for i in train_image:
  shutil.copy(os.path.join(image_path, i + '.jpg'), image_output_train)
  if os.path.exists(os.path.join(xml_path, i + '.xml')):
    xml_path_sub = os.path.join(xml_path, i + '.xml')
    gtbox_label = read_xml_gtbox_and_label(xml_path_sub)
    np.savetxt(os.path.join(label_output_train, i + '.txt'), gtbox_label)

  if count % 1000 == 0:
    print("process step {}".format(count))
  count += 1

for i in test_image:
  shutil.copy(os.path.join(image_path, i + '.jpg'), image_output_val)
  if os.path.exists(os.path.join(xml_path, i + '.xml')):
    xml_path_sub = os.path.join(xml_path, i + '.xml')
    gtbox_label = read_xml_gtbox_and_label(xml_path_sub)
    np.savetxt(os.path.join(label_output_val, i + '.txt'), gtbox_label)
  if count % 1000 == 0:
    print("process step {}".format(count))
  count += 1
