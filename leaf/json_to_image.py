
import os
import json
import numpy as np
from PIL import Image

name_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
             'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
             'V', 'W', 'X', 'Y', 'Z',
             'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
             'v', 'w', 'x', 'y', 'z'
             ]

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def json_2_numpy(img_size, file_path):
    """
    read json file to numpy
    Args:
        img_size (list): contain three elements: the height, width, channel of image
        file_path (str): root path of 3500 json files
    return:
        image_numpy (numpy)
        label_numpy (numpy)
    """
    # open json file
    with open(file_path, 'r') as load_f_train:
        load_dict = json.load(load_f_train)
        num_samples = load_dict['num_samples']
        x = load_dict['user_data']['x']
        y = load_dict['user_data']['y']
        size = (num_samples, img_size[0], img_size[1], img_size[2])
        image_numpy = np.array(x, dtype=np.float32).reshape(size)  # mindspore doesn't support float64 and int64
        label_numpy = np.array(y, dtype=np.int32)
    return image_numpy, label_numpy

def json_2_img(json_path, save_path):
    """
    transform single json file to images

    Args:
        json_path (str): the path json file
        save_path (str): the root path to save images

    """
    data, label = json_2_numpy([28, 28, 1], json_path)
    for i in range(data.shape[0]):
        img = data[i] * 255  # PIL don't support the 0/1 image ,need convert to 0~255 image
        im = Image.fromarray(np.squeeze(img))
        im = im.convert('L')
        img_name = str(label[i]) + '_' + name_list[label[i]] + '_' + str(i) + '.png'
        path1 = os.path.join(save_path, str(label[i]))
        mkdir(path1)
        img_path = os.path.join(path1, img_name)
        im.save(img_path)
        print('-----', i, '-----')

def all_json_2_img(root_path, save_root_path):
    """
    transform json files to images
    Args:
        json_path (str): the root path of 3500 json files
        save_path (str): the root path to save images
    """
    usage = ['train', 'test']
    for i in range(2):
        x = usage[i]
        files_path = os.path.join(root_path, x)
        files = os.listdir(files_path)

        for name in files:
            user_name = name.split('.')[0]
            json_path = os.path.join(files_path, name)
            save_path1 = os.path.join(save_root_path, user_name)
            mkdir(save_path1)
            save_path = os.path.join(save_path1, x)
            mkdir(save_path)
            print('=============================' + name + '=======================')
            json_2_img(json_path, save_path)

all_json_2_img("./3500_client_json/", "./3500_client_img/")

