
import numpy as np
import os
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as tC
import mindspore.dataset.vision.py_transforms as PV
import mindspore.dataset.transforms.py_transforms as PT
import mindspore

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def count_id(path):
    files = os.listdir(path)
    ids = {}
    for i in files:
        ids[i] = int(i)
    return ids

def create_dataset_from_folder(data_path, img_size, batch_size=32, repeat_size=1, num_parallel_workers=1, shuffle=False):
    """ create dataset for train or test
        Args:
            data_path: Data path
            batch_size: The number of data records in each group
            repeat_size: The number of replicated data records
            num_parallel_workers: The number of parallel workers
        """
    # define dataset
    ids = count_id(data_path)
    mnist_ds = ds.ImageFolderDataset(dataset_dir=data_path, decode=False, class_indexing=ids)
    # define operation parameters
    resize_height, resize_width = img_size[0], img_size[1]  # 32

    transform = [
        PV.Decode(),
        PV.Grayscale(1),
        PV.Resize(size=(resize_height, resize_width)),
        PV.Grayscale(3),
        PV.ToTensor(),
    ]
    compose = PT.Compose(transform)

    # apply map operations on images
    mnist_ds = mnist_ds.map(input_columns="label", operations=tC.TypeCast(mindspore.int32))
    mnist_ds = mnist_ds.map(input_columns="image", operations=compose)

    # apply DatasetOps
    buffer_size = 10000
    if shuffle:
        mnist_ds = mnist_ds.shuffle(buffer_size=buffer_size)  # 10000 as in LeNet train script
    mnist_ds = mnist_ds.batch(batch_size, drop_remainder=True)
    mnist_ds = mnist_ds.repeat(repeat_size)
    return mnist_ds

def img2bin(root_path, root_save):
    """
    transform images to bin files

    Args:
    root_path: the root path of 3500 images files
    root_save: the root path to save bin files

    """

    use_list = []
    train_batch_num = []
    test_batch_num = []
    mkdir(root_save)
    users = os.listdir(root_path)

    blank_data = []

    for user in users:
        use_list.append(user)
        user_path = os.path.join(root_path, user)
        train_test = os.listdir(user_path)
        for tag in train_test:
            data_path = os.path.join(user_path, tag)
            dataset = create_dataset_from_folder(data_path, (32, 32, 1), 32)
            batch_num = 0
            img_list = []
            label_list = []
##            if data_path != 'leaf/data/femnist/3500_client_img/f1840_35/test':
#                continue
            
            for data in dataset.create_dict_iterator():
                batch_x_tensor = data['image']
                batch_y_tensor = data['label']
                trans_img = np.transpose(batch_x_tensor.asnumpy(), [0, 2, 3, 1])
                img_list.append(trans_img)
                label_list.append(batch_y_tensor.asnumpy())
                batch_num += 1

            if batch_num ==0:
                blank_data.append(data_path)
            if tag == "train":
                train_batch_num.append(batch_num)
            elif tag == "test":
                test_batch_num.append(batch_num)

            imgs = np.array(img_list)  # (batch_num, 32,3,32,32)
            labels = np.array(label_list)
            path1 = os.path.join(root_save, user)
            mkdir(path1)
            image_path = os.path.join(path1, user + "_" + "bn_" + str(batch_num) + "_" + tag + "_data.bin")
            label_path = os.path.join(path1, user + "_" + "bn_" + str(batch_num) + "_" + tag + "_label.bin")

            imgs.tofile(image_path)
            labels.tofile(label_path)
            print("user: " + user + " " + tag + "_batch_num: " + str(batch_num))
    print("total " + str(len(use_list)) + " users finished!")

    print("------------------------------------------------------------")
    print("path that test_batch_num is 0: ",blank_data)
    print("the number of the path that test_batch_num is 0: ",len(blank_data))

root_path = "./3500_client_img"
root_save = "./3500_clients_bin"
img2bin(root_path, root_save)

