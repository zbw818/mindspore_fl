import os
import shutil
from random import choice

def count_dir(path):
    num = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            num += 1
    return num

def get_img_list(path):
    img_path_list = []
    label_list = os.listdir(path)
    for i in range(len(label_list)):
        label = label_list[i]
        imgs_path = os.path.join(path,label)
        imgs_name = os.listdir(imgs_path)
        for j in range(len(imgs_name)):
            img_name = imgs_name[j]
            img_path = os.path.join(imgs_path, img_name)
            img_path_list.append(img_path)
    return img_path_list

def data_aug(data_root_path): 
    users = os.listdir(data_root_path)
    tags = ["train", "test"]
    aug_use=[]
    for i in range(len(users)):
        use = users[i]
        for tag in tags:
            data_path = os.path.join(data_root_path, use, tag)
            num_data = count_dir(data_path)
            if num_data < 32:
                aug_use.append(use)
                print("user: " + use + "  "+tag + " data num: " + str(num_data))
                aug_num = 32 - num_data
                img_path_list = get_img_list(data_path)
                for j in range(aug_num):
                    img_path = choice(img_path_list)
                    info = img_path.split(".")
                    aug_img_path = info[0]+"_aug_"+str(j)+".png"
                    shutil.copy(img_path, aug_img_path)
                    print("[aug "+str(j)+"]"+"  ===========copy file: " + img_path + " to -> " + aug_img_path)

    print("aug user number: "+str(len(aug_use)))
    for k in range(len(aug_use)):
        print("aug user name: ", aug_use[k], end = " ")

if __name__=="__main__":
    data_aug("3500_client_img/")
