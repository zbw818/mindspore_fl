
import os
import json

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def partition_json(root_path, new_root_path):
    """
    partition 35 json files to 3500 json file

    Each raw .json file is an object with 3 keys:
    1. 'users', a list of users
    2. 'num_samples', a list of the number of samples for each user
    3. 'user_data', an object with user names as keys and their respective data as values; for each user, data is represented as a list of images, with each image represented as a size-784 integer list (flattened from 28 by 28)

    Each new .json file is an object with 3 keys:
    1. 'user_name', the name of user
    2. 'num_samples', the number of samples for the user
    3. 'user_data', an dict object with 'x' as keys and their respective data as values; with 'y' as keys and their respective label as values;

    Args:
        root_path (str): raw root path of 35 json files
        new_root_path (str): new root path of 3500 json files
    """
    paths = os.listdir(root_path)
    count = 0
    file_num = 0
    for i in paths:
        file_num += 1
        file_path = os.path.join(root_path, i)
        print('======== process ' + str(file_num) + ' file: ' + str(file_path) + '======================')
        with open(file_path, 'r') as load_f:
            load_dict = json.load(load_f)
            users = load_dict['users']
            num_users = len(users)
            num_samples = load_dict['num_samples']
            for j in range(num_users):
                count += 1
                print('---processing user: ' + str(count) + '---')
                cur_out = {'user_name': None, 'num_samples': None, 'user_data': {}}
                cur_user_id = users[j]
                cur_data_num = num_samples[j]
                cur_user_path = os.path.join(new_root_path, cur_user_id + '.json')
                cur_out['user_name'] = cur_user_id
                cur_out['num_samples'] = cur_data_num
                cur_out['user_data'].update(load_dict['user_data'][cur_user_id])
                with open(cur_user_path, 'w') as f:
                    json.dump(cur_out, f)
                #print(count)#insert by zbw 10.22 18:51
    f = os.listdir(new_root_path)
    print(len(f), ' users have been processed!')
# partition train json files
partition_json("./data/train", "./3500_client_json/train")
# partition test json files
partition_json("./data/test", "./3500_client_json/test")

