
import shutil
import os

def copy_file(raw_path,new_path,copy_num):
    # Copy the specified number of files from the raw path to the new path
    for i in range(copy_num):
        file_name = "lenet_train" + str(i) + ".ms"
        new_file_path = os.path.join(new_path, file_name)
        shutil.copy(raw_path ,new_file_path)
        print('====== copying ',i, ' file ======')
    print("the number of copy .ms files: ", len(os.listdir(new_path)))

if __name__ == "__main__":
    raw_path = "lenet_train.ms"
    new_path = "ms/lenet"
    num = 5
    copy_file(raw_path, new_path, num)

