
import os
import argparse
import subprocess
import random

parser = argparse.ArgumentParser(description="Run TestClient.java case")
parser.add_argument("--jarPath", type=str, default="/mnt/data/mindspore-lite-1.5.0-linux-x64/runtime/lib/mindspore-lite-java-flclient.jar")  # must be absolute path
parser.add_argument("--train_dataset", type=str, default="/mnt/data/leaf/data/femnist/3500_clients_bin/")   # must be absolute path
parser.add_argument("--test_dataset", type=str, default="null")   # must be absolute path
parser.add_argument("--vocal_file", type=str, default="null")   # must be absolute path
parser.add_argument("--ids_file", type=str, default="null")   # must be absolute path
parser.add_argument("--flName", type=str, default="lenet")
parser.add_argument("--train_model_path", type=str, default="/mnt/data/ms/lenet/")    # must be absolute path of .ms files
parser.add_argument("--infer_model_path", type=str, default="/mnt/data/ms/lenet/")    # must be absolute path of .ms files
parser.add_argument("--use_ssl", type=str, default="false")
parser.add_argument("--domain_name", type=str, default="http://10.21.5.236:6668")
parser.add_argument("--server_num", type=int, default=5)
parser.add_argument("--client_num", type=int, default=5)
parser.add_argument("--if_use_elb", type=str, default="false")
parser.add_argument("--cert_path", type=str, default="null")
parser.add_argument("--task", type=str, default="train")

args, _ = parser.parse_known_args()
jarPath = args.jarPath
train_dataset = args.train_dataset
test_dataset = args.test_dataset
vocal_file = args.vocal_file
ids_file = args.ids_file
flName = args.flName
train_model_path = args.train_model_path
infer_model_path = args.infer_model_path
use_ssl = args.use_ssl
domain_name = args.domain_name
server_num = args.server_num
client_num = args.client_num
if_use_elb = args.if_use_elb
cert_path = args.cert_path
task = args.task

users = os.listdir(train_dataset)

def get_client_data_path(data_root_path, user):
    use_path = os.path.join(data_root_path, user)
    bin_file_paths = os.listdir(use_path)

    train_data_path = ""
    train_label_path = ""
    train_batch_num = ""

    test_data_path = ""
    test_label_path = ""
    test_batch_num = ""

    for file in bin_file_paths:
        info = file.split(".")[0].split("_")
        if info[4] == "train" and info[5] == "data":
            train_data_path = os.path.join(use_path, file)
            train_batch_num = info[3]
        elif info[4] == "train" and info[5] == "label":
            train_label_path = os.path.join(use_path, file)
        elif info[4] == "test" and info[5] == "data":
            test_data_path = os.path.join(use_path, file)
            test_batch_num = info[3]
        elif info[4] == "test" and info[5] == "label":
            test_label_path = os.path.join(use_path, file)
    train_path = train_data_path + "," + train_label_path
    test_path = test_data_path + "," + test_label_path
    return train_path, test_path, train_batch_num, test_batch_num

for i in range(client_num):
    user = users[i]
    train_path, test_path = "", ""
    train_path, test_path, _, _= get_client_data_path(train_dataset, user)
    print("===========================")
    print("process id: ", i)
    print("train path: ", train_path)
    print("test path: ", test_path)

    cmd_client = "execute_path=$(pwd) && self_path=$(dirname \"${script_self}\") && "
    cmd_client += "rm -rf ${execute_path}/client_" + str(i) + "/ &&"
    cmd_client += "mkdir ${execute_path}/client_" + str(i) + "/ &&"
    cmd_client += "cd ${execute_path}/client_" + str(i) + "/ || exit &&"

    cmd_client += "java -jar "
    cmd_client += jarPath + " "
    cmd_client += train_path + " "
    cmd_client += vocal_file + " "
    cmd_client += ids_file + " "
    cmd_client += test_path + " "
    cmd_client += flName + " "
    cmd_client += train_model_path + "lenet_train" + str(i) + ".ms" + " "
    print("model path: ", train_model_path + "lenet_train" + str(i) + ".ms" + " ")
    cmd_client += infer_model_path + "lenet_train" + str(i) + ".ms" + " "
    print("model path: ", infer_model_path + "lenet_train" + str(i) + ".ms" + " ")
    cmd_client += use_ssl + " "
    cmd_client += domain_name + " "
    cmd_client += if_use_elb + " "
    cmd_client += str(server_num) + " "
    cmd_client += cert_path + " "
    cmd_client += task + " "
    cmd_client += " > client" + ".log 2>&1 &"
    subprocess.call(['bash', '-c', cmd_client])

