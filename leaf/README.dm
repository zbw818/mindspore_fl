This file is a script for preprocessing data sets in the leaf/data/femnist path


split_json.py:
        Divide 35 JSON files into 3500 JSON files (each JSON file represents a user). The training and test folders are processed separately
        
json_to_img.py:
        Convert the JSON file to an image file.

img_to_bin.py:
        Convert the image data set to a bin file format available to the federated learning framework.
        

solve_batchnum.py：
         Resolve a large number of test_batch_num=0 problems with the img2bin format conversion process.
