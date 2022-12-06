
import json
import logging
import os
import pickle
import shutil
import time
import urllib
import zipfile

import cv2
import numpy as np


def clear_bucket(s3_client, bucket_name):
    lists = s3_client.list_objects_v2(Bucket=bucket_name)
    if lists['KeyCount'] > 0:
        objects = lists['Contents']
    else:
        objects = None
    if objects is not None:
        obj_names = []
        for obj in objects:
            file_key = urllib.parse.unquote_plus(obj["Key"], encoding='utf-8')
            obj_names.append(file_key)
        if len(obj_names) >= 1:
            obj_list = [{'Key': obj} for obj in obj_names]
            s3_client.delete_objects(Bucket=bucket_name, Delete={
                                     'Objects': obj_list})
    return True


# 打开cifar-10数据集文件目录
def unpickle(batch):
    with open("/data/cifar-10-batches-py/data_batch_" + str(batch), 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def zip_file(postfix):
    # src_dir为待压缩文件夹地址和名称
    # zip_name为生成zip文件地址和名称
    zip_name = "/local_data/cifar-10-pictures-" + postfix + '.zip'
    src_dir = "/local_data/cifar-10-pictures-" + postfix
    z = zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED)
    for dirpath, dirnames, filenames in os.walk(src_dir):
        fpath = dirpath.replace(src_dir, '')
        fpath = fpath and fpath + os.sep or ''
        for filename in filenames:
            z.write(os.path.join(dirpath, filename), fpath + filename)
    z.close()
    return True



def trigger(num_workers, batch_size, epoches, l_r):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
#    s3_client = boto3.client('s3')

#    clear_bucket(s3_client, "tmp-bucket-chen")
#    clear_bucket(s3_client, "merged-bucket-chen")
#    clear_bucket(s3_client, "traindata-bucket-chen")
    #mkdir files
    files = "/local_data"

    if os.path.exists(files):
        shutil.rmtree(files)
    os.mkdir(files)

    file = "/file"

    if os.path.exists(file):
        shutil.rmtree(file)
    os.mkdir(file)
    
    # setting of scatter_reduce
#    num_workers = 10 #int(event['num_workers'])  # num of total workers
#    batch_size = 128 #int(event['batch_size'])    # batch size
#    epoches = 5 #int(event['epoches'])          # num of epoches
#    l_r = 0.05 #float(event['l_r'])

    label_name = ['airplane', 'automobile', 'brid', 'cat',
                  'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # create folder
    for i in range(num_workers):
        if not os.path.exists('/local_data/cifar-10-pictures-' + str(i)):
            os.mkdir('/local_data/cifar-10-pictures-' + str(i))
            for j in range(10):
                os.mkdir('/local_data/cifar-10-pictures-' +
                        str(i) + '/' + str(label_name[j]))
    if not os.path.exists('/local_data/cifar-10-pictures-test'):
        os.mkdir('/local_data/cifar-10-pictures-test')
        for j in range(10):
            os.mkdir('/local_data/cifar-10-pictures-test/' + str(label_name[j]))

    order = [0] * 10
    num_pics_one = 5000 // num_workers

    # download train-batch and open
    for batch in range(1, 6):
#        s3_client.download_file(
#            "cifar10-batches", "data_batch_"+str(batch), "/tmp/data_batch_"+str(batch))
        data_batch = unpickle(batch)
        # print(data_batch)

        cifar_label = data_batch[b'labels']
        
        cifar_data = data_batch[b'data']
        

        # 把字典的值转成array格式，方便操作
        cifar_label = np.array(cifar_label)
        # print(cifar_label.shape)
        cifar_data = np.array(cifar_data)
        # print(cifar_data.shape)

        for i in range(10000):
            image = cifar_data[i]
            image = image.reshape(-1, 1024)
            r = image[0, :].reshape(32, 32)  # 红色分量
            g = image[1, :].reshape(32, 32)  # 绿色分量
            b = image[2, :].reshape(32, 32)  # 蓝色分量
            img = np.zeros((32, 32, 3))
            #RGB还原成彩色图像
            img[:, :, 0] = r
            img[:, :, 1] = g
            img[:, :, 2] = b
            j = order[cifar_label[i]] // num_pics_one
            
            cv2.imwrite("/local_data/cifar-10-pictures-" + str(j) + "/" + str(label_name[cifar_label[i]]) + "/" +
                        str(label_name[cifar_label[i]]) + "_" + str(order[cifar_label[i]]) + ".jpg", img)
            order[cifar_label[i]] += 1
    
    # zip the folder and upload
    for i in range(num_workers):
        zip_file(str(i))

#        file_name = ".local_data/cifar-10-pictures-" + str(i) + ".zip"
#        key = "cifar-10-pictures-" + str(i) + ".zip"
#        s3_client.upload_file(file_name, "traindata-bucket-chen", key)

    # download test-batch and open
#    s3_client.download_file(
#        "cifar10-batches", "test_batch", "/tmp/test_batch")
    with open("/data/cifar-10-batches-py/test_batch", 'rb') as fo:
        data_batch = pickle.load(fo, encoding='bytes')
    # print(data_batch)

    cifar_label = data_batch[b'labels']
    cifar_data = data_batch[b'data']

    # 把字典的值转成array格式，方便操作
    cifar_label = np.array(cifar_label)
    # print(cifar_label.shape)
    cifar_data = np.array(cifar_data)
    # print(cifar_data.shape)
    
    for i in range(10000):
        image = cifar_data[i]
        image = image.reshape(-1, 1024)
        r = image[0, :].reshape(32, 32)  # 红色分量
        g = image[1, :].reshape(32, 32)  # 绿色分量
        b = image[2, :].reshape(32, 32)  # 蓝色分量
        img = np.zeros((32, 32, 3))
        #RGB还原成彩色图像
        img[:, :, 0] = r
        img[:, :, 1] = g
        img[:, :, 2] = b
        cv2.imwrite("/local_data/cifar-10-pictures-test/" + str(label_name[cifar_label[i]]) + "/" +
                    str(label_name[cifar_label[i]]) + "_" + str(i) + ".jpg", img)

    zip_file("test")
#    s3_client.upload_file("/tmp/cifar-10-pictures-test.zip", "traindata-bucket-chen", "cifar-10-pictures-test.zip")

    # lambda payload
    payload = dict()
    payload['num_workers'] = num_workers
    payload['batch_size'] = batch_size
    payload['epoches'] = epoches
    payload['l_r'] = l_r

    print("end trigger")

