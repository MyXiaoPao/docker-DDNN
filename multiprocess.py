import json
import logging
import os
import pickle
import random
import time
import urllib
import zipfile
from multiprocessing import Process
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from alexnet_cifar10 import Net, load_data

# 定义一个函数，供主进程调用


def action(name, *add):
    print(name)
    for arc in add:
        print("%s --当前进程%d" % (arc, os.getpid()))

# 创建一个进程类Pros
class Pros(Process):
    def __init__(self, name, worker_index, pro_num, pattern, agg_mod):
        super().__init__()
        self.name = name
        self.worker_index = worker_index
        self.pro_num = pro_num
        self.pattern = pattern
        self.agg_mod = agg_mod
        print(worker_index)

    def run(self):
        
        setup_seed(1234)

        # mc = memcache.Client(["127.0.0.1:11211"], debug=True)   #debug为调试参数
    
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        # training setting
        pattern_k = self.pattern  # num pf aggregator
        worker_index = self.worker_index #int(event['worker_index'])  # index of worker
        num_workers = self.pro_num #int(event['num_workers'])  # total num of workers
        # memory = int(event['memory'])  # memory of function
        batch_size = 128 #int(event['batch_size'])
        epoches = 2 #int(event['epoches'])
        agg_mod = self.agg_mod

#        output_file = open('/tmp/file.txt', 'w', encoding='utf-8')
#        output_file.write(f"Worker Index: {worker_index}\n")

        if agg_mod != 'epoch':
            splits = agg_mod.split("_")
            num_batches = int(splits[1])
        else:
            num_batches = 'epoch'

        # download file from S3
        #file_name = "cifar-10-pictures-" + str(worker_index)
        #s3_client.download_file("scatter-traindata-bucket", file_name,
        #                        '/tmp/cifar-10-pictures.zip')
        #unzip_file()
        train_name = "/local_data/cifar-10-pictures-" + str(worker_index)
        test_name = "/local_data/cifar-10-pictures-test"

        net = Net()
        # trainloader, testloader = load_data()
        trainloader = load_data(batch_size, train_name)
        testloader = load_data(batch_size, test_name)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

        num_pics = 50000 // num_workers
        num_mini_batches = num_pics // batch_size
        num_output_m_b = num_mini_batches // 5  # 输出5次

        t_start = time.time()
        count_batches = 0

        logging.info(f"this is worker {worker_index}")
        logging.info(f"batch size is {batch_size}")
        for epoch in range(epoches):  # loop over the dataset multiple times
            #output_file.write(f"\n#Epoch {epoch+1}\n")

            train_start = time.time()
            running_loss = 0.0
            
            for i, data in enumerate(trainloader, 0):
                
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                # if i % num_output_m_b == num_output_m_b - 1: 
                    # output_loss = running_loss / num_output_m_b
                    # logging.info(f'[{i+1}] : {output_loss:.3f}')
                    # running_loss = 0.0

                count_batches += 1
                if agg_mod != 'epoch' and count_batches % num_batches == 0:
                    # scatter_reduce every 'num_batches' batches
                    weights = [param.data.numpy() for param in net.parameters()]
                    merged_weights = scatter_reduce(
                        weights, epoch+1, num_workers, worker_index, pattern_k, i+1)
                    for layer_index, param in enumerate(net.parameters()):
                        param.data = torch.from_numpy(merged_weights[layer_index])
            logging.info(f"total batch is {i+1}")
            
            if agg_mod == 'epoch':
                # scatter_reduce at the end of epoch
                weights = [param.data.numpy() for param in net.parameters()]
                merged_weights = scatter_reduce(weights, epoch, num_workers, worker_index, pattern_k)
                for layer_index, param in enumerate(net.parameters()):
                    param.data = torch.from_numpy(merged_weights[layer_index])

        logging.info(f'Finished Training')
        end_time = time.time() - t_start
        print("process_time", end_time)
#        output_file.write(f"\nTotal Time: {training_time:.2f}\n")
#        output_file.write('Finished Training')
#        output_file.close()

#        path = str(num_workers) + '_' + str(batch_size) + '_' + str(epoches) + '/'
#        file_name = str(worker_index) + '.txt'
#        s3_client.upload_file(
#            '/tmp/file.txt', 'scatter-output-bucket', path + file_name)

        PATH = '/local_data/cifar_net.pth'
        torch.save(net.state_dict(), PATH)

        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                # calculate outputs by running images through the network
                outputs = net(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                accuracy = 100 * correct / total

        # print(f'Accuracy of the network on the 10000 test images: {accuracy:.2f} %')
        logging.info(f'Accuracy of the network on the 10000 test images: {accuracy:.2f} %')
        logging.info(f'Finished Testing')
        return {"result": "succeed!"}

    
def scatter_reduce(weights, epoch, num_workers, worker_index, pattern_k, batch = 0):
    vector = weights[0].reshape(1,-1)
    for i in range(1,len(weights)):
        vector = np.append(vector, weights[i].reshape(1,-1))
        # vector is supposed to be a 1-d numpy array
    num_all_values = vector.size
    num_values_per_agg = num_all_values // pattern_k
    residue = num_all_values

    logging.info(f'create files')
    if not os.path.exists('/file/tmp'):
        os.mkdir('/file/tmp')
    if not os.path.exists('/file/merged'):
        os.mkdir('/file/merged')

    # write partitioned vector to the shared memory, except the chunk charged by myself
    logging.info(f'create tmp key file')
    for i in range(pattern_k):
        if i != worker_index:
            offset = (num_values_per_agg * i) + min(residue, i)
            length = num_values_per_agg + (1 if i < residue else 0)
            # indicating the chunk number and which worker it comes from
            # format of key in tmp-bucket: chunkID_epoch_batch_workerID
            key = "{}_{}_{}_{}".format(i, epoch, batch, worker_index)
            tmp_file = '/file/tmp/' + key
            #os.mkdir(tmp_file)
            file = open(tmp_file, 'wb')
            pickle.dump(vector[offset: offset + length], file)
            file.close()

    merged_value = dict()
    
    logging.info(f'Start aggregator')
    # aggregator only
    if worker_index < pattern_k:
        my_offset = (num_values_per_agg * worker_index) + \
            min(residue, worker_index)
        my_length = num_values_per_agg + (1 if worker_index < residue else 0)
        my_chunk = vector[my_offset: my_offset + my_length]
        
        # read and aggregate the corresponding chunk
        tmp_prefix = "{}_{}_{}_".format(worker_index, epoch, batch)
        for index in range(num_workers):
            if index != worker_index:
                cur_file = '/file/tmp/' + tmp_prefix + str(index)
                while not os.path.exists(cur_file):
                    time.sleep(0.5)
                flag = True
                while flag:
                    try:
                        cur_data = pickle.load(open(cur_file, 'rb'))
                        flag = False
                    except EOFError:
                        time.sleep(0.5)
                my_chunk += cur_data
                os.remove(cur_file)
    
        # average weights
        my_chunk /= float(num_workers)
        # write the aggregated chunk back
        # key format in merged_bucket: epoch_batch_chunkID
        key = "{}_{}_{}".format(epoch, batch, worker_index)
        merged_dir = '/file/merged/epoch' + str(epoch)
        try:
            os.mkdir(merged_dir)
        except FileExistsError:
            time.sleep(0.1)
        merged_file = merged_dir + '/' + key
        file = open(merged_file, 'wb')
        pickle.dump(my_chunk, file)
        file.close()
        merged_value[worker_index] = my_chunk

    logging.info(f'read other aggregated chunks')
    # read other aggregated chunks
    merged_prefix = "{}_{}_".format(epoch, batch)
    for index in range(pattern_k):
        if index != worker_index:
            cur_file = '/file/merged/epoch' + str(epoch) + '/' + merged_prefix + str(index)
            while not os.path.exists(cur_file):
                time.sleep(0.5)
            flag = True
            while flag:
                try:
                    cur_data = pickle.load(open(cur_file, 'rb'))
                    flag = False
                except EOFError:
                    time.sleep(0.5)
            merged_value[index] = cur_data

    logging.info(f'new_vector')
    # reconstruct the whole vector
    new_vector = merged_value[0]
    for k in range(1, pattern_k):
        new_vector = np.concatenate((new_vector, merged_value[k]))
    
    result = dict()
    index = 0
    for k in range(len(weights)):
        lens = weights[k].size
        tmp_arr = new_vector[index:index + lens].reshape(weights[k].shape)
        result[k] = tmp_arr
        index += lens
        
    return result  



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def unzip_file():
    # dir_name为待解压文件夹名称
    # zip_name为带解压zip文件地址和名称
    # dst_dir为生成的文件夹
    zip_name = "/tmp/cifar-10-pictures.zip"
    dst_dir = "/tmp/cifar-10-pictures"
    fz = zipfile.ZipFile(zip_name, 'r')
    for file in fz.namelist():
        fz.extract(file, dst_dir)
    fz.close()
    return True







