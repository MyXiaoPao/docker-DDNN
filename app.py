import os
import time

from multiprocess import Pros, action
from trigger_local import trigger

if __name__ == "__main__":
    # def handler(event, context):
    print("当前进程ID：",os.getpid())
    pro_num = 2 #int(os.environ['process_num'])
    agg_mod = "epoch"
    pattern_k = pro_num
    batch_size = 128
    epoches = 2
    l_r = 0.05

    trigger(pro_num, batch_size, epoches, l_r)
    pro_list = []
    # 创建n个子进程
    print("process start：",os.getpid())
    time_start = time.time()
    for i in range(pro_num):
        num = "p" + str(i+1)
        num = Pros(f"{num}", i, pro_num, pattern_k, agg_mod)
        num.cpu_affinity([0, 1, 2])
        num.start()     # start方法会自动调用进程类中的run方法
        pro_list.append(num)
    
    # 子进程阻塞，主进程会等待所有子进程结束再结束
    for i in pro_list:
        i.join()
    end_time = time.time() - time_start
    print("process_time", end_time)
 
    print('主进程结束')

#    return {"result":"succeed!"}
