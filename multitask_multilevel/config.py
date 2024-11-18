config = {
    "epoch": 3,
    "curPath": 'C:/Users/HX/three/moxing/multitask_multilevel/',
    "Datasets_path": '/data/',
    "Result_path": '/save_info/',
    "Model_path": '/model/',
    "info": 500,
    "Time": 30,
    "Length":6000, #路径长度
    "Width":50,#双向车道，一边25m
    "Speed": 0, # 10m/s 车速
    "Edge_position":[[0,100],[50,8],[0,14],[50,16],[0,25]],#edge的位置
    "vehicle_position":[[5,1],[5,2],[10,4],[15,6],[20,50],
                     [25,80],[30,60],[35,90],[40,110],[45,21],
                        [8,140],[9,8],[27,107],[20,77],[70,25],
                        [4,1],[15,37],[30,6],[20,30],[41,27],
                        [77,77],[37,37],[30,130],[20,200],[47,150]],#edge的位置
    "Scope":2400,
    "Dev_edge": 5, # 边缘服务器个数
    "Dev_dev": 25, # 车辆数
    "cpu_task_num": 22677, # 总任务数量
    "gpu_task_num": 4248, # 总任务数量
    "reverse": 2,
    "V":20,
    "B": 5*10**6,
    "noisy": -174, #dbm  公式要换算
    "P_tran_min":1, #watt
    "P_tran_max":1.5,
    "f_local_min":7*(10**8),
    "f_local_max":9*(10**8),
    "alpha_1":0.6,
    "kapa":10**(-24),
    "K":3
    }

