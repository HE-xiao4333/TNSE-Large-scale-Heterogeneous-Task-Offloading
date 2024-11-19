config = {
    "epoch": 3,
    "curPath": 'C:/Users/HX/three/moxing/multitask_multilevel/',
    "Datasets_path": '/data/',
    "Result_path": '/save_info/',
    "Model_path": '/model/',
    "info": 500,
    "Time": 30,
    "Length":6000,
    "Width":50,
    "Speed": 0,
    "Edge_position":[[0,100],[50,8],[0,14],[50,16],[0,25]],
    "vehicle_position":[[5,1],[5,2],[10,4],[15,6],[20,50],
                     [25,80],[30,60],[35,90],[40,110],[45,21],
                        [8,140],[9,8],[27,107],[20,77],[70,25],
                        [4,1],[15,37],[30,6],[20,30],[41,27],
                        [77,77],[37,37],[30,130],[20,200],[47,150]],#user location
    "Scope":2400,
    "Dev_edge": 5, 
    "Dev_dev": 25, # user_number
    "cpu_task_num": 22677,
    "gpu_task_num": 4248,
    "reverse": 2,
    "V":20,
    "B": 5*10**6,
    "noisy": -174,
    "P_tran_min":1, #watt
    "P_tran_max":1.5,
    "f_local_min":7*(10**8),
    "f_local_max":9*(10**8),
    "alpha_1":0.6,
    "kapa":10**(-24),
    "K":3
    }

