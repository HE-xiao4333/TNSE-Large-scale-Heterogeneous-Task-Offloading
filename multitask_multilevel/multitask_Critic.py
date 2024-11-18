import copy
import importlib
import math

import numpy as np

import gv_c as gvc
from config import config
import critic_End_User as ceu
import critic_MES as cm
N =config.get('Dev_dev')
M =config.get('Dev_edge')
V=config.get('V')
def critic_en(time,x,y,velocity):
    optimal_x,optimal_y=[],[]
    local_singl_value=[]
    local_value=[]
    mes_value=[]
    all_value=[]
    for i in range(2):
        for j in range(2):
            ceu.global_update()
            cm.global_update()
            info = gvc.info
            task_queue = gvc.task_queue
            wait_tackel_queue = gvc.wait_tackel_queue
            wait_offload_queue = gvc.wait_offload_queue
            ES_wait_queue = gvc.ES_wait_queue
            wait_size_ES = gvc.wait_size_ES
            wait_size_Vehile = gvc.wait_size_Vehile
            ceu.critic(time,x,y,i,j,velocity)
            result_local = info[(info['time'].values == time) & (info['offload'].values == 0)]
            result_mes = info[(info['time'].values == time) & (info['offload'].values == 1)]
            use_time = np.zeros(N)
            local_queue_update=np.zeros(N)
            singl_value=[]
            for dev in range(N):
                update_size=0
                result = result_local[result_local['from'] == dev]
                task_number=0
                for index, row in result.iterrows():
                    use_time[dev] += row['cpu_milli'] / (row['end'] - time)
                    update_size += row['cpu_milli']
                    task_number+=1
                if wait_size_Vehile[dev][time] == 0:
                    wait_size_Vehile[dev][time] = 1
                local_queue_update[dev] += update_size*wait_size_Vehile[dev][time]
                if task_number == 0:
                    task_number = 1
                singl_value.append((V*use_time[dev]-local_queue_update[dev]))
            local_singl_value.append(singl_value)
            local_value.append(sum(local_singl_value[-1]))
            use_time = 0
            MES_queue_update = 0
            for index, row in result_mes.iterrows():
                use_time += row['cpu_milli'] / (row['end'] - time)
            for dev in range(M):
                result = result_mes[result_mes['to'] == dev]
                update_size = 0 if result.empty else result['cpu_milli'].sum()
                if wait_size_ES[dev][time] == 0:
                    wait_size_ES[dev][time] = 1
                MES_queue_update += update_size * wait_size_ES[dev][time]
            mes_value.append(V*use_time -MES_queue_update)
            all_value.append(mes_value[-1]+local_value[-1])
    #select the max value? or select
    index_single=[]
    max_value, index = (0*local_value[0]+1*all_value[0]), 0
    for dev in range(4):
        if (0*local_value[dev]+1*all_value[dev]) > max_value:
            max_value = 0*local_value[dev]+1*all_value[dev]
            index = dev
    index_fin = index % 2
    return index_fin,index_fin,local_singl_value[index_fin],0*local_value[index]+1*all_value[index]




