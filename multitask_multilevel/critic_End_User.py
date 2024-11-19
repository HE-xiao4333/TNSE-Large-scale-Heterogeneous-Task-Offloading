import copy
import numpy as np
import importlib
import math
from config import config
import gv_c as gvc
import critic_MES as MES

info = gvc.info
task_queue = gvc.task_queue
wait_tackel_queue = gvc.wait_tackel_queue
wait_offload_queue = gvc.wait_offload_queue
ES_wait_queue = gvc.ES_wait_queue
ES_First_level = gvc.ES_First_level
ES_Second_level = gvc.ES_Second_level
ES_Third_level = gvc.ES_Third_level
wait_size_ES = gvc.wait_size_ES
wait_size_Vehile = gvc.wait_size_Vehile
wait_Energy_Vehile =gvc.wait_Energy_Vehile
f_local = gvc.f_local
def global_update():
    global info,task_queue,sub_task_queue,wait_tackel_queue,wait_offload_queue,ES_wait_queue,ES_First_level,ES_Second_level,ES_Third_level,wait_size_ES,wait_size_Vehile
    importlib.reload(gvc)
    info = gvc.info
    task_queue = gvc.task_queue
    wait_tackel_queue = gvc.wait_tackel_queue
    wait_offload_queue = gvc.wait_offload_queue
    ES_wait_queue = gvc.ES_wait_queue
    ES_First_level = gvc.ES_First_level
    ES_Second_level = gvc.ES_Second_level
    ES_Third_level = gvc.ES_Third_level
    wait_size_ES = gvc.wait_size_ES
    wait_size_Vehile = gvc.wait_size_Vehile



N=config.get('Dev_dev')
M = config.get('Dev_edge')
Band = config.get("B")
f_local_max=config.get('f_local_max')
f_local_min = config.get('f_local_min')
P_tran_min = config.get('P_tran_min')
P_tran_max = config.get('P_tran_max')
alpha_1=config.get('alpha_1')
coverage = config.get('Scope')
edge_position = config.get("Edge_position")
speed = config.get("Speed")
noisy = config.get("noisy")
loss_exponent = 3
light = 3 * 10 ** 8 
Ad = 3  
fc = 915 * 10 ** 6  
K= 5


def find_sub_task(task):
    the_task = info[(info['task'] == task[1]) & (info['task'].index < (task[0] + 8)) & (info['task'].index >= task[0])]
    relate_DAG = dtq.DAGchange(the_task)
    rank = [0, 0, 0, 0, 0, 0, 0, 0]
    for j in range(8):
        dtq.get_rank(rank, relate_DAG, j, the_task)
    task_index = dtq.sort_rank(rank)
    return task_index

def updage_offload_decision(time,x,y,dev):
    wait_task = task_queue[dev][time]
    if not wait_task:
        return
    for z in range(len(x[dev])):
        if x[dev][z][0] == 1:
            info.loc[((info['name'] == wait_task[z][1]) & (info['name'].index == wait_task[z][0])), 'offload'] = 1
            info.loc[((info['name'] == wait_task[z][1]) & (info['name'].index == wait_task[z][0])), 'to'] = y[dev][z][0]
            task_queue[dev][time][z][3] = 1
            task_queue[dev][time][z].append(y[dev][z][1])
        else:
            task_queue[dev][time][z].append(x[dev][z][1])

def str2int(task):
    for A, res in task.iterrows():
        strings = (res.loc['depend']).strip("[]").split(",")  # remove[],and split with ’,‘
        strings = [element for element in strings if element != '']  # remove ''
        strings = [num.replace("'", "") for num in strings]  # “’1‘”-->"1"
        numbers = [int(num) for num in strings]
        return numbers

def judge_task_is_availble(i,dev,t):
    if not wait_tackel_queue[dev][i]:
        return t
    task_id = wait_tackel_queue[dev][i][0][0]
    result = info.loc[(info['sub_task']==wait_tackel_queue[dev][i][0][2]+1)&(info['sub_task'].index<wait_tackel_queue[dev][i][0][0]+8)&(info['sub_task'].index>=wait_tackel_queue[dev][i][0][0])&(info['task'] == wait_tackel_queue[dev][i][0][1])]
    result=str2int(result)
    if not result:
        info.loc[(info['sub_task']==wait_tackel_queue[dev][i][0][2]+1)&(info['sub_task'].index<wait_tackel_queue[dev][i][0][0]+8)&(info['sub_task'].index>=wait_tackel_queue[dev][i][0][0])&(info['task'] == wait_tackel_queue[dev][i][0][1]), 'pre_task'] = 1
        return 0
    else:
        judge=[]
        sub_result_time=[]
        for sub_task_num in result:
            sub_result = info.loc[(info['sub_task']==sub_task_num)&(info['task']==wait_tackel_queue[dev][i][0][1])&(info['sub_task'].index<task_id+8)&(info['sub_task'].index>=task_id)]
            if sub_result['end'].values[0] > t:
                judge.append(0)
            else:
                judge.append(sub_result['complete'].values[0])
            if sub_result['complete'].values[0] == 1:
                sub_result_time.append(sub_result['end'].values[0])
            else:
                sub_result_time.append(0)
        k=np.all(np.array(judge)==1)
        if k:
            info.loc[(info['sub_task']==wait_tackel_queue[dev][i][0][2]+1)&(info['sub_task'].index<wait_tackel_queue[dev][i][0][0]+8)&(info['sub_task'].index>=wait_tackel_queue[dev][i][0][0])&(info['task'] == wait_tackel_queue[dev][i][0][1]), 'pre_task'] = 1
        return max(sub_result_time)

def local_task_tackel(i,dev,f_local,t1,t2,type):
    if not wait_tackel_queue[dev][i][type]:
        return [t1,t2]
    if (t1 - i >= 1) & (t2 - i >= 1):
        return [i+1,i+1]
    result =info.loc[(info['name'] == wait_tackel_queue[dev][i][type][0][1])&(info['name'].index == wait_tackel_queue[dev][i][type][0][0])]
    task_need_cpu = result['cpu_milli'].values[0] - result['complete_size_cpu'].values[0]
    task_need_gpu = result['gpu_milli'].values[0] - result['complete_size_gpu'].values[0]
    if t1 - i== 0:
        time_cpu=1
    else:
        time_cpu=1-(t1 - i)
    f_cpu = f_local[0] * time_cpu
    if t2 - i== 0:
        time_gpu=1
    else:
        time_gpu=1-(t2 - i)
    f_gpu = f_local[1] * time_gpu
    if (result['complete_size_cpu'].values[0] == 0) & (result['complete_size_gpu'].values[0] == 0):
        info.loc[(info['name'] == wait_tackel_queue[dev][i][type][0][1])&(
                         info['name'].index == wait_tackel_queue[dev][i][type][0][0]), 'start'] = min(t1,t2)
    complete_local=[0,0]
    if f_cpu>task_need_cpu:
        info.loc[(info['name'] == wait_tackel_queue[dev][i][type][0][1])&(
                         info['name'].index == wait_tackel_queue[dev][i][type][0][0]), 'complete_size_cpu'] =result['complete_size_cpu'].values[0]+task_need_cpu
        info.loc[(info['name'] == wait_tackel_queue[dev][i][type][0][1])&(
                         info['name'].index == wait_tackel_queue[dev][i][type][0][0]), 'complete_cpu'] = 1
        loss_time = task_need_cpu / f_cpu
        time_cpu = t1 + loss_time
        complete_local[0] = 1
    else:
        info.loc[(info['name'] == wait_tackel_queue[dev][i][type][0][1]) & (
                info['name'].index == wait_tackel_queue[dev][i][type][0][0]), 'complete_size_cpu'] = result['complete_size_cpu'].values[0]+f_cpu
        time_cpu = i+1
    if f_gpu>task_need_gpu:
        info.loc[(info['name'] == wait_tackel_queue[dev][i][type][0][1])&(
                         info['name'].index == wait_tackel_queue[dev][i][type][0][0]), 'complete_size_gpu'] =result['complete_size_gpu'].values[0]+task_need_gpu
        info.loc[(info['name'] == wait_tackel_queue[dev][i][type][0][1])&(
                         info['name'].index == wait_tackel_queue[dev][i][type][0][0]), 'complete_gpu'] = 1
        loss_time = task_need_cpu / f_cpu
        time_gpu = t2 + loss_time
        complete_local[1] = 1
    else:
        info.loc[(info['name'] == wait_tackel_queue[dev][i][type][0][1]) & (
                info['name'].index == wait_tackel_queue[dev][i][type][0][0]), 'complete_size_gpu'] = result['complete_size_gpu'].values[0]+f_gpu
        time_gpu = i+1
    if all(complete_local):
        info.loc[(info['name'] == wait_tackel_queue[dev][i][type][0][1])&(
                        info['name'].index == wait_tackel_queue[dev][i][type][0][0]), 'end'] = max(time_cpu,time_gpu)
        wait_size_Vehile[dev][i] -= info.loc[(info['name'] == wait_tackel_queue[dev][i][type][0][1]) & (
                info['name'].index == wait_tackel_queue[dev][i][type][0][0]), 'cpu_milli']
        wait_tackel_queue[dev][i][type].pop(0)
        return [max(time_cpu,time_gpu),max(time_cpu,time_gpu)]

    return [time_cpu,time_gpu]

def offload_task_tackel(i,dev,f_off,new_t):
    if not wait_offload_queue[dev][i]:
            return new_t
    if new_t - i >=1:
        return i+1
    result = info.loc[(info['name'].index == wait_offload_queue[dev][i][0][0])]
    task_need_offload = result['memory_mib'].values[0] - result['complete_offload'].values[0]
    if new_t - i == 0:
        time_slot = 1
    else:
        time_slot = 1 - (new_t - i)
    f = time_slot *f_off[result['to'].values[0]]
    if result['complete_offload'].values[0] == 0:
        info.loc[wait_offload_queue[dev][i][0][0], 'off_start'] = new_t
    if f > task_need_offload:
        info.loc[wait_offload_queue[dev][i][0][0], 'complete_offload'] = result['memory_mib'].values[0]
        info.loc[wait_offload_queue[dev][i][0][0], 'offload_success'] = 1
        loss_time = task_need_offload / f
        info.loc[wait_offload_queue[dev][i][0][0], 'off_end'] = new_t + loss_time
        #wait_size_ES
        wait_offload_queue[dev][i].pop(0)
        return new_t+loss_time
    else:
        info.loc[wait_offload_queue[dev][i][0][0], 'complete_offload'] =result['complete_offload'].values[0] + f
        return i+1

def End_user_loop(i,loacl_in_time,offload_in_time,f_local,f_offload):
    not_same_tackel =[True] * config.get('Dev_dev')
    while 1:
        The_End_tackel_available = [[True, True, True] for _ in range(config.get('Dev_dev'))]
        The_End_offload_available = [True for _ in range(config.get('Dev_dev'))]
        for dev in range(config.get('Dev_dev')):
            old_loacl_time = loacl_in_time[dev].copy()
            old_offload_time = offload_in_time[dev]
            for type in range(3):
                loacl_in_time[dev][type] = local_task_tackel(i, dev, f_local[type], loacl_in_time[dev][type][0], loacl_in_time[dev][type][1], type)
                if (loacl_in_time[dev] == [i + 1,i + 1]).all() or (
                        loacl_in_time[dev][type] == old_loacl_time[type]).all():
                    The_End_tackel_available[dev][type] = False
            offload_in_time[dev] = offload_task_tackel(i, dev, f_offload[dev], offload_in_time[dev])
            if (offload_in_time[dev] == i+1) or (offload_in_time[dev] == old_offload_time):
                The_End_offload_available[dev] = False
            if (loacl_in_time[dev][0] == old_loacl_time[0]).all() &(loacl_in_time[dev][1] == old_loacl_time[1]).all() &(loacl_in_time[dev][2] == old_loacl_time[2]).all() & (offload_in_time[dev] == old_offload_time).all():
                not_same_tackel[dev] = False
        if (not any(The_End_offload_available)) & ( all(not value for sublist in The_End_tackel_available for value in sublist)):
            break
    return any(not_same_tackel)

def update_ES_queue(i,task_information,optimal_RSU):
    for edge in range(N * M):
        if task_information[edge][4] == -1:
            continue
        info.loc[(info['task'].values[1] == task_information[edge][2]) & (
                info['task'].index >= task_information[edge][1]) & (
                         info['task'].index < task_information[edge][1] + 8) & (
                         info['sub_task'] == task_information[edge][3] + 1), 'to'] = optimal_RSU[edge]
        A = []
        for awaiting in ES_First_level[task_information[edge][0]][i]:
            if (awaiting[0] == task_information[edge][1]) & (awaiting[1] == task_information[edge][2]) & (
                    awaiting[2] == task_information[edge][3]):
                A.append(awaiting)
        ES_First_level[task_information[edge][0]][i].remove(A[0])
        B = []
        C = []
        for indexx in range(len(ES_First_level[optimal_RSU[edge]][i])):
            awaiting = ES_First_level[optimal_RSU[edge]][i][indexx]
            if (awaiting[0] == A[0][0]) & (awaiting[1] == A[0][1]):
                B.append(awaiting)
                C.append(indexx)
        if not B:
            ES_First_level[optimal_RSU[edge]][i].append(A[0])
        else:
            rank = find_sub_task([A[0][0], A[0][1]])
            index = rank.index(A[0][2])
            index_all = []
            for taskk in B:
                index_all.append(rank.index(taskk[2]))
            candidi = 0
            for ia in range(len(index_all)):
                if index_all[ia] > index: 
                    pass
                else:
                    candidi += 1
                    break
            if candidi == 0:
                ES_First_level[optimal_RSU[edge]][i].insert(C[candidi], A[0])
            else:
                ES_First_level[optimal_RSU[edge]][i].insert(C[candidi - 1] + 1, A[0])


def critic(time,x_,y_,index_x,index_y,velocity):
    f_offload = velocity
    for i in range(time,config.get('Time') + 700):
        x, y = [], []
        off_index = 0
        for dev in range(config.get('Dev_dev')):
            x.append([])
            y.append([])
            for task in x_[dev]:
                x[dev].append(task[index_x])
                if x[dev][-1][0] == 1:
                    y[dev].append(y_[index_x][off_index][index_y])
                    off_index += 1
                else:
                    y[dev].append([])
        if i == time:
            for dev in range(config.get('Dev_dev')):
                updage_offload_decision(i, x, y, dev)
        for dev in range(config.get('Dev_dev')):
            if len(task_queue[dev][i]) == 0:
                continue
            task_queue_py = task_queue[dev][i].copy()
            for sub_task in task_queue_py:
                result = info[((info['time'] == i) & (info['time'].index == sub_task[0]))]
                if sub_task[3] == 1:
                    wait_offload_queue[dev][i].append(sub_task[:3])
                    wait_size_ES[int(result['to'].values[0])][i] += result['cpu_milli'].values[0]
                    if result['gpu_spec'].values[0] == -1:
                        gpu_spec = sub_task[4]
                        info.loc[(info['name'] == sub_task[1]) & (info['name'].index == sub_task[0]), 'gpu_spec'] = gpu_spec
                else:
                    if result['gpu_spec'].values[0] == -1:
                        gpu_spec = sub_task[4]
                        info.loc[
                            (info['name'] == sub_task[1]) & (info['name'].index == sub_task[0]), 'gpu_spec'] = gpu_spec
                    else:
                        gpu_spec = int(result['gpu_spec'].values[0])
                    wait_tackel_queue[dev][i][gpu_spec].append(sub_task[:3])
                    wait_size_Vehile[dev][i] += result['cpu_milli'].values[0]
                task_queue[dev][i].pop(0)
        loacl_in_time, offload_in_time = np.zeros((config.get('Dev_dev'), 3, 2)), np.zeros(config.get('Dev_dev'))
        for dev in range(config.get('Dev_dev')):
            offload_in_time[dev] = offload_task_tackel(i, dev, f_offload[dev], i)
            for type in range(3):
                loacl_in_time[dev][type] = local_task_tackel(i, dev, f_local[type], i, i, type)
        End_loop_judge = End_user_loop(i, loacl_in_time, offload_in_time, f_local, f_offload)
        MES.MES_task_tackel(i)
        update_tackel_queue(i)
        result = info[(info['time'].values == time) & ((info['complete_cpu'].values == 0) | (info['complete_gpu'].values == 0))]
        if result.empty:
            break
    return 0

def update_tackel_queue(i):
    if i == config.get('Time') + 700:
        return
    for dev in range(config.get('Dev_dev')):
        for type in range(3):
            while wait_tackel_queue[dev][i][type]:
                if not wait_tackel_queue[dev][i][type]:
                    break
                old_wait_task = wait_tackel_queue[dev][i][type].pop(0)
                wait_tackel_queue[dev][i + 1][type].append(old_wait_task)
        while wait_offload_queue[dev][i]:
            if not wait_offload_queue[dev][i]:
                break
            old_offload_task = wait_offload_queue[dev][i].pop(0)
            wait_offload_queue[dev][i+1].append(old_offload_task)
        wait_size_Vehile[dev][i+1]=wait_size_Vehile[dev][i]+wait_size_Vehile[dev][i+1]
