import copy
import importlib
import random
import numpy as np
import pandas as pd
from tqdm import trange
import math
import torch
from config import config
from memory import multiagent
from memory import unit_all
import gloable_variation as gv
import Multitask_MES as MES
import multitask_Critic as MCT

curPath = config.get("curPath")
Datasets_path = config.get("Datasets_path")
epoch_all = config.get("epoch")
Model_path = config.get("Model_path")
Result_path = config.get("Result_path")
alpha = config.get("alpha")
save_path = config.get("Datasets_path")


def append_to_task_queue(time,gen_task):
    if len(gen_task) == 0:
        return
    for id in gen_task:
        task_name=info.loc[id]['name']
        from_=int(info.loc[id]['from'])
        reslut = info[(info['name'].index == id)]
        task_size = reslut['memory_mib'].values[0]
        task=[id,task_name,task_size,0]
        task_queue[from_][time].append(task)

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

def the_fact_user(time,user):
    result = info[(info['time'] == time) & (info['from'] == user)]
    the_number = [result.shape[0]/5]
    the_size = [0] if result.empty else [result['cpu_milli'].sum()/50000]
    the_GPU,the_proity=[0,0,0,0],[0,0,0]
    for index,row in result.iterrows():
        the_GPU[row['gpu_spec']+1] += 1/5
        the_proity[row['qos']-1] +=1/5
    return the_number+the_size+the_GPU+the_proity




def user_predict(time,user):
    user_task = np.zeros((5, 1+1+4+3))
    for t in range(time-4,time+1):
        if time < 3:
            if t < 0:
                continue
        result = info[(info['time'] == t) & (info['from'] == user)]
        user_task[t + 4 - time][0] = (result.shape[0])/5
        user_task[t + 4 - time][1] = (result['cpu_milli'].sum())/50000
        for gpu_spec in range(4):
            user_task[t + 4 - time][2 + gpu_spec] = ((result['gpu_spec'] == (gpu_spec - 1)).sum())/5
        for qos in range(3):
            user_task[t + 4 - time][6 + qos] = ((result['qos'] == (qos + 1)).sum())/5
    return user_task



def get_user_para(time,user,task_number,pre_next):
    #the_first
    ones_time = [time/(config.get('Time')+1)]
    #the second
    ones_task = []
    task_dicision = task_queue[user][time][task_number]
    result = info[((info['time'] == time) & (info['time'].index == task_dicision[0]))]
    if result['cpu_milli'].values[0] >10000:
        ones_task.append(1)
    else:
        ones_task.append(result['cpu_milli'].values[0]/10000)
    if result['gpu_milli'].values[0] >10:
        ones_task.append(1)
    else:
        ones_task.append(result['gpu_milli'].values[0]/10)
    ones_task.append((result['qos'].values[0]) / 3)
    ones_task.append((result['gpu_spec'].values[0]+1)/4)
    #the third
    ones_other=[]
    result_others = list(filter(lambda x: x != task_dicision, task_queue[user][time]))
    number_other_task,size_cpu,size_gpu,priorty_lx,GPU_lx=0,0,0,[0,0,0],[0,0,0,0]
    for task_other in result_others:
        result = info[((info['time'] == time) & (info['time'].index == task_other[0]))]
        number_other_task +=1
        size_cpu += result['cpu_milli'].values[0]
        size_gpu += result['gpu_milli'].values[0]
        priorty_lx[result['qos'].values[0]-1] +=1
        GPU_lx[(result['gpu_spec'].values[0]+1)] += 1
    size_cpu = size_cpu/50000
    size_gpu = size_gpu/50
    priorty_lx = [x/5 for x in priorty_lx]
    GPU_lx = [x / 5 for x in GPU_lx]
    ones_other.append(number_other_task/100)
    ones_other.append(size_cpu)
    ones_other.append(size_gpu)
    ones_other = ones_other+priorty_lx+GPU_lx
    #the forth:user's ability
    ones_user = []
    user_size,user_number_task,user_size_max=0,[0,0,0],[0,0,0]
    for form in range(3):
        for task_other in wait_tackel_queue[user][time][form]:
            result = info[(info['time'].index == task_other[0])]
            user_size += 1
            user_number_task[form] +=1
            if (result['cpu_milli'].values[0] > 50000) or (result['gpu_milli'].values[0] > 50):
                user_size_max[form] += 1
    ones_user = [user_size/50] + [x / 50 for x in user_number_task] + [x / 50 for x in user_size_max]
    #the fifth:MES's ability
    pre_next =pre_next.tolist()
    output = Guiyi.get_output_user(ones_other+ones_user+pre_next[0])
    return ones_time+ones_task+output.tolist()

def get_mes_para(time,number,offload_task_list):
    # the_first
    ones_time = [time / (config.get('Time') + 1)]
    # the second
    ones_task = []
    task_dicision = offload_task_list[number]
    result = info[((info['time'] == time) & (info['time'].index == task_dicision[0]))]
    if result['cpu_milli'].values[0] > 10000:
        ones_task.append(1)
    else:
        ones_task.append(result['cpu_milli'].values[0] / 10000)
    if result['gpu_milli'].values[0] > 10:
        ones_task.append(1)
    else:
        ones_task.append(result['gpu_milli'].values[0] / 10)
    ones_task.append((result['qos'].values[0]) / 3)
    ones_task.append((result['gpu_spec'].values[0] + 1) / 4)
    # the third
    ones_others=[]
    result_others = list(filter(lambda x: x != task_dicision, offload_task_list))
    number_other_task, size_cpu, size_gpu, priorty_lx, GPU_lx = 0, 0, 0, [0, 0, 0], [0, 0, 0, 0]
    for task_other in result_others:
        result = info[((info['time'] == time) & (info['time'].index == task_other[0]))]
        number_other_task += 1
        size_cpu += result['cpu_milli'].values[0]
        size_gpu += result['gpu_milli'].values[0]
        priorty_lx[result['qos'].values[0] - 1] += 1
        GPU_lx[(result['gpu_spec'].values[0] + 1)] += 1
    size_cpu = size_cpu / 50000
    size_gpu = size_gpu / 50
    priorty_lx = [x / 5 for x in priorty_lx]
    GPU_lx = [x / 5 for x in GPU_lx]
    ones_others.append(number_other_task / 100)
    ones_others.append(size_cpu)
    ones_others.append(size_gpu)
    ones_other = ones_others + priorty_lx + GPU_lx
    #the forth
    ones_mes=[]
    mes_task_wait,mes_level_task,mes_size_cpu,mes_size_gpu=np.zeros((M,3)),np.zeros((M,3)),np.zeros((M,3)),np.zeros((M,3))
    for mes in range(M):
        for type_ in range(3):
            mes_task_wait[mes][type_] += len(ES_First_level[mes][type_][time])+len(ES_Second_level[mes][type_][time])+len(ES_Third_level[mes][type_][time])
            mes_level_task[mes][0] += len(ES_First_level[mes][type_][time])
            mes_level_task[mes][1] += len(ES_Second_level[mes][type_][time])
            mes_level_task[mes][2] += len(ES_Third_level[mes][type_][time])
            for task_other in ES_First_level[mes][type_][time]:
                result = info[(info['time'].index == task_other[0])]
                mes_size_cpu[mes][type_] += result['cpu_milli'].values[0]
                mes_size_gpu[mes][type_] += result['gpu_milli'].values[0]
            for task_other in ES_Second_level[mes][type_][time]:
                result = info[(info['time'].index == task_other[0])]
                mes_size_cpu[mes][type_] += result['cpu_milli'].values[0]
                mes_size_gpu[mes][type_] += result['gpu_milli'].values[0]
            for task_other in ES_Third_level[mes][type_][time]:
                result = info[(info['time'].index == task_other[0])]
                mes_size_cpu[mes][type_] += result['cpu_milli'].values[0]
                mes_size_gpu[mes][type_] += result['gpu_milli'].values[0]
        for type_ in range(3):
            mes_task_wait[mes][type_] = mes_task_wait[mes][type_] / (10*N)
            mes_level_task[mes][type_] = mes_task_wait[mes][type_] / (5 * N)
            mes_size_cpu[mes][type_] = mes_size_cpu[mes][type_] / (10*N*100000)
            mes_size_gpu[mes][type_] = mes_size_gpu[mes][type_] / (10*N*10)
    ones_mes = mes_task_wait.reshape(-1).tolist() + mes_level_task.reshape(-1).tolist()+mes_size_cpu.reshape(-1).tolist()+mes_size_gpu.reshape(-1).tolist()
    output = Guiyi.get_output_mes(ones_other+ones_mes)
    return ones_time+ones_task+output.tolist()

def save_file(ephoch,info,wait_size_Vehile,wait_size_ES,value_all):
    save_name = curPath + save_path + "info_EU_" + str(config.get("Time")) + "_"+str(ephoch)+ "_" + '5' + ".csv"
    info.to_csv(save_name, index=None)
    save_name = curPath + save_path + "info_EU_" + str(config.get("Time")) + "_"+str(ephoch)+ "_" + 'EU_wait_size' + ".csv"
    wait_size_Vehile = pd.DataFrame(wait_size_Vehile)
    wait_size_Vehile.to_csv(save_name, index=False, header=False)
    save_name = curPath + save_path + "info_EU_" + str(config.get("Time")) + "_"+str(ephoch)+ "_" + 'MES_wait_size' + ".csv"
    wait_size_ES = pd.DataFrame(wait_size_ES)
    wait_size_ES.to_csv(save_name, index=False, header=False)
    save_name = curPath + save_path + "info_EU_" + str(config.get("Time")) + "_" + str(ephoch) + "_" + 'value_all' + ".csv"
    value_all = pd.DataFrame(value_all)
    value_all.to_csv(save_name, index=False, header=False)

if __name__ == '__main__':
    model_save_RSU = gv.model_save_RSU
    model_save_VE = gv.model_save_VE
    model_save_LSTM = gv.model_save_LSTM
    Guiyi_RSU = gv.Guiyi_RSU
    Guiyi_VE = gv.Guiyi_VE
    N = config.get("Dev_dev")
    M = config.get('Dev_edge')
    Band = config.get("B")
    K = config.get("K")
    P_tran_min = config.get('P_tran_min')
    P_tran_max = config.get('P_tran_max')
    alpha_1=config.get('alpha_1')
    coverage = config.get('Scope')
    vehicle_position = config.get("vehicle_position")
    edge_position = config.get("Edge_position")
    speed = config.get("Speed")
    noisy = config.get("noisy")
    loss_exponent = 3
    light = 3 * 10 ** 8 
    Ad = 3 
    fc = 915 * 10 ** 6
    multi_user = multiagent(config.get('Dev_dev'),config.get('Dev_edge'),1+4+6,2,1+4+6,2,9,9+M)
    Guiyi = unit_all(26,6,70,6)

    '''
    mem_mul_agent.Actor_RSU.load_state_dict(torch.load(model_save_RSU))
    environ_critic.critic_RSU.load_state_dict(torch.load(model_critic_RSU))
    for dev in range(N):
        mem_mul_agent.Actor_ve[dev].load_state_dict(torch.load(model_save_VE+str(dev)+"_"+"model_VE.pt"))
        environ_critic.Critic_VE[dev].load_state_dict(torch.load(model_save_VE+str(dev)+"_"+"model_critic_VE.pt"))
    '''
    for epoch in range(epoch_all):
        info = gv.info
        TimeZone = gv.TimeZone
        task_queue = gv.task_queue  # ID
        wait_tackel_queue = gv.wait_tackel_queue  # ID
        wait_offload_queue = gv.wait_offload_queue  # ID
        wait_size_ES = gv.ES_wait_size
        wait_size_Vehile = gv.wait_size_Vehile
        wait_Energy_Vehile = gv.Energy_Vehile
        ES_First_level = gv.ES_First_level
        ES_Second_level = gv.ES_Second_level
        ES_Third_level = gv.ES_Third_level
        All_F =np.zeros((config.get('Dev_dev'),config.get('Time')+701))
        All_P = np.zeros((config.get('Dev_dev'), config.get('Time') + 701))
        All_reward = np.zeros(config.get('Time') + 701)
        f_local = gv.F_eu
        pre_user=np.zeros((5, 1+1+4+3))
        value_all=[]
        for i in trange(config.get('Time') + 700):
            h0 = np.zeros((config.get('Dev_dev'), config.get('Dev_edge')))
            dist_v = np.zeros((config.get('Dev_dev'),config.get('Dev_edge')))
            velocity = np.zeros((config.get('Dev_dev'), config.get('Dev_edge')))
            for dev in range(config.get('Dev_dev')):
                for edge in range(config.get('Dev_edge')):
                   
                    dist_v[dev][edge] =math.sqrt((vehicle_position[dev][0] - edge_position[edge][0])**2+(vehicle_position[dev][1] - edge_position[edge][1])**2)
            dist_v_flatten =dist_v.flatten()
            for j in range(config.get('Dev_dev')):  #d*(light/4/math.pi/fc/dist_v[j])**(loss_exponent)
                for k in range(config.get('Dev_edge')):
                    h0[j][k] = Ad * (light / 4 / math.pi / fc / dist_v[j][k]) ** (loss_exponent)
            gen_task = TimeZone[i]

            append_to_task_queue(i, gen_task)
            for j in range(config.get('Dev_dev')):
                for k in range(config.get('Dev_edge')):
                    velocity[j][k] = Band * math.log2(
                        1 + ((h0[j][k] * config.get('P_tran_max')) / (10 ** ((noisy - 30) / 10)))) 
            ### 
            if i < config.get('Time'):
                ####user decision
                xz = []
                xz_sigmod = []
                input_user = []
                for n in range(N):
                    xz.append([])
                    xz_sigmod.append([])
                    input_user.append([])
                    the_fact = the_fact_user(i, n)
                    multi_user.update_para_LSTM(pre_user, the_fact)
                    pre_user = user_predict(i, n)
                    pre_next = multi_user.predict_L(pre_user)
                    for task_number in range(len(task_queue[n][i])):
                        single_task = get_user_para(i, n, task_number, pre_next)
                        input_user[n].append(single_task)
                        x = multi_user.choose_action_Vehicle(n, single_task, i)
                        for element in range(len(x)):
                            x[element][1] = math.floor(x[element][1] * 3)
                            if x[element][1] == 3:
                                x[element][1] = 2
                            if x[element][0] < 0.5:
                                x[element][0] = 0
                            else:
                                x[element][0] = 1
                        xz[n].append(x)
                        xz_sigmod[n].append([[x[0][0], x[0][1] / 2], [x[1][0], x[1][1] / 2]])
                #### MES decision
                yz = []
                yz_sigmod = []
                task_mes_input = []
                for index in range(2):
                    yz.append([])
                    yz_sigmod.append([])
                    task_mes_input.append([])
                    wait_offload_task = []
                    offload_task_y = []
                    for n in range(N):
                        wait_offload_task.append([])
                        for task_number in range(len(task_queue[n][i])):
                            if xz[n][task_number][index][0] == 1:
                                wait_offload_task[n].append(task_queue[n][i][task_number])
                                offload_task_y.append(task_queue[n][i][task_number])
                    #  obave get the all task need offload
                    for task_number in range(len(offload_task_y)):
                        single_task_mes = get_mes_para(i, task_number, offload_task_y)
                        task_mes_input[index].append(single_task_mes)
                        y = multi_user.choose_action_RSU(single_task_mes, i)
                        for element in range(len(y)):
                            y[element][0] = math.floor(y[element][0] * M)
                            if y[element][0] == M:
                                y[element][0] = M - 1
                            y[element][1] = math.floor(y[element][1] * 3)
                            if y[element][1] == 3:
                                y[element][1] = 2
                        yz[index].append(y)
                        yz_sigmod[index].append([[y[0][0] / (M - 1), y[0][1] / 2], [y[1][0] / (M - 1), y[1][1] / 2]])
                index_x, index_y, user_CV, value = MCT.critic_en(i, xz, yz, velocity)
                value_all.append(value)
                ###train
                # for dev in range(N):
                for dev in range(len(task_mes_input[index_x])):
                    multi_user.remember_RSU(task_mes_input[index_x][dev], yz_sigmod[index_x][dev][index_y])
                for dev in range(N):
                    for task_number in range(len(task_queue[dev][i])):
                        multi_user.remember_Vehile(dev, input_user[dev][task_number],
                                                   xz_sigmod[dev][task_number][index_x], user_CV[dev])
                x, y = [], []
                for dev in range(config.get('Dev_dev')):
                    x.append([])
                    y.append([])
                    off_index = 0
                    for task in xz[dev]:
                        x[dev].append(task[index_x])
                        if x[dev][-1][0] == 1:
                            y[dev].append(yz[index_x][off_index][index_y])
                            off_index += 1
                        else:
                            y[dev].append([])
                for dev in range(config.get('Dev_dev')):
                    updage_offload_decision(i, x, y, dev)
            f_offload = velocity  # f_offload

            
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
                            info.loc[(info['name'] == sub_task[1]) & (info['name'].index == sub_task[0]), 'gpu_spec'] = gpu_spec
                        else:
                            gpu_spec = int(result['gpu_spec'].values[0])
                        wait_tackel_queue[dev][i][gpu_spec].append(sub_task[:3])
                        wait_size_Vehile[dev][i] += result['cpu_milli'].values[0]
                    task_queue[dev][i].pop(0)
            loacl_in_time, offload_in_time = np.zeros((config.get('Dev_dev'),3,2)), np.zeros(config.get('Dev_dev'))
            for dev in range(config.get('Dev_dev')):
                offload_in_time[dev] = offload_task_tackel(i, dev, f_offload[dev], i)
                for type in range(3):
                    loacl_in_time[dev][type] = local_task_tackel(i, dev, f_local[type], i, i, type)
            End_loop_judge = End_user_loop(i, loacl_in_time, offload_in_time, f_local, f_offload)
            MES.MES_task_tackel(i)
            update_tackel_queue(i)
        torch.save(multi_user.Actor_RSU.state_dict(), model_save_RSU)
        torch.save(multi_user.LSTM_L.state_dict(), model_save_LSTM)
        torch.save(Guiyi.model_user.state_dict(), Guiyi_VE)
        torch.save(Guiyi.model_mes.state_dict(), Guiyi_RSU)
        for dev in range(N):
            torch.save(multi_user.Actor_ve[dev].state_dict(), model_save_VE+str(dev)+"_"+"model_VE.pt")
        save_file(epoch,info,wait_size_Vehile,wait_size_ES,value_all)
        importlib.reload(gv)
        importlib.reload(MES)
