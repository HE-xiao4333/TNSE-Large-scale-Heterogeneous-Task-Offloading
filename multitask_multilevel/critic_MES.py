import gv_c as ce
import gloable_variation as gv
from config import config
import numpy as np
import importlib




info = ce.info
ES_wait_queue=ce.ES_wait_queue
ES_First_level=ce.ES_First_level
ES_Second_level=ce.ES_Second_level
ES_Third_level=ce.ES_Third_level
wait_size_ES = ce.wait_size_ES
F_es=gv.F_es
ES_cycle=gv.ES_cycle
time_slot = gv.time_slot
F_cycle_use = gv.F_cycle_use
TimeZone = gv.TimeZone


def global_update():
    global ES_wait_queue,ES_First_level,ES_Second_level,ES_Third_level,wait_size_ES,info
    info = ce.info
    ES_wait_queue = ce.ES_wait_queue
    ES_First_level = ce.ES_First_level
    ES_Second_level = ce.ES_Second_level
    ES_Third_level = ce.ES_Third_level
    wait_size_ES = ce.wait_size_ES

def get_all_subtask(time,gen_task):
    task_queue=[]
    for task_number in gen_task:
        result = info.loc[(info['name'].index==task_number)&(info['time']==time)]
        if result['offload'].values[0] == 1:
            task_queue.append([task_number, result['name'].values[0], result['memory_mib'].values[0],result['off_end'].values[0]])
    return task_queue

def append_First_level(time,gen_task):
    if not gen_task:
        return
    sorted_list = sorted(gen_task, key=lambda x: x[-1])
    for task in sorted_list:
        result = info.loc[(info['name'] == task[1])&(info['name'].index==task[0])]
        es_select = result['to'].values[0]
        ES_First_level[es_select][int(result['gpu_spec'].values[0])][time].append(task)


def judge_waittask_is_availble(time,es,t,type_,i,Queue_ES):
    if not Queue_ES[es][time]:
        return [-1,-1]
    if i >= len(Queue_ES[es][time]):
        return [-1,-1]
    task_id = Queue_ES[es][time][i]
    result = info.loc[(info['name'] ==task_id[1]) & (info['name'].index == task_id[0]) & (info['gpu_spec'] == type_)]
    k = [0, 0]
    if result.empty:
        return [-1,-1]

    if result['off_end'].values[0] > t:
        judge = 0
    else:
        judge = result['offload_success'].values[0]
    k[0] = judge
    # judge the task is or not affloaded complete
    if result['offload_success'].values[0] == 1:
        if (result['off_end'].values[0] <= t):  # offload complete and end_time <now_time
            k[1] = 1
        elif (result['off_end'].values[0] - t < 0.000000001):
            k[1] = 1
    return k

def judge_task_is_availble(time,es,t,type,Queue_ES=ES_First_level):
    k,t__ = [0, 0],t
    if not Queue_ES[es][type][time]:
         return -1,t
    task_id = Queue_ES[es][type][time][0]
    result = info.loc[(info['name'] == task_id[1])& (info['name'].index == task_id[0])]
    if result['off_end'].values[0] > t:  
        judge=0
    else: 
        judge=result['offload_success'].values[0]
    k[0] = judge
    #judge the task is or not affloaded complete
    if result['offload_success'].values[0] == 1:
        if result['off_end'].values[0] <= t:#offload complete and end_time <now_time
            k[1] = 1
        elif result['off_end'].values[0] - t < 0.000000001:
            k[1] = 1
        else:
            if (result['off_end'].values[0] < time + 1) & (result['off_end'].values[0] > t):
                k[1] = 1
                t__ = result['off_end'].values[0]
    return k,t__

def off_task_tackel_Fir(time, es, t,type_): 
    k, i = [0, 0], 0
    while np.any(np.array(k) != 1):
        if len(ES_wait_queue[es][time]) != 0:
            k = judge_waittask_is_availble(time, es, t, type_, i,ES_wait_queue)
            if (k == [-1, -1]):
                break
            i = i + 1
        else:
            break
    if t - time >= 1:
        return time+1  # 'time out'
    if np.all(np.array(k) == 1):
        ES_First_level[es][type_][time].insert(0, ES_wait_queue[es][time][i - 1])
        ES_wait_queue[es][time].remove(ES_wait_queue[es][time][i - 1])
    if not ES_First_level[es][type_][time]:
        return t

    K, t = judge_task_is_availble(time, es, t, type_,ES_First_level)

    if np.all(np.array(K) == 1):
        task = ES_First_level[es][type_][time][0]
        result = info.loc[(info['name'] == task[1])& (info['name'].index == task[0])]
        task_need_cpu = result['cpu_milli'].values[0] - result['complete_size_cpu'].values[0]
        task_need_gpu = result['gpu_milli'].values[0] - result['complete_size_gpu'].values[0]
        run = info.loc[(info['name'] == task[1])& (info['name'].index == task[0]), 'run'].values[0]
        if t - time <= (1 - time_slot[0]):
            F_tack = F_cycle_use[type_][0] 
            if (run >= 0) & (run <= time_slot[0]):
                F_tack = np.multiply((time_slot[0] - run), ES_cycle[type_][0])
        else:
            F_tack = np.multiply((1 - (t - time)), ES_cycle[type_][0])
            if (run >= 0) & (run <= time_slot[0]):
                if (time_slot[0] - run) < (1 - (t - time)):
                    F_tack = np.multiply((time_slot[0] - run), ES_cycle[type_][0])
        complete,run_all=[0,0],[0,0]
        if (result['complete_size_cpu'].values[0] == 0) & (result['complete_size_gpu'].values[0] == 0):
            info.loc[(info['name'] == task[1]) & (info['name'].index == task[0]), 'start'] = t

        if task_need_cpu <= F_tack[0]:
            info.loc[(info['name'] == task[1]) & (info['name'].index == task[0]), 'complete_size_cpu'] = result['complete_size_cpu'].values[0] + task_need_cpu
            info.loc[(info['name'] == task[1]) & (info['name'].index == task[0]), 'complete_cpu'] = 1
            use_time = task_need_cpu / ES_cycle[type_][0][0]
            time_cpu = t +use_time
            complete[0]=1
        else:
            info.loc[(info['name'] == task[1]) & (info['name'].index == task[0]), 'complete_size_cpu'] = result['complete_size_cpu'].values[0] + F_tack[0]
            use_time = F_tack[0] / ES_cycle[type_][0][0]
            time_cpu = t +use_time
            run_all[0] = use_time+ run
            if time_cpu > time + 1:
                time_cpu = time + 1

        if task_need_gpu <= F_tack[1]:
            info.loc[(info['name'] == task[1]) & (info['name'].index == task[0]), 'complete_size_gpu'] = result['complete_size_gpu'].values[0] + task_need_gpu
            info.loc[(info['name'] == task[1]) & (info['name'].index == task[0]), 'complete_gpu'] = 1
            use_time = task_need_gpu / ES_cycle[type_][0][1]
            time_gpu = t +use_time
            complete[1]=1
        else:
            info.loc[(info['name'] == task[1]) & (info['name'].index == task[0]), 'complete_size_gpu'] = result['complete_size_gpu'].values[0] + F_tack[1]
            use_time = F_tack[1] / ES_cycle[type_][0][1]
            time_gpu = t +use_time
            run_all[1] = use_time+ run
            if time_gpu > time + 1:
                time_gpu = time + 1
        if (complete[1]==1) & (complete[0]==1):
            info.loc[(info['name'] == task[1]) & (info['name'].index == task[0]), 'end'] = max(time_gpu, time_cpu)
            wait_size_ES[result['to'].values[0]][time] = wait_size_ES[result['to'].values[0]][time]  - result['cpu_milli'].values[0]
            ES_First_level[es][type_][time].pop(0)
            return max(time_gpu, time_cpu)
        if (run_all[0] >= time_slot[0]) or (run_all[1] >= time_slot[0]):
            info.loc[(info['name'] == task[1]) & (info['name'].index == task[0]), 'run'] = 0
            info.loc[(info['name'] == task[1]) & (info['name'].index == task[0]), 'end'] = max(time_gpu,time_cpu)
            ES_Second_level[es][type_][time].append(ES_First_level[es][type_][time].pop(0))
        else:
            info.loc[(info['name'] == task[1]) & (info['name'].index == task[0]), 'run'] = max(run_all)
        return max(time_gpu,time_cpu)
    else:
        ES_wait_queue[es][time].append(ES_First_level[es][type_][time].pop(0))
        return t

def off_task_tackel_Sec(time, es, t,type_):
    if not ES_Second_level[es][type_][time]:
        return t
    if t - time > 1:
        return time + 1
    K, t = judge_task_is_availble(time, es, t, type_,ES_Second_level)
    task = ES_Second_level[es][type_][time][0]
    result = info.loc[(info['name'] == task[1]) & ((info['name'].index == task[0]))]
    if (result['complete_size_cpu'].values[0] >0) or (result['complete_size_gpu'].values[0] >0):
        K[0] = 1
    if np.all(np.array(K) == 1):
        task_need_cpu = result['cpu_milli'].values[0] - result['complete_size_cpu'].values[0]
        task_need_gpu = result['gpu_milli'].values[0] - result['complete_size_gpu'].values[0]
        if (result['complete_size_cpu'].values[0] == 0) & (result['complete_size_gpu'].values[0] == 0):
            info.loc[(info['name'] == task[1]) & (info['name'].index == task[0]), 'start'] = t
        run =info.loc[(info['name'] == task[1])  & ((info['name'].index == task[0])), 'run'].values[0]
        if result['end'].values[0] > t:
            t = result['end'].values[0]
        if t - time <= (1 - time_slot[1]):
            F_tack = F_cycle_use[type_][1]
            if (run < time_slot[1]):
                F_tack = np.multiply((time_slot[1] - run), ES_cycle[type_][1])
        else:
            F_tack = np.multiply((1 - (t - time)), ES_cycle[type_][1])
            if (run < time_slot[1]):
                if (time_slot[1] - run) <= (1 - (t - time)):
                    F_tack = np.multiply((time_slot[1] - run), ES_cycle[type_][1])
        complete,run_all = [0,0],[0,0]
        if task_need_cpu <= F_tack[0]:
            info.loc[(info['name'] == task[1])  & ((info['name'].index == task[0])), 'complete_size_cpu'] = result['complete_size_cpu'].values[0] + task_need_cpu
            info.loc[(info['name'] == task[1])  & ((info['name'].index == task[0])), 'complete_cpu'] = 1
            use_time = task_need_cpu / ES_cycle[type_][1][0]
            run_all[0] = use_time + run
            time_cpu = t + use_time
            complete[0] = 1
        else:
            info.loc[(info['name'] == task[1])  & ((info['name'].index == task[0])), 'complete_size_cpu'] = result['complete_size_cpu'].values[0] + F_tack[0]
            use_time = F_tack[0] / ES_cycle[type_][1][0]
            run_all[0] = use_time + run
            time_cpu = t + use_time
            if time_cpu > time + 1:
                time_cpu = time + 1

        if task_need_gpu <= F_tack[1]:
            info.loc[(info['name'] == task[1])  & ((info['name'].index == task[0])), 'complete_size_gpu'] = result['complete_size_gpu'].values[0] + task_need_gpu
            info.loc[(info['name'] == task[1])  & ((info['name'].index == task[0])), 'complete_gpu'] = 1
            use_time = task_need_gpu / ES_cycle[type_][1][1]
            run_all[1] = use_time + run
            time_gpu = t + use_time
            complete[1] = 1
        else:
            info.loc[(info['name'] == task[1])  & ((info['name'].index == task[0])), 'complete_size_gpu'] = result['complete_size_gpu'].values[0] + F_tack[1]
            use_time = F_tack[1] / ES_cycle[type_][1][1]
            run_all[1] = use_time + run
            time_gpu = t + use_time
            if time_gpu > time + 1:
                time_gpu = time + 1

        if (complete[1] == 1) & (complete[0] == 1):
            info.loc[(info['name'] == task[1]) & (info['name'].index == task[0]), 'end'] = max(time_gpu, time_cpu)
            wait_size_ES[result['to'].values[0]][time] = wait_size_ES[result['to'].values[0]][time] - result['cpu_milli'].values[0]
            ES_Second_level[es][type_][time].pop(0)
            return max(time_gpu, time_cpu)
        if (run_all[0] >= time_slot[1]) or (run_all[1] >= time_slot[1]):
            info.loc[(info['name'] == task[1]) & (info['name'].index == task[0]), 'run'] = 0
            info.loc[(info['name'] == task[1]) & (info['name'].index == task[0]), 'end'] = max(time_gpu, time_cpu)
            ES_Third_level[es][type_][time].append(ES_Second_level[es][type_][time].pop(0))
        else:
            info.loc[(info['name'] == task[1]) & (info['name'].index == task[0]), 'run'] = max(run_all)
        return max(time_gpu, time_cpu)
    else:
        return t
def off_task_tackel_Tir(time, es, t,type_):
    if not ES_Third_level[es][type_][time]:
        return t
    if t - time > 1:
        return time + 1
    K, t = judge_task_is_availble(time, es, t, type_, ES_Third_level)
    task = ES_Third_level[es][type_][time][0]
    result = info.loc[(info['name'] == task[1]) & ((info['name'].index == task[0]))]
    if (result['complete_size_cpu'].values[0] > 0) or (result['complete_size_gpu'].values[0] > 0):
        K[0] = 1
    if np.all(np.array(K) == 1):
        task_need_cpu = result['cpu_milli'].values[0] - result['complete_size_cpu'].values[0]
        task_need_gpu = result['gpu_milli'].values[0] - result['complete_size_gpu'].values[0]
        if (result['complete_size_cpu'].values[0] == 0) & (result['complete_size_gpu'].values[0] == 0):
            info.loc[(info['name'] == task[1]) & (info['name'].index == task[0]), 'start'] = t
        run = info.loc[(info['name'] == task[1]) & ((info['name'].index == task[0])), 'run'].values[0]
        if result['end'].values[0] > t:
            t = result['end'].values[0]
        if t - time <= (1 - time_slot[2]):
            F_tack = F_cycle_use[type_][2]
            if (run < time_slot[2]): 
                F_tack = np.multiply((time_slot[2] - run), ES_cycle[type_][2])
        else:
            F_tack = np.multiply((1 - (t - time)), ES_cycle[type_][2]) 
            if (run < time_slot[2]):
                if (time_slot[2] - run) <= (1 - (t - time)):
                    F_tack = np.multiply((time_slot[2] - run), ES_cycle[type_][2])

        complete, run_all = [0, 0], [0, 0]
        if task_need_cpu <= F_tack[0]:
            info.loc[(info['name'] == task[1]) & ((info['name'].index == task[0])), 'complete_size_cpu'] = result['complete_size_cpu'].values[0] + task_need_cpu
            info.loc[(info['name'] == task[1]) & ((info['name'].index == task[0])), 'complete_cpu'] = 1
            use_time = task_need_cpu / ES_cycle[type_][2][0]
            run_all[0] = use_time + run
            time_cpu = t + use_time
            complete[0] = 1
        else:
            info.loc[(info['name'] == task[1]) & ((info['name'].index == task[0])), 'complete_size_cpu'] = result['complete_size_cpu'].values[0] + F_tack[0]
            use_time = F_tack[0] / ES_cycle[type_][2][0]
            run_all[0] = use_time + run
            time_cpu = t + use_time
            if time_cpu > time + 1:
                time_cpu = time + 1

        if task_need_gpu <= F_tack[1]:
            info.loc[(info['name'] == task[1]) & ((info['name'].index == task[0])), 'complete_size_gpu'] = result['complete_size_gpu'].values[0] + task_need_gpu
            info.loc[(info['name'] == task[1]) & ((info['name'].index == task[0])), 'complete_gpu'] = 1
            use_time = task_need_gpu / ES_cycle[type_][2][1]
            run_all[1] = use_time + run
            time_gpu = t + use_time
            complete[1] = 1
        else:
            info.loc[(info['name'] == task[1]) & ((info['name'].index == task[0])), 'complete_size_gpu'] = result['complete_size_gpu'].values[0] + F_tack[1]
            use_time = F_tack[1] / ES_cycle[type_][2][1]
            run_all[1] = use_time + run
            time_gpu = t + use_time
            if time_gpu > time + 1:
                time_gpu = time + 1

        if (complete[1] == 1) & (complete[0] == 1):
            info.loc[(info['name'] == task[1]) & (info['name'].index == task[0]), 'end'] = max(time_gpu, time_cpu)
            wait_size_ES[result['to'].values[0]][time] = wait_size_ES[result['to'].values[0]][time] - result['cpu_milli'].values[0]
            ES_Third_level[es][type_][time].pop(0)
            return max(time_gpu, time_cpu)
        if (run_all[0] >= time_slot[2]) or (run_all[1] >= time_slot[2]):
            info.loc[(info['name'] == task[1]) & (info['name'].index == task[0]), 'run'] = 0
            info.loc[(info['name'] == task[1]) & (info['name'].index == task[0]), 'end'] = max(time_gpu, time_cpu)
            ES_Third_level[es][type_][time].append(ES_Third_level[es][type_][time].pop(0))
        else:
            info.loc[(info['name'] == task[1]) & (info['name'].index == task[0]), 'run'] = max(run_all)
        return max(time_gpu, time_cpu)
    else:
        return t

def MES_loop_test(i,First_time,Second_time,Third_time):
    not_same_tackel = [True] * (config.get('Dev_edge')*3)
    old_Fir = First_time.copy()
    old_Sec = Second_time.copy()
    old_Thi = Third_time.copy()
    for dev in range(config.get('Dev_edge')):
        for type in range(3):
            if (First_time[dev][type] - i) < 1 :
                First_time[dev][type] = off_task_tackel_Fir(i, dev, First_time[dev][type],type)
            if (Second_time[dev][type] - i < 1) :
                Second_time[dev][type] = off_task_tackel_Sec(i, dev, First_time[dev][type],type)
            if (Third_time[dev][type] - i < 1):
                Third_time[dev][type] = off_task_tackel_Tir(i, dev, Second_time[dev][type],type)
            if (First_time[dev][type] == old_Fir[dev][type]) & (Second_time[dev][type] == old_Sec[dev][type]) & (
                    Third_time[dev][type] == old_Thi[dev][type]):
                not_same_tackel[dev*3+type] = False
        #sec_thi_sort(dev,i)
    return any(not_same_tackel),First_time,Second_time,Third_time

def MES_task_tackel(i):
    gen_task = TimeZone[i]
    gen_task = get_all_subtask(i, gen_task)
    append_First_level(i, gen_task)
    t = [[i, i, i] for _ in range(config.get('Dev_edge'))]
    t_2 = [[i, i, i] for _ in range(config.get('Dev_edge'))]
    t_3 = [[i, i, i] for _ in range(config.get('Dev_edge'))]
    for dev in range(config.get('Dev_edge')):
        for type in range(3):
            if not ES_First_level[dev][type][i]:
                t[dev][type] = i
            elif ES_First_level[dev][type][i][0][3]>i:
                t[dev][type] = ES_First_level[dev][type][i][0][3]
            if not ES_Second_level[dev][type][i]:
                t_2[dev][type] = i
            elif ES_Second_level[dev][type][i][0][3] > i:
                t_2[dev][type] = ES_Second_level[dev][type][i][0][3]
            if not ES_Third_level[dev][type][i]:
                t_3[dev][type] = i
            elif ES_Third_level[dev][type][i][0][3] > i:
                t_3[dev][type] = ES_Third_level[dev][type][i][0][3]

    First_time, Second_time, Third_time = np.zeros((config.get('Dev_edge'), 3)), np.zeros(
        (config.get('Dev_edge'), 3)), np.zeros((config.get('Dev_edge'), 3))
    for dev in range(config.get('Dev_edge')):
        for type in range(3):
            First_time[dev][type] = off_task_tackel_Fir(i, dev, t[dev][type], type)
            Second_time[dev][type] = off_task_tackel_Sec(i, dev, First_time[dev][type], type)
            Third_time[dev][type] = off_task_tackel_Tir(i, dev, Second_time[dev][type], type)
    while 1:
        update, First_time, Second_time, Third_time = MES_loop_test(i, First_time, Second_time, Third_time)
        if update == False:
            break
    ES_Queue_update(i)

def ES_Queue_update(time):
    if time == config.get('Time') + 700:
        return
    for dev in range(config.get('Dev_edge')):
        for type in range(3):
            while ES_First_level[dev][type][time]:
                ES_First_level[dev][type][time+1].append(ES_First_level[dev][type][time].pop(0))
            while ES_Second_level[dev][type][time]:
                ES_Second_level[dev][type][time+1].append(ES_Second_level[dev][type][time].pop(0))
            while ES_Third_level[dev][type][time]:
                ES_Third_level[dev][type][time+1].append(ES_Third_level[dev][type][time].pop(0))
        while ES_wait_queue[dev][time]:
            ES_wait_queue[dev][time+1].append(ES_wait_queue[dev][time].pop(0))
        wait_size_ES[dev][time + 1] = wait_size_ES[dev][time] + wait_size_ES[dev][time + 1]
