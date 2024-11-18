import numpy as np
from config import config
import pandas as pd

curPath = config.get("curPath")
Datasets_path = config.get("Datasets_path")
epoch_all = config.get("epoch")
Model_path = config.get("Model_path")
Result_path = config.get("Result_path")
#alpha = config.get("alpha")
save_path = config.get("Datasets_path")

file_info = curPath + Datasets_path + "info_" + str(config.get("Time")) +"_"+str(config.get("Dev_dev"))+".csv"
info = pd.read_csv(file_info)  # 读任务文件
wait_size_Vehile = np.zeros((config.get('Dev_dev'),config.get('Time')+701))
Energy_Vehile = np.zeros((config.get('Dev_dev'),config.get('Time')+701))
file_es = curPath + Datasets_path + "info_GPU.csv"
info_GPU = pd.read_csv(file_es)  # 读任务文件

task_queue = []  # ID
wait_tackel_queue = []  # ID
wait_offload_queue = []  # ID
for _ in range(config.get('Dev_dev')):
    task_queue.append([[]])
    wait_tackel_queue.append([[[],[],[]]])
    wait_offload_queue.append([[]])

for i in range(config.get('Time')+701):
    for j in range(config.get('Dev_dev')):
        task_queue[j].append([])
        wait_tackel_queue[j].append([[], [], []])
        wait_offload_queue[j].append([])

F_eu=[[(info_GPU.iloc[0]['cpu_milli']*2),(info_GPU.iloc[0]['gpu']*2)],[(info_GPU.iloc[1]['cpu_milli']*2),(info_GPU.iloc[1]['gpu']*2)],[(info_GPU.iloc[2]['cpu_milli']*2),(info_GPU.iloc[2]['gpu']*2)]]
#######################################################################################################

ES_wait_queue=[]
ES_wait_size=[]
ES_First_level=[]
ES_Second_level=[]
ES_Third_level=[]

for _ in range(config.get("Dev_edge")):
    ES_wait_size.append([])
    ES_wait_queue.append([])
    ES_First_level.append([[],[],[]])
    ES_Second_level.append([[],[],[]])
    ES_Third_level.append([[],[],[]])

for _ in range(config.get('Time')+701):
    for j in range(config.get('Dev_edge')):
        ES_wait_size[j].append(0)
        ES_wait_queue[j].append([])
        for i in range(3):
            ES_First_level[j][i].append([])
            ES_Second_level[j][i].append([])
            ES_Third_level[j][i].append([])

F_es=[]
F_cycle_use = []
ES_cycle=[]
time_slot = [0.3,0.8,1]
for i in range(3):
    F_es.append([(info_GPU.iloc[i]['cpu_milli']),(info_GPU.iloc[i]['gpu'])])
    ES_cycle.append([[F_es[-1][0]*(12),F_es[-1][1]*(12)],[F_es[-1][0]*(12),F_es[-1][1]*(12)],[F_es[-1][0]*(12),F_es[-1][1]*(12)]])
    F_cycle_use.append([[time_slot[0] * ES_cycle[-1][0][0],time_slot[0] * ES_cycle[-1][0][1]],[time_slot[1] * ES_cycle[-1][1][0],time_slot[1] * ES_cycle[-1][1][1]],[time_slot[2] * ES_cycle[-1][2][0],time_slot[2] * ES_cycle[-1][2][1]]])  # each slot will tackel cycles

###########################################################################################################################
TimeZone = []
for i in range(config.get("Time") + 700):
    temp = info.loc[info['time'] == i].index.tolist()
    TimeZone.append(temp)



#all string to save
#NL para save
model_save_RSU = curPath + save_path + "model_" + str(config.get("Time")) + "_" + str(config.get("Dev_dev")) + "_1_" + "model_RSU.pt"
model_save_VE = curPath + save_path + "model_" + str(config.get("Time")) + "_" + str(config.get("Dev_dev")) + "_1_"
model_save_LSTM = curPath + save_path + "model_" + str(config.get("Time")) + "_" + str(config.get("Dev_dev")) + "LSTM.pt"
Guiyi_RSU = curPath + save_path + "model_" + str(config.get("Time")) + "_" + str(config.get("Dev_dev")) + "Allin_RSU.pt"
Guiyi_VE = curPath + save_path + "model_" + str(config.get("Time")) + "_" + str(config.get("Dev_dev")) + "Allin_VE.pt"