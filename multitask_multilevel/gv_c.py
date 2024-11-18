import copy
import gloable_variation as gv


info = copy.deepcopy(gv.info)
task_queue = copy.deepcopy(gv.task_queue)
#sub_task_queue = copy.deepcopy(gv.sub_task_queue)
wait_tackel_queue = copy.deepcopy(gv.wait_tackel_queue)
wait_offload_queue = copy.deepcopy(gv.wait_offload_queue)
ES_wait_queue = copy.deepcopy(gv.ES_wait_queue)
ES_First_level = copy.deepcopy(gv.ES_First_level)
ES_Second_level = copy.deepcopy(gv.ES_Second_level)
ES_Third_level = copy.deepcopy(gv.ES_Third_level)
wait_size_ES = copy.deepcopy(gv.ES_wait_size)
wait_size_Vehile = copy.deepcopy(gv.wait_size_Vehile)
wait_Energy_Vehile = copy.deepcopy(gv.Energy_Vehile)
f_local = gv.F_eu
