# %%
from utils_excel import read_log_databaseloader as databaseloader
import numpy as np
from utils_calculation import calculator

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
check_line_list = [163,183,3,63,83,103,23,43,207,227,247,267,287,143,123]

save_root_path = 'statistics_figure_int8/'
rld = databaseloader(file_path=os.path.abspath('../log/orderlog/cpu_test_database_after_2020_8_run_by_arc3.xls'))
c = calculator()
check_float32_line_inception_list = [243,263,83,143,163,183,103,123,283,323,363,403,444,223,203]
rld_float32 = databaseloader()


def modelpath2modelname(model_path):  #transform float32 model path to model name
    if model_path == '../quantization_models/inception_v3.pth':
        model_name = 'inception_v3_quantization'
    else:
        model_name = ((model_path.split("/")[1]).split("scripted")[0])[:-1]
    return model_name
#%% 
def get_storage_usage(estimate_line_list = []):# need to fill up all the size parameters
    model_name_list = []
    model_mean_list = []
    model_mean_confinterval = []
    model_geo_mean_list = []
    model_geo_mean_confinterval = []
    for current_estimate_line in estimate_line_list:
        model_path = rld.read_model_path(current_estimate_line)
        current_model_name = modelpath2modelname(model_path)
        model_name_list.append(current_model_name)
        select_model_id_list = rld.find_special_line_id(model_path)
        current_model_result_list = rld.read_line_vector(rld.read_size,select_model_id_list)
        # #mean information log
        mean_result = c.mean(current_model_result_list)
        model_mean_list.append(mean_result)
        meanInterval = c.mean_interval(arraylist = current_model_result_list)
        model_mean_confinterval.append(mean_result - meanInterval[0])

        # #Geo mean log
        geo_mean_result = c.geo_mean(current_model_result_list)
        model_geo_mean_list.append(geo_mean_result)
        geo_meanInterval = c.geo_mean_interval(arraylist = current_model_result_list)
        model_geo_mean_confinterval.append(geo_mean_result - geo_meanInterval[0])
    return model_name_list, model_mean_list, model_mean_confinterval, model_geo_mean_list, model_geo_mean_confinterval

def get_float32_storage_usage(estimate_line_list = []):# need to fill up all the size parameters
    model_name_list = []
    model_mean_list = []
    model_mean_confinterval = []
    model_geo_mean_list = []
    model_geo_mean_confinterval = []
    for current_estimate_line in estimate_line_list:
        model_path = rld_float32.read_model_path(current_estimate_line)
        current_model_name = ((model_path.split("/")[1]).split("pretrained")[0])[:-1]# transform float32 model path to model name
        model_name_list.append(current_model_name)
        select_model_id_list = rld_float32.find_special_line_id(model_path)
        current_model_result_list = rld_float32.read_line_vector(rld_float32.read_size,select_model_id_list)

        mean_result = c.mean(current_model_result_list)
        model_mean_list.append(mean_result)
        meanInterval = c.mean_interval(arraylist = current_model_result_list)
        model_mean_confinterval.append(mean_result - meanInterval[0])

        # #Geo mean log
        geo_mean_result = c.geo_mean(current_model_result_list)
        model_geo_mean_list.append(geo_mean_result)
        geo_meanInterval = c.geo_mean_interval(arraylist = current_model_result_list)
        model_geo_mean_confinterval.append(geo_mean_result - geo_meanInterval[0])
    return model_name_list, model_mean_list, model_mean_confinterval, model_geo_mean_list, model_geo_mean_confinterval

#%%
model_name_list, model_mean_list, model_mean_confinterval, model_float32_geo_mean_list, model_geo_mean_confinterval = get_float32_storage_usage(check_float32_line_inception_list)
model_name_list, model_mean_list, model_mean_confinterval, model_geo_mean_list, model_geo_mean_confinterval = get_storage_usage(check_line_list)
print(len(model_name_list))
print(model_name_list)
decrease_storage_usage_percentage =  1 - np.array(model_geo_mean_list)/ np.array(model_float32_geo_mean_list)
print(decrease_storage_usage_percentage)


x_lable = model_name_list

x = np.arange(0,len(x_lable))
plt.figure(figsize=(len(x_lable)+1,7))

bar_width = 0.2
bar = plt.bar(x, decrease_storage_usage_percentage, bar_width, color=(32/256, 234/256, 54/256))

#show the bar spesific number
for x_axis, a, b in zip(x, x_lable, decrease_storage_usage_percentage):
    plt.text(x_axis, b+0.0015, round(b, 4), ha='center', va='bottom', fontsize=12)
    
plt.xticks(rotation=20)
plt.xticks(x,x_lable, horizontalalignment='right')

plt.ylabel('decrease storage usage(unti: %')
plt.savefig(save_root_path + "decrease storage usage.png",dpi=500,bbox_inches = 'tight')
plt.show()


# %%
