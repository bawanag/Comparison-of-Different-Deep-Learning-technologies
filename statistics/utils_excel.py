import xlrd
import os
from datetime import datetime
import time
import utils_calculation as calculator
class read_log_databaseloader:
    def __init__(self, file_path="../log/gpu_test_database.xls", sheet_index = 0):
        if os.path.exists(file_path):
            self.wb = xlrd.open_workbook(filename=file_path)
            self.sheet = self.wb.sheet_by_index(sheet_index)
        else:
            print("excel database file is not exit.")

    def transform_tensorstr2list(self, tensorstr=""):
        a = tensorstr.strip('\n').strip('tensor(').strip(')').strip(
            '\n').strip('[').strip(']').strip('\n').strip(' ')
        list_a = a.split(",")
        new_list = []
        for stringss in list_a:
            sl = stringss.split('0.')
            if(len(sl) > 1):
                new_list.append(float("0."+sl[1]))
            else:
                new_list.append(float(stringss))
        return new_list

    def read_model_path(self, row=0):
        return (self.sheet.cell_value(row, 1))

    def read_execute_current_time(self, row=0):
        result = self.sheet.cell_value(row, 2)
        return self.current_time_string2obj(result)

    def read_size(self, row=0):
        result = self.sheet.cell_value(row, 3)
        return round(float(result),4)

    def read_Inference_time(self, row=0):
        result = self.sheet.cell_value(row, 4)
        return self.transform_datatime_to_floatsecond(self.execute_time_string2obj(result))

    def read_running_time(self, row=0):
        result = self.sheet.cell_value(row, 5)
        return self.transform_datatime_to_floatsecond(self.execute_time_string2obj(result))

    def read_Top1_Accuracy(self, row=0):
        result = self.sheet.cell_value(row, 6)
        return round(float(result),4)

    def read_Top5_Accuracy(self, row=0):
        result = self.sheet.cell_value(row, 7)
        return round(float(result),4)

    def read_precision(self, row=0):
        result = self.sheet.cell_value(row, 8)
        return self.transform_tensorstr2list(result)

    def read_recall(self, row=0):
        result = self.sheet.cell_value(row, 9)
        return self.transform_tensorstr2list(result)

    def read_F1_score(self, row=0):
        result = self.sheet.cell_value(row, 10)
        class1k_result = self.transform_tensorstr2list(result)
        return class1k_result

    def read_model_host_memory_usage(self, row=0):
        result = self.sheet.cell_value(row, 11)
        return float(result)

    def read_cross_entropy_loss(self, row=0):
        result = self.sheet.cell_value(row, 12)
        return round(float(result),8)

    def read_model_cuda_memory_usage(self, row=0):
        result = self.sheet.cell_value(row, 13)
        return float(result)

    def read_cpu_info(self, row=0):
        result = self.sheet.cell_value(row, 14)
        return result

    def read_parameter_number(self, row=0):
        result = self.sheet.cell_value(row, 15)
        return round(float(result/1000000),4)

    def read_GFLOPs(self, row=0):
        result = self.sheet.cell_value(row, 16)
        return float(result)

    def read_consecutive_line(self, read_function, start_row=0, end_row=1):

        return_vector = []
        for i in range(start_row, end_row):
            return_vector.append(read_function(i))
        return return_vector

    def read_line_vector(self, read_function, line_vector=[]):

        return_vector = []
        for i in line_vector:
            return_vector.append(read_function(i))
        return return_vector

    def find_special_line_id(self, model_path=""):
        table = self.sheet
        nrows = table.nrows
        result_list = []
        for i in range(0, nrows):
            current_line = table.row_values(i)
            if (current_line[1] == model_path):
                result_list.append(i)
        return result_list

    def execute_time_string2obj(self, date_time_str=""):
        date_time_obj = datetime.strptime(date_time_str, '%H:%M:%S.%f')
        return date_time_obj

    def current_time_string2obj(self, date_time_str=""):
        date_time_obj = datetime.strptime(
            date_time_str, '%y-%m-%d %H:%M:%S.%f')
        return date_time_obj

    def transform_datatime_to_floatsecond(self, dt): #cannot exceed one day
        second_count = 0.0
        second_count = dt.second + dt.minute*60 + \
            dt.hour*60*60+round(dt.microsecond*0.000001, 6)
        return second_count


if __name__ == "__main__":
    rld = read_log_databaseloader('../log/test_model_running_effciency_cpu_quantized_model.xls')
    c = calculator.calculator()
    print(rld.read_model_host_memory_usage(23))
    haha = rld.read_line_vector(rld.read_model_host_memory_usage,rld.find_special_line_id(rld.read_model_path(23)))
    print(type(haha[0]))   
    print(haha)
    print(c.mean(haha))

    # print(time1.microsecond)
    # haha = rld.transform_datatime_to_floatsecond(time1)
    # print(haha)


# database_dictionary = {"model_path" : 1, "execute_current_time" : 2, \
#                         "size" : 3, "Inference_time" : 4, "running_time" : 5, "Top1_Accuracy" : 6, \
#                         "Top5_Accuracy" : 7, "precision" : 8, "recall" : 9, "F1_score" : 10, \
#                         "mode_hostl_memory_usage(MB)" : 11, "cross_entropy_loss" : 12, \
#                         "model_cuda_memory_usage" : 13, "Testbed_cpu_information" : 14}

# default_gpu_test_log_path = "../log/gpu_test_database.xls"
# default_cpu_test_log_path = "../log/cpu_test_database.xls"
# default_test_model_running_effciency_cpu_path = "../log/test_model_running_effciency_cpu.xls"
