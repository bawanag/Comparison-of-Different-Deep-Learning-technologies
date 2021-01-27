import xlrd
from xlutils.copy import copy
from xlutils.margins import number_of_good_rows
class TestDatabaseUtil(object):
    def  __init__(self, database_path_location): 
        self.database_path_location = database_path_location
        workbook = xlrd.open_workbook(filename = database_path_location, formatting_info=False)
        self.new_book = copy(workbook)
        self.sheet = self.new_book.get_sheet(0)
        self.work_row = len(self.sheet._Worksheet__rows)

#		model_path	execute_current_time	size	Inference_time	running_time	Top1_Accuracy	Top5_Accuracy	precision	recall	F1_score	model_host_memory_usage	cross_entropy_loss model_cuda_memory_usage Testbed_cpu_information    parameters_number   GFLOPs
#_0	______1	______________2	_________________3	_____________4	__________5	__________________6	_________7	_____________8	_______9	____10	_____________11	_____________12______________________13________________________14______________________15___________________16

    def add_model_path(self, input):
        input = str(input)
        self.sheet.write(self.work_row, 1, input)
        return True

    def add_execute_current_time(self, input):
        input = str(input)
        self.sheet.write(self.work_row, 2, input)
        return True

    def add_size(self, input):
        input = str(input)
        self.sheet.write(self.work_row, 3, input)
        return True

    def add_Inference_time(self, input):
        input = str(input)
        self.sheet.write(self.work_row, 4, input)
        return True

    def add_running_time(self, input):
        input = str(input)
        self.sheet.write(self.work_row, 5, input)
        return True

    def add_Top1_Accuracy(self, input):
        input = str(input)
        self.sheet.write(self.work_row, 6, input)
        return True

    def add_Top5_Accuracy(self, input):
        input = str(input)
        self.sheet.write(self.work_row, 7, input)
        return True

    def add_precision(self, input):
        input = str(input)
        self.sheet.write(self.work_row, 8, input)
        return True

    def add_recall(self, input):
        input = str(input)
        self.sheet.write(self.work_row, 9, input)
        return True

    def add_F1_score(self, input):
        input = str(input)
        self.sheet.write(self.work_row, 10, input)
        return True

    def add_model_host_memory_usage(self, input):
        input = str(input)
        self.sheet.write(self.work_row, 11, input)
        return True

    def add_cross_entropy_loss(self,input):
        input = str(input)
        self.sheet.write(self.work_row, 12, input)
        return True
        
    def add_model_cuda_memory_usage(self,input):
        input = str(input)
        self.sheet.write(self.work_row, 13, input)
        return True

    def add_cpu_info(self,input):
        input = str(input)
        self.sheet.write(self.work_row, 14, input)
        return True

    def save_worksheet(self):
        self.new_book.save(self.database_path_location)
        return True

if __name__ == "__main__":
    cts = TestDatabaseUtil('../log/cpu_test_database.xls')
    cts.add_F1_score(0.34256)
    cts.save_worksheet()