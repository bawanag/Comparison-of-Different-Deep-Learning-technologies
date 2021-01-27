
import os
import sys
import subprocess
def get_linux_version():
    result = "system version---- %s" % ", ".join(sys.version.split("\n"))
#     print(result)
    return result
 
 
def get_cpu_info():
    processor_cnt = 0
    cpu_model = ""
    result = ""
    f_cpu_info = open("/proc/cpuinfo")
    try:
            for line in f_cpu_info:
                    if (line.find("processor") == 0):
                            processor_cnt += 1
                    elif (line.find("model name") == 0):
                            if (cpu_model == ""):
                                    cpu_model = line.split(":")[1].strip()


            result = "cpu counts: %s, cpu model: %s" % (processor_cnt, cpu_model)
        #     print(result)

    finally:
            f_cpu_info.close()
    return result
 
def get_mem_info():
        mem_info = ""
        result = ""
        f_mem_info = open("/proc/meminfo")
        try:
                for line in f_mem_info:
                        if (line.find("MemTotal") == 0):
                                mem_info += line.strip()+ ", "
                        elif (line.find("SwapTotal") == 0):
                                mem_info += line.strip()
                                break
                result = "mem_info---- {:s}".format(mem_info)
                # print(result)
        finally:
                f_mem_info.close()
        return result
 
 
def get_disc_info():
    #disc_info = os.popen("df -h").read()
    #disc_info = subprocess.Popen("df -h", shell=True).communicate()[0]
    #print(disc_info)
    pipe = subprocess.Popen("df -h", stdout=subprocess.PIPE, shell=True)
    disc_info = pipe.stdout.read()
#     print(disc_info)
    return disc_info