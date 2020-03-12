import os
import sys
import time
import subprocess
path = os.getcwd()
path = path + "/Code_bug/"
start = time.process_time()
subprocess.call("python3 " + path + "localcontext.py", shell=True)
subprocess.call("python3 " + path + "globalcontext.py", shell=True)
time_length = time.process_time() - start
print("Cost " + str(time_length) + " s to finish the model")
