import os
import time
import subprocess
from configparser import ConfigParser


cfg = ConfigParser()
cfg.read('config.ini')
path = os.getcwd()
#path = path + "/Code_bug/"
start = time.process_time()
subprocess.call("python3 " + path + "localcontext.py", shell=True)
local_file = cfg.get('globalcontext', 'test_local_data')
if os.path.exists(local_file):
    subprocess.call("python3 " + path + "globalcontext.py", shell=True)
else:
    print("Local Context Generation Error")
time_length = time.process_time() - start
print("Cost " + str(time_length) + " s to finish the model")
