import os, sys, glob
import subprocess
from subprocess import DEVNULL, STDOUT

here = os.path.dirname(os.path.abspath(__file__))

files0 = glob.glob(here+"/../examples/*/*.py")
files1 = glob.glob(here+"/../examples/*.py")

files = files0 + files1

print("run files:", files)

for file_path in files:
    file_name = str(file_path.rsplit("/", maxsplit=1)[1])

    try:
        print("\033[0;33;40m Testing", file_name, end="...\r")
        subprocess.check_call(["python", file_path], stdout=DEVNULL, stderr=STDOUT)
    except subprocess.CalledProcessError:
        print("\033[0;31;40m Error in", file_name)
    else:
        print("\033[0;32;40m", file_name, "is correct")
print("\n")
