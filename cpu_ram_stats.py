"""
Script to monitor CPU and RAM Usage % and output to a CSV file

Required module:
    pip3 install psutil
"""

import csv
import socket
import psutil
import shutil
import argparse
from datetime import datetime
from filelock import FileLock

parser = argparse.ArgumentParser()
parser.add_argument('--csv', required=True, help="CSV output path")
parser.add_argument('--polling', required=False, type=int, default=4, help="CPU polling delay (in seconds)")
args = parser.parse_args()

cpu_usage = round(psutil.cpu_percent(args.polling), 1)
ram_usage = round(psutil.virtual_memory()[2], 1)
disk_usage = round(psutil.disk_usage('/').percent, 1)
hostname = socket.gethostname()
ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

lock_file = args.csv + '.lock'
lock = FileLock(lock_file, timeout=10)
with lock:
    with open(args.csv, 'a') as f:
        writer = csv.writer(f)
        writer.writerow((hostname, cpu_usage, ram_usage, disk_usage, ts))

