"""
Script to monitor CPU and RAM Usage % and output to a CSV file

Required module:
    pip3 install psutil
"""

import os
import csv
import socket
import psutil
import shutil
import argparse
from datetime import datetime
import subprocess

script_dir = os.path.dirname(__file__)

parser = argparse.ArgumentParser()
parser.add_argument('--csv', required=True, help="CSV output dir")
parser.add_argument('--polling', required=False, type=int, default=4, help="CPU polling delay (in seconds)")
args = parser.parse_args()

cpu_usage = round(psutil.cpu_percent(args.polling), 1)
ram_usage = round(psutil.virtual_memory()[2], 1)
disk_usage = round(psutil.disk_usage('/').percent, 1)
sd_card_writes = int(subprocess.check_output(["/bin/sh", os.path.join(script_dir, "sd_card_writes.sh")], text=True).strip())
hostname = socket.gethostname()
ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
csv_dir = args.csv
csv_file = os.path.join(csv_dir, hostname + '.csv')

with open(csv_file, 'a') as f:
    writer = csv.writer(f)
    writer.writerow((hostname, cpu_usage, ram_usage, disk_usage, ts, sd_card_writes))

