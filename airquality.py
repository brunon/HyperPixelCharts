from datetime import datetime
"""
This script will generate a raw air quality chart for a single day
"""
import argparse

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter


parser = argparse.ArgumentParser()
parser.add_argument('--nfs', dest='nfs', required=True, help="NFS directory to store results in")
parser.add_argument('--date', dest='date', required=True, help='Which date to generate the chart for')
parser.add_argument('--chart', dest='chart', required=True, help='Path to the chart png file to generate')
args = parser.parse_args()
nfs_dir = args.nfs
date = datetime.strptime(args.date, '%Y-%m-%d').date()

df = pd.read_csv(f"{nfs_dir}/airquality.csv")
df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], format='%b %d %Y @ %H:%M:%S')

df = df.loc[df['TIMESTAMP'].dt.date == date]
df = df.set_index('TIMESTAMP').sort_index()

ax = df.plot(figsize=(10,6), xlabel='')
ax.legend(title=False, loc='upper left', fontsize=12)
ax.set_title(f"Air Quality on {date}", fontsize=14)

ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))

plt.savefig(f'{args.chart}',
    bbox_inches='tight',
    facecolor='w',
    edgecolor='w',
    orientation='landscape',
    transparent=False,
    dpi=100)
