"""
Script to download statistics from a PiHole DNS adblocker

Ref https://pi-hole.net/
"""

import argparse
import requests
import pandas as pd
from datetime import datetime, timedelta

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--hostname', dest='hostname', required=True, help="PiHole FQDN")
parser.add_argument('--csv-dir', dest='csv_dir', required=True, help="CSV output path")
args = parser.parse_args()

data = requests.get(f"http://{args.hostname}/admin/api.php?getAllQueries").json().get('data')

df = pd.DataFrame(data=data, columns=['timestamp','type','requested_domain','client','answer',
                                      '1','2','3','4','5','6','7'])
df = df.drop(columns=['1','2','3','4','5','6','7']) # don't care about this extra crap

df['timestamp'] = df['timestamp'].map(lambda x: pd.Timestamp(int(x) * 1e9, tz='GMT').tz_convert('America/Montreal'))

yesterday = (pd.Timestamp.now() - pd.offsets.Day(1)).date()
df = df.loc[df['timestamp'].dt.date == yesterday]

df['hour'] = df['timestamp'].dt.to_period('h')
df = df.groupby('hour').size().to_frame('request_count')
df = df.assign(timestamp=df.index.strftime('%Y-%m-%d %H:%M'))

filepath = f"{args.csv_dir}/{yesterday.strftime('%Y-%m-%d')}.csv"
df[['timestamp','request_count']].to_csv(filepath, index=False)

