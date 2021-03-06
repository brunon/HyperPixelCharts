import os
import re
import glob
import locale
import logging
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import requests


# Setup basic logging for crontab log file
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger('HyperPixelCharts')
locale.setlocale(locale.LC_ALL, "en_CA.UTF-8")

parser = argparse.ArgumentParser()
parser.add_argument('--nfs', dest='nfs', required=True, help="NFS directory to store results in")
parser.add_argument('--bandwidth', dest='bandwidth', action='store_true', help="Generate bandwidth chart")
parser.add_argument('--covid', dest='covid', action='store_true', help="Generate COVID charts")
parser.add_argument('--pihole', dest='pihole', action='store_true', help="Generate PiHole chart")
parser.add_argument('--temp', dest='temp', action='store_true', help="Generate CPU Temperature chart")
parser.add_argument('--airquality', dest='airquality', action='store_true', help="Generate Air Quality Chart")
parser.add_argument('--iperf', dest='iperf', action='store_true', help="Generate iPerf Chart")
args = parser.parse_args()
nfs_dir = args.nfs


def format_k(x, pos):
    return "%1.1fk" % (x * 1e-3)


def iqr_outlier_removal(df, colname, whisker_width=1.5, interpolate=None):
    q1, q3 = df[colname].quantile([0.25,0.75])
    iqr = q3 - q1
    outlier = (df[colname] < (q1 - whisker_width * iqr)) | (df[colname] > (q3 + whisker_width * iqr))
    df.loc[outlier, colname] = np.nan
    if interpolate: df[colname].interpolate(interpolate, inplace=True)


def save_image(filename):
    logging.info('Generating chart: %s', filename)
    plt.savefig(f'{nfs_dir}/hyperpixel/{filename}',
                bbox_inches='tight',
                facecolor='w',
                edgecolor='w',
                orientation='landscape',
                transparent=False,
                dpi=100)
    plt.close()


def generate_covid_df(json, datecol, valuecol):
    df = pd.DataFrame(data=json)
    df = df.astype({datecol: 'datetime64'})
    df = df.sort_values(by=datecol, ascending=True)

    # convert to 7d/30d rolling windows
    df = df.assign(rolling7=df[valuecol].rolling(7).mean(),
                   rolling30=df[valuecol].rolling(30).mean())
    df = df[[datecol, valuecol, 'rolling7', 'rolling30']]

    # focus on last 1yr only
    df = df.loc[df[datecol] >= (pd.Timestamp.now() - pd.offsets.MonthBegin(12))]

    return df.set_index([datecol])


def generate_covid_chart(df, title, filename, y_format_fn=None, update_ts=None, save_chart=True):
    ax = df[['rolling7','rolling30']].plot(
        kind='line',
        linewidth=2,
        xlabel='',
        figsize=(10, 6)
    )
    ax.legend(['7d moving avg', '30d moving avg'])
    if y_format_fn: ax.yaxis.set_major_formatter(plt.FuncFormatter(y_format_fn))
    if update_ts:
        title += f" (updated {update_ts.strftime('%b %d %H:%M')})"
    ax.set_title(title, fontsize=14)
    if save_chart: save_image(filename)
    return ax


def download_covid_data():
    return requests.get("http://api.opencovid.ca/timeseries?loc=QC").json()


COVID_CHART_CONFIG = {
    'cases.png': ('cases', 'date_report', 'cases', 'New Cases', format_k),
    'vaccines.png': ('avaccine', 'date_vaccine_administered', 'avaccine', 'Vaccines Administered', format_k),
    'mortality.png': ('mortality', 'date_death_report', 'deaths', 'Deaths', None),
    'tests.png': ('testing', 'date_testing', 'testing', 'Tests', format_k),
    'active.png': ('active', 'date_active', 'active_cases', 'Active Cases', format_k),
}


def generate_bandwitdh_chart():
    df_list = []
    for csv in glob.glob(f"{nfs_dir}/bandwidth/*.csv"):
        df = pd.read_csv(csv)
        filename = os.path.basename(csv).split('.')[0]
        df_list.append((filename,df))

    for filename, df in df_list:
        df['download'] /= 1e6
        df['upload'] /= 1e6
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%b %d %Y @ %H:%M:%S')
        df['hour'] = df['timestamp'].dt.to_period('h')
        update_ts = df['timestamp'].max()
        df = df.groupby('hour').mean()
        df = df.rolling(24).mean()
        df = df.groupby(pd.Grouper(freq='D')).mean()
        iqr_outlier_removal(df, 'download', interpolate='linear')
        iqr_outlier_removal(df, 'upload', interpolate='linear')
        download_lim = (np.floor(df['download'].min()), np.ceil(df['download'].max()))
        upload_lim = (np.floor(df['upload'].min()), np.ceil(df['upload'].max()))
        ax = df['download'].plot(
            figsize=(10,6),
            xlabel='',
            color='#1f77b4',
            linewidth=2
        )
        ax.set_ylabel('Download Speed (Mbps)', color='#1f77b4', weight='bold', fontsize=12)
        ax.set_ylim(download_lim)
        ax.set_title(f"Internet Bandwidth Monitor - {filename} (updated {update_ts.strftime('%b %d %H:%M')})", fontsize=14)
        ax2 = ax.twinx()
        df['upload'].plot(
            ax=ax2,
            color='#ff7f0e',
            linewidth=2
        )
        ax2.set_ylabel('Upload Speed (Mbps)', color='#ff7f0e', weight='bold', fontsize=12)
        ax2.set_ylim(upload_lim)
        save_image(f"bandwidth-{filename}.png")


def generate_pihole_chart():
    df_list = []
    for csv in glob.glob(f"{nfs_dir}/pihole/*.csv"):
        df = pd.read_csv(csv)
        df_list.append(df)

    df = pd.concat(df_list, sort=False)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M')
    df.sort_values(by='timestamp', ascending=True, inplace=True)
    update_ts = df['timestamp'].max()
    df = df.groupby(pd.Grouper(freq='D', key='timestamp')).agg({'request_count': 'mean'})
    df['request_count'].interpolate(inplace=True)
    df['rolling'] = df['request_count'].copy()
    iqr_outlier_removal(df, 'rolling', interpolate='linear')
    df['rolling'] = df['rolling'].rolling(7).mean()
    ax = df[['request_count','rolling']].plot(
        figsize=(10,6),
        xlabel='',
        linewidth=2
    )
    ax.set_title(f"PiHole DNS Queries (updated {update_ts.strftime('%b %d %H:%M')})", fontsize=14)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(format_k))
    ax.legend(['Average DNS Queries / day', '7d moving avg'], fontsize=12)
    save_image('pihole.png')


def generate_cpu_temp_chart():
    df = pd.read_csv(f"{nfs_dir}/pitemp.csv", parse_dates=['timestamp'])
    update_ts = df['timestamp'].max()

    # keep one data point per Pi per day
    df = df.groupby([pd.Grouper(freq='D', key='timestamp'), 'hostname']).agg({'temp':'mean'})

    # reshape DF to have one column per Pi
    df = df.unstack('hostname')['temp']

    ax = df.plot(
        figsize=(10,6),
        linewidth=2,
        xlabel=''
    )
    ax.legend(title=False, loc='upper left', fontsize=12)
    ax.set_title(f"Raspberry Pi CPU Temperature (updated {update_ts.strftime('%b %d %H:%M')})", fontsize=14)
    save_image('pitemp.png')


def generate_air_quality_chart():
    df = pd.read_csv(f"{nfs_dir}/airquality.csv")
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], format='%b %d %Y @ %H:%M:%S')
    update_ts = df['TIMESTAMP'].max()
    df = df.groupby(pd.Grouper(freq='D', key='TIMESTAMP')).max() # keep highest value per day
    ax = df.plot(
            figsize=(10,6),
            xlabel=''
    )
    ax.legend(title=False, loc='upper left', fontsize=12)
    ax.set_title(f"Air Quality Monitor (updated {update_ts.strftime('%b %d %H:%M')})", fontsize=14)
    save_image('airquality.png')

def generate_iperf_chart():
    df_list = []
    file_pat = re.compile(f"{nfs_dir}/iperf/(?P<client>[^.]+)\.csv")
    for csv in glob.glob(f"{nfs_dir}/iperf/*.csv"):
        m = file_pat.match(csv)
        df = pd.read_csv(csv)
        df['client'] = m.group('client')
        df_list.append(df)
    df = pd.concat(df_list, sort=False)

    df['date'] = pd.to_datetime(df['date'], format='%a, %d %b %Y %H:%M:%S %Z')
    df['date'] = df['date'].dt.tz_localize(None)
    df['hour'] = df['date'].dt.to_period('h')
    update_ts = df['date'].max()

    df = df[['client','hour','rcvd_mbps']]
    colors = iter(mpl.rcParams['axes.prop_cycle'])
    clients = df['client'].unique()
    colors = {client: color for client in clients for color in next(colors).values()}

    client_avg = df.groupby('client', as_index=False)['rcvd_mbps'].mean()
    client_low = client_avg.loc[client_avg['rcvd_mbps'] < 100, 'client']
    df_low = df.loc[df['client'].isin(client_low)]
    df_high = df.loc[~df['client'].isin(client_low)]

    df_low = df_low.set_index(['hour','client'])['rcvd_mbps'].unstack('client')
    df_high = df_high.set_index(['hour','client'])['rcvd_mbps'].unstack('client')
    low_max = df_low.max().max()

    df_low = df_low.rolling(3, center=True).mean().iloc[1:]
    df_high = df_high.rolling(3, center=True).mean().iloc[1:]

    ax1 = df_low.plot(
            figsize=(10,6),
            xlabel='',
            color=[colors[c] for c in df_low.columns]
    )
    ax1.legend(title=False, loc='lower left', fontsize=12)
    ax1.set_title(f"Internal Network Performance (Mbits); updated {update_ts.strftime('%b %d %H:%M')}", fontsize=14)
    ax1.yaxis.grid()
    ax1.set_ylim((0, low_max))
    ax2 = ax1.twinx()
    df_high.plot(
        ax=ax2,
        color=[colors[c] for c in df_high.columns]
    )
    ax2.legend(title=False, loc='lower right', fontsize=12)
    ax2.set_ylim((0,1000))
    save_image('iperf.png')

if __name__ == '__main__':
    if args.bandwidth: generate_bandwitdh_chart()

    if args.pihole: generate_pihole_chart()

    if args.temp: generate_cpu_temp_chart()

    if args.airquality: generate_air_quality_chart()

    if args.covid:
        update_ts = requests.get("http://api.opencovid.ca/version").json().get('version')
        update_ts = datetime.strptime(update_ts, '%Y-%m-%d %H:%M %Z')
        data = download_covid_data()
        for filename, (data_key, datecol, valuecol, title, format_fn) in COVID_CHART_CONFIG.items():
            df = generate_covid_df(data[data_key], datecol, valuecol)
            generate_covid_chart(df, title, filename, format_fn, update_ts, save_chart=True)

    if args.iperf: generate_iperf_chart()

