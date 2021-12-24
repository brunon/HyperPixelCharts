import glob
import locale
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests


locale.setlocale(locale.LC_ALL, "en_CA.UTF-8")

parser = argparse.ArgumentParser()
parser.add_argument('--nfs', dest='nfs', required=True, help="NFS directory to store results in")
parser.add_argument('--bandwidth', dest='bandwidth', action='store_true', help="Generate bandwidth chart")
parser.add_argument('--covid', dest='covid', action='store_true', help="Generate COVID charts")
parser.add_argument('--pihole', dest='pihole', action='store_true', help="Generate PiHole chart")
parser.add_argument('--temp', dest='temp', action='store_true', help="Generate CPU Temperature chart")
args = parser.parse_args()
nfs_dir = args.nfs


def format_k(x, pos):
    return "%1.1fk" % (x * 1e-3)


def save_image(filename):
    print(f'Generating chart: {filename}')
    plt.savefig(f'{nfs_dir}/hyperpixel/{filename}',
                bbox_inches='tight',
                facecolor='w',
                edgecolor='w',
                orientation='landscape',
                transparent=False,
                dpi=100)


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
    csv_file = f"{nfs_dir}/bandwidth_monitor.csv"
    df = pd.read_csv(csv_file)
    df['download'] /= 1e6
    df['upload'] /= 1e6
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%b %d %Y @ %H:%M:%S')
    update_ts = df['timestamp'].max()
    df = df.set_index('timestamp')
    df = df.rolling(5).mean()
    download_lim = (np.floor(df.download.min()), np.ceil(df.download.max()))
    upload_lim = (np.floor(df.upload.min()), np.ceil(df.upload.max()))
    ax = df.download.plot(
        figsize=(10,6),
        xlabel='',
        color='#1f77b4',
        linewidth=2
    )
    ax.set_ylabel('Download Speed (Mbps)', color='#1f77b4', weight='bold', fontsize=12)
    ax.set_ylim(download_lim)
    ax.set_title(f"Internet Bandwidth Monitor (updated {update_ts.strftime('%b %d %H:%M')})", fontsize=14)
    ax2 = ax.twinx()
    df.upload.plot(
        ax=ax2,
        color='#ff7f0e',
        linewidth=2
    )
    ax2.set_ylabel('Upload Speed (Mbps)', color='#ff7f0e', weight='bold', fontsize=12)
    ax2.set_ylim(upload_lim)
    save_image('bandwidth.png')


def generate_pihole_chart():
    df_list = []
    for csv in glob.glob(f"{nfs_dir}/pihole/*.csv"):
        df = pd.read_csv(csv, parse_dates=['timestamp'])
        df_list.append(df)

    df = pd.concat(df_list, sort=False)
    update_ts = df['timestamp'].max()
    df = df.set_index('timestamp')
    ax = df['request_count'].plot(
        figsize=(10,6),
        xlabel='',
        linewidth=2
    )
    ax.set_title(f"PiHole DNS Queries (updated {update_ts.strftime('%b %d %H:%M')})", fontsize=14)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(format_k))
    save_image('pihole.png')


def generate_cpu_temp_chart():
    df = pd.read_csv(f"{nfs_dir}/pitemp.csv", parse_dates=['timestamp'])
    update_ts = df['timestamp'].max()

    # keep one data point per Pi per hour
    df['hour'] = df['timestamp'].dt.to_period('h')

    # reshape DF to have one column per Pi
    df = df.set_index(['hour','hostname']).unstack('hostname')['temp']

    df = df.ffill() # fill gaps in timing of measurement

    ax = df.plot(
        figsize=(10,6),
        xlabel=''
    )
    ax.legend(title=False, loc='upper left', fontsize=12)
    ax.set_title(f"Raspberry Pi CPU Temperature (updated {update_ts.strftime('%b %d %H:%M')})", fontsize=14)
    save_image('pitemp.png')


if __name__ == '__main__':
    if args.bandwidth: generate_bandwitdh_chart()

    if args.pihole: generate_pihole_chart()

    if args.temp: generate_cpu_temp_chart()

    if args.covid:
        update_ts = requests.get("http://api.opencovid.ca/version").json().get('version')
        update_ts = datetime.strptime(update_ts, '%Y-%m-%d %H:%M %Z')
        data = download_covid_data()
        for filename, (data_key, datecol, valuecol, title, format_fn) in COVID_CHART_CONFIG.items():
            df = generate_covid_df(data[data_key], datecol, valuecol)
            generate_covid_chart(df, title, filename, format_fn, update_ts, save_chart=True)

