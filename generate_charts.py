import os
import re
import sys
import yaml
import glob
import json
import locale
import logging
import argparse
from typing import Dict
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from subprocess import Popen, PIPE

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import requests
import influxdb_client


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
parser.add_argument('--pistat', dest='pistat', action='store_true', help="Generate CPU/RAM/DISK Chart")
parser.add_argument('--enviro', dest='enviro', action='store_true', help="Generate Enviro Charts")
parser.add_argument('--weather', dest='weather', action='store_true', help="Generate Weather Charts")
parser.add_argument('--life', dest='life', action='store_true', help="Generate Game of Life Stats Chart")
parser.add_argument('--alert-email', help='Send alert for missing data to provided email')
parser.add_argument('--check-last-updated', action='store_true', help="Check all charts are updated today/yesterday")
parser.add_argument('--influx-config', dest='influx_config', help="Influx DB Config file")
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


def pi_colors(hostnames: pd.Series, file_suffix: str = "") -> Dict[str, str]:
    filename = f"picolors{file_suffix}.json"
    picolors_json = os.path.join(nfs_dir, filename)
    if not os.path.exists(picolors_json):
        pi_colors = {}
    else:
        with open(picolors_json, 'r') as f:
            pi_colors = json.load(f)

    # if any new Pi host is added, need to rebuild colors dict
    if any(h not in pi_colors for h in hostnames.unique()):
        logging.info(f"Regenerating {filename} file, had %s needed %s", ','.join(sorted(pi_colors.keys())), ','.join(sorted(hostnames.unique())))
        colors_iter = iter(mpl.rcParams['axes.prop_cycle'])
        all_hostnames = set(list(pi_colors.keys()) + list(hostnames.values))
        sorted_hostnames = sorted(all_hostnames, key=lambda h: h.lower())
        pi_colors = {h: color for h in sorted_hostnames for color in next(colors_iter).values()}
        with open(picolors_json, 'w') as f:
            json.dump(pi_colors, f)

    return pi_colors


def save_update_ts(chart: str, last_updated: datetime):
    file_path = os.path.join(nfs_dir, 'last_updated.json')
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            last_updated_dict = json.load(f)
    else:
        last_updated_dict = {}

    last_updated_dict[chart] = last_updated.strftime('%Y-%m-%d %H:%M')
    with open(file_path, 'w') as f:
        json.dump(last_updated_dict, f)


def send_alert_email(subject, body, email):
    msg = MIMEText(body)
    msg['From'] = email
    msg['To'] = email
    msg['Subject'] = subject
    p = Popen(['/usr/sbin/sendmail', '-t', '-oi'], stdin=PIPE)
    p.communicate(msg.as_bytes())


def check_last_updated(alert_email):
    today = datetime.today().date()
    yesterday = today - timedelta(days=1)
    charts_missing_data = []
    with open(os.path.join(nfs_dir, 'last_updated.json')) as f:
        last_updated = json.load(f)
        for chart, update_ts in last_updated.items():
            logging.info("Data for chart %s last updated on %s", chart, update_ts)
            update_ts = datetime.strptime(update_ts, '%Y-%m-%d %H:%M')
            if update_ts.date() not in [today, yesterday]:
                logging.warn("Chart %s is out of date!", chart)
                charts_missing_data.append(chart)

    if charts_missing_data and alert_email:
        send_alert_email(
                subject='HyperPixel Charts are not updated',
                body=f"These charts have not been updated in 48h: {', '.join(charts_missing_data)}",
                email=alert_email
                )


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
        save_update_ts('covid', update_ts)
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


def concat_csv_files(glob_path, **kwargs):
    df_list = []
    for csv in glob.glob(glob_path):
        df = pd.read_csv(csv, **kwargs)
        df_list.append(df)
    return pd.concat(df_list, sort=False)


def generate_bandwitdh_chart():
    df_list = []
    for csv in glob.glob(f"{nfs_dir}/bandwidth/*.csv"):
        df = pd.read_csv(csv)
        filename = os.path.basename(csv).split('.')[0]
        df_list.append((filename,df))

    update_ts = []
    for filename, df in df_list:
        df['download'] /= 1e6
        df['upload'] /= 1e6
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%b %d %Y @ %H:%M:%S')
        df['hour'] = df['timestamp'].dt.to_period('h')
        this_update_ts = df['timestamp'].max()
        update_ts.append(this_update_ts)
        df = df.groupby('hour').mean(numeric_only=True)
        df = df.rolling(24).mean()
        df = df.groupby(pd.Grouper(freq='D')).mean()
        df = df.dropna()
        if df.empty:
            continue
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
        ax.set_title(f"Internet Bandwidth Monitor - {filename.replace('_',' ')} (updated {this_update_ts.strftime('%b %d %H:%M')})", fontsize=14)
        ax2 = ax.twinx()
        df['upload'].plot(
            ax=ax2,
            color='#ff7f0e',
            linewidth=2
        )
        ax2.set_ylabel('Upload Speed (Mbps)', color='#ff7f0e', weight='bold', fontsize=12)
        ax2.set_ylim(upload_lim)
        save_image(f"bandwidth-{filename}.png")

    save_update_ts('bandwidth', max(update_ts))

def generate_pihole_chart():
    df = concat_csv_files(f"{nfs_dir}/pihole/*.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M')
    df.sort_values(by='timestamp', ascending=True, inplace=True)
    update_ts = df['timestamp'].max()
    save_update_ts('pihole', update_ts)
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


def generate_cpu_temp_chart(alert_email: str):
    df = concat_csv_files(f"{nfs_dir}/pitemp/*.csv", parse_dates=['timestamp'])
    
    # check if all hosts have up-to-date information pushed
    today = datetime.today().date()
    yesterday = today - timedelta(days=1)
    updates = df.groupby('hostname')['timestamp'].max().to_frame('max_timestamp')
    updated_last_24h = updates['max_timestamp'].dt.date.isin([today,yesterday])
    missing_data = updates.loc[~updated_last_24h].index.values
    if len(missing_data) > 0 and alert_email:
        send_alert_email(
                subject=f"Missing RaspberryPi data on {today}",
                body=f"Please investigate missing RaspberryPi data for hosts: {', '.join(missing_data)}",
                email=alert_email
                )

    update_ts = df['timestamp'].max()
    save_update_ts('temp', update_ts)

    hostnames = df['hostname'].unique()
    colors = pi_colors(df['hostname'])

    # keep one data point per Pi per day
    df = df.groupby([pd.Grouper(freq='D', key='timestamp'), 'hostname'])['temp'].agg(['mean','std'])

    ax = df['mean'].unstack('hostname').plot(
        figsize=(10,6),
        linewidth=2,
        xlabel='',
        color=[colors[h] for h in df['mean'].unstack('hostname').columns]
    )
    for h in hostnames:
        y = df['mean'].unstack('hostname')[h]
        err = df['std'].unstack('hostname')[h]
        plt.fill_between(y.index,
                         y - err,
                         y + err,
                         color=colors[h],
                         alpha=.2)
    ax.legend(title=False, loc='upper left', fontsize=12)
    ax.set_title(f"Raspberry Pi CPU Temperature (updated {update_ts.strftime('%b %d %H:%M')})", fontsize=14)
    save_image('pitemp.png')


def generate_air_quality_chart():
    df = pd.read_csv(f"{nfs_dir}/airquality.csv")
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], format='%b %d %Y @ %H:%M:%S')
    update_ts = df['TIMESTAMP'].max()
    save_update_ts('airquality', update_ts)

    # compute peak value in a given hour
    df = df.set_index('TIMESTAMP').sort_index()
    df = df.groupby(pd.Grouper(freq='H')).max()

    # reshape the DF to one row per PM type
    df = df.stack().to_frame('PM_VALUE')
    df.index = df.index.set_names(['TIMESTAMP','PM_TYPE'])
    df = df.reset_index()

    pm_types = df.PM_TYPE.unique()
    colors_iter = iter(mpl.rcParams['axes.prop_cycle'])
    colors = {pm: next(colors_iter)['color'] for pm in pm_types}

    # compute average value per day (with standard deviation for error bars)
    df = df.groupby([pd.Grouper(freq='D', key='TIMESTAMP'), 'PM_TYPE'])['PM_VALUE'].agg(['mean','std'])

    ax = df['mean'].unstack('PM_TYPE')[pm_types].plot(figsize=(10,6), xlabel='', linewidth=2, color=colors.values())
    for p in pm_types:
        y = df['mean'].unstack('PM_TYPE')[p]
        err = df['std'].unstack('PM_TYPE')[p]
        plt.fill_between(y.index,
                         (y - err).clip(lower=0),
                         y + err,
                         color=colors[p],
                         alpha=.2)

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
    update_ts = df['date'].max().astimezone('America/Montreal')
    df['date'] = df['date'].dt.tz_localize(None)
    df['hour'] = df['date'].dt.to_period('h')
    save_update_ts('iperf', update_ts)

    df = df[['client','hour','rcvd_mbps']]
    clients = df['client']
    colors = pi_colors(df['client'])

    client_avg = df.groupby('client', as_index=False)['rcvd_mbps'].mean()
    client_low = client_avg.loc[client_avg['rcvd_mbps'] < 100, 'client']
    client_high = set(clients) - set(client_low)
    df_low = df.loc[df['client'].isin(client_low)]
    df_high = df.loc[df['client'].isin(client_high)]

    df_low = df_low.set_index(['hour','client'])['rcvd_mbps'].unstack('client')
    df_high = df_high.set_index(['hour','client'])['rcvd_mbps'].unstack('client')
    low_max = df_low.max().max()

    df_low = df_low.rolling('1D', center=True).agg(['mean','std']).stack('client')
    df_high = df_high.rolling('1D', center=True).agg(['mean','std']).stack('client')

    ax1 = df_low['mean'].unstack('client').plot(
            figsize=(10,6),
            linewidth=2,
            xlabel='',
            color=[colors[c] for c in df_low['mean'].unstack('client').columns]
    )
    for c in client_low:
        y = df_low['mean'].unstack('client')[c]
        err = df_low['std'].unstack('client')[c]
        plt.fill_between(y.index,
                         y - err,
                         y + err,
                         color=colors[c],
                         alpha=.2)

    ax1.legend(title=False, loc='lower left', fontsize=12)
    ax1.set_title(f"Internal Network Performance (Mbits); updated {update_ts.strftime('%b %d %H:%M')}", fontsize=14)
    ax1.yaxis.grid()
    ax1.set_ylim((0, low_max * 1.5))
    ax2 = ax1.twinx()
    df_high['mean'].unstack('client').plot(
        ax=ax2,
        linewidth=2,
        color=[colors[c] for c in df_high['mean'].unstack('client').columns]
    )
    for c in client_high:
        y = df_high['mean'].unstack('client')[c]
        err = df_high['std'].unstack('client')[c]
        plt.fill_between(y.index,
                         y - err,
                         y + err,
                         color=colors[c],
                         alpha=.2)
    
    ax2.legend(title=False, loc='lower right', fontsize=12)
    ax2.set_ylim((0,1000))
    save_image('iperf.png')

def generate_pistat_chart():
    df = concat_csv_files(f'{nfs_dir}/pistat/*.csv', parse_dates=['timestamp'])
    update_ts = df['timestamp'].max()
    save_update_ts('pistats', update_ts)

    colors = pi_colors(df['hostname'])

    # keep one data point per Pi per day
    df = df.groupby([pd.Grouper(freq='D', key='timestamp'), 'hostname']).agg(['mean','std'])

    # reshape DF to have one column per Pi
    df = df.unstack('hostname')

    for stat in ['cpu','ram','disk']:
        y = df[stat]['mean']
        hosts = y.columns
        ax = y.plot(
            figsize=(10,6),
            linewidth=2,
            color=[colors[h] for h in hosts],
            xlabel='',
        )
        for h in hosts:
            err = df[stat]['std'][h]
            plt.fill_between(df.index,
                             y[h] - err,
                             y[h] + err,
                             alpha=.2,
                             color=colors[h])
        ax.legend(title=False, loc='upper left', fontsize=12)
        ax.set_title(f"Raspberry Pi {stat.capitalize()} Usage % (updated {update_ts.strftime('%b %d %H:%M')})", fontsize=14)
        save_image(f'pi{stat}.png')


def _query_influx_db(query: str) -> pd.DataFrame:
    with open(args.influx_config) as f:
        config = yaml.safe_load(f)
    client = influxdb_client.InfluxDBClient(url=config['influx']['url'], token=config['influx']['token'], org=config['influx']['org'])
    query_api = client.query_api()
    df_list = query_api.query_data_frame(query)
    df = pd.concat(df_list, sort=False) if isinstance(df_list, list) else df_list
    df = df.drop(columns=['result','table'])
    return df


def generate_enviro_charts():
    query = f"""
from(bucket:"enviro")
    |> range(start: 1970-01-01T00:00:00Z, stop: now())
    |> aggregateWindow(every: 1h, fn: mean, createEmpty: false)
    |> pivot(rowKey: ["_time","_measurement"], columnKey: ["device"], valueColumn: "_value")
    |> drop(columns: ["_start","_stop","_field"])
    """.strip()
    df = _query_influx_db(query)
    update_ts = df['_time'].max().astimezone('America/Montreal')
    save_update_ts('enviro', update_ts)

    colors = pi_colors(pd.Series(df.drop(columns=['_time','_measurement']).columns.values), file_suffix="-enviro")
    stats_to_ignore = ['color_temperature','luminance']

    for stat in (s for s in df['_measurement'].unique() if s not in stats_to_ignore and not s.startswith('gas_')):
        stat_df = df.loc[df['_measurement'] == stat].drop(columns=['_measurement']).set_index('_time')
        hosts = stat_df.columns
        stat_df = stat_df.rename(columns={
            c: c.replace('-',' ').replace('_',' ').title()
            for c in stat_df.columns
        })
        ax = stat_df.plot(
            figsize=(10,6),
            linewidth=2,
            color=[colors[h] for h in hosts],
            xlabel=''
        )
        ax.legend(title=False, loc='upper left', fontsize=12)
        title = "Air Quality Index" if stat == 'aqi' else stat.capitalize()

        ax.set_title(f"Enviro Indoor: {title} (updated {update_ts.strftime('%b %d %H:%M')})", fontsize=14)
        save_image(f'enviro-{stat}.png')

    gas_stats = [s for s in df['_measurement'].unique() if s.startswith('gas_')]
    if gas_stats:
        gas_df = df.loc[df['_measurement'].isin(gas_stats)]
        gas_df = gas_df.assign(_measurement=gas_df['_measurement'].map({
            'gas_resistance': 'CO2',
            'gas_nh3': 'NH3',
            'gas_no2': 'NO2'
        }))
        gas_df = gas_df.set_index(['_time','_measurement'])
        gas_df = gas_df.unstack('_measurement')
        gas_df.columns = gas_df.columns.to_flat_index()
        hosts = [h for h,_ in gas_df.columns]
        gas_df.columns = [h.replace('-',' ').replace('_',' ').title() + " - " + g for h,g in gas_df.columns]
        ax = gas_df.plot(
			figsize=(10,6),
			linewidth=2,
            color=[colors[h] for h in hosts],
			xlabel=''
    	)
        ax.legend(title=False, loc='upper left', fontsize=12)
        ax.set_title(f"Enviro Indoor: Gases (updated {update_ts.strftime('%b %d %H:%M')})", fontsize=14)
        save_image('enviro-gas.png')


def generate_weather_charts(alert_email: str):
    tz = "America/Montreal"
    df_list = []
    for bucket, tag in [('enviro-urban', 'device'), ('weather', 'source')]:
        query = f"""
    from(bucket:"{bucket}")
        |> range(start: 1970-01-01T00:00:00Z, stop: now())
        |> aggregateWindow(every: 1h, fn: mean, createEmpty: false)
        |> pivot(rowKey: ["_time","_measurement"], columnKey: ["{tag}"], valueColumn: "_value")
        |> drop(columns: ["_start","_stop","_field"])
        """.strip()
        df = _query_influx_db(query)
        time_since_update = (pd.Timestamp.now(tz=tz) - df['_time'].max().astimezone(tz)).total_seconds()
        hours_since_update = time_since_update / 60 / 60
        if hours_since_update > 24 and alert_email:
            send_alert_email(
                    subject='Stale Weather Data',
                    body=f"Weather data from {tag} is {hours_since_update} hours old",
                    email=alert_email
                    )
        df_list.append(df)

    df = df_list[0]
    for x in df_list[1:]:
        df = df.merge(x, on=['_time','_measurement'], how='outer')

    update_ts = df['_time'].max().astimezone(tz)
    save_update_ts('weather', update_ts)

    # convert forecast values into temperature fields in their own columns
    forecasts = [f for f in df['_measurement'].unique() if f.startswith("forecast-")]
    for f in forecasts: df = df.assign(**{f: np.NaN})
    non_forecast_df = df.loc[~df['_measurement'].isin(forecasts)]
    forecast_df = df.loc[df['_measurement'].isin(forecasts)]
    forecast_temp = forecast_df.drop(columns=['_time','_measurement']).dropna(axis=1, how='all').iloc[:,0]
    for f in forecasts:
        forecast_df.loc[forecast_df['_measurement'] == f, f] = forecast_temp

    forecast_df = forecast_df.assign(**{'_measurement': 'temperature'})
    forecast_df = forecast_df.drop(columns=[forecast_temp.name])

    df = pd.concat([non_forecast_df, forecast_df], sort=False)
    colors = pi_colors(pd.Series(df.drop(columns=['_time','_measurement']).columns.values), file_suffix="-enviro")

    for stat in ['humidity', 'pressure', 'temperature']:
        stat_df = df.loc[df['_measurement'] == stat].drop(columns=['_measurement'])
        stat_df = stat_df.sort_values('_time').groupby(pd.Grouper(freq='H', key='_time')).mean(numeric_only=True).dropna(how='all')
        stat_df = stat_df.dropna(axis=1, how='all')
        stat_df = stat_df.interpolate()
        hosts = stat_df.columns
        stat_df = stat_df.rename(columns={
            c: (
                'EC - Current' if c.startswith('ec-')
                else f"EC - {c[-1]} Day Forecast" if c.startswith('forecast-')
                else c.replace('-',' ').replace('_',' ').title()
                )
            for c in stat_df.columns
        })
        ax = stat_df.plot(
            figsize=(10,6),
            linewidth=2,
            color=[colors[h] for h in hosts],
            xlabel=''
        )
        ax.legend(title=False, loc='upper left', fontsize=12)
        title = stat.capitalize()

        ax.set_title(f"Weather: {title} (updated {update_ts.strftime('%b %d %H:%M')})", fontsize=14)
        save_image(f'weather-{stat}.png')


def generate_game_of_life_chart():
    df = pd.read_csv(f"{nfs_dir}/gameoflife.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%b %d %Y @ %H:%M:%S')
    update_ts = df['timestamp'].max()
    save_update_ts('gameoflife', update_ts)

    colors_iter = iter(mpl.rcParams['axes.prop_cycle'])
    duration_color = next(colors_iter)['color']
    iterations_color = next(colors_iter)['color']

    df = df.drop(columns=['all_dead'])
    df = df.groupby(pd.Grouper(freq='H', key='timestamp')).agg(['mean','std'])
    df = df.dropna(how='all')

    y1 = df['duration']['mean']
    err = df['duration']['std']
    ax1 = y1.plot(figsize=(10,6), color=duration_color, xlabel='', linewidth=3)
    plt.fill_between(y1.index, y1 - err, y1 + err, alpha=.2, color=duration_color)
    ax1.set_ylabel("Duration (sec)", fontsize=12)
    ax1.set_ylim((0, max(y1.max(), 30)))
    ax1.set_title(f"Game of Life Stats (updated {update_ts.strftime('%b %d %H:%M')})", fontsize=14)
    ax1.legend(["Duration (sec)"], title=False, loc='lower right', fontsize=12)

    ax2 = ax1.twinx()
    y2 = df['iterations']['mean']
    y2.plot(ax=ax2, xlabel='', color=iterations_color, linewidth=3)
    ax2.set_ylim((0, y2.max() * 1.1))
    ax2.set_ylabel("Iterations", fontsize=12)
    ax2.legend(["Iterations"], title=False, loc='upper left', fontsize=12)

    save_image('gameoflife.png')


if __name__ == '__main__':
    if args.bandwidth: generate_bandwitdh_chart()

    if args.pihole: generate_pihole_chart()

    if args.temp: generate_cpu_temp_chart(args.alert_email)

    if args.airquality: generate_air_quality_chart()

    if args.covid:
        update_ts = requests.get("http://api.opencovid.ca/version").json().get('version')
        update_ts = datetime.strptime(update_ts, '%Y-%m-%d %H:%M %Z')
        data = download_covid_data()
        for filename, (data_key, datecol, valuecol, title, format_fn) in COVID_CHART_CONFIG.items():
            df = generate_covid_df(data[data_key], datecol, valuecol)
            generate_covid_chart(df, title, filename, format_fn, update_ts, save_chart=True)

    if args.iperf: generate_iperf_chart()

    if args.pistat: generate_pistat_chart()

    if args.check_last_updated: check_last_updated(args.alert_email)

    if args.enviro: generate_enviro_charts()

    if args.weather: generate_weather_charts(args.alert_email)

    if args.life: generate_game_of_life_chart()
