#!/bin/bash

script_dir=$(dirname $(readlink -f $0))
dir=${1:-$script_dir}

now=$(date +"%Y-%m-%d %H:%M:%S")
temp=$(vcgencmd measure_temp | egrep -o '[0-9]*\.[0-9]*')
host=$(hostname -s)
csv=$dir/pitemp.csv

echo "${host},${now},${temp}" >> $csv
