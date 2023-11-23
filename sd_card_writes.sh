#!/bin/sh

cat /proc/diskstats | grep mmcblk0 | head -n1 | awk '{print $8}'
