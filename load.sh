#!/bin/bash

python3 build_csv.py


for model in {'DPN92','VGG','S_DLA'}
do
	python3 main.py -r -m $model -p /mnt/sda
	rm /mnt/pmem0/pmem.dat
	echo 3 > /proc/sys/vm/drop_caches
	python3 main.py -r -m $model -p .
	rm /mnt/pmem0/pmem.dat
	echo 3 > /proc/sys/vm/drop_caches
	python3 main.py -r -m $model -p /mnt/pmem0
	rm /mnt/pmem0/pmem.dat
	echo 3 > /proc/sys/vm/drop_caches
done
sleep 5s
python3 result.py
