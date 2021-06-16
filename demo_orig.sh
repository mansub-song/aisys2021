#!/bin/bash

python3 build_csv.py

for model in {'DPN92','VGG','S_DLA'}
do
	python3 main_sata.py -r -m $model
	rm /mnt/pmem0/pmem.dat
	echo 3 > /proc/sys/vm/drop_caches
	python3 main.py -r -m $model
	rm /mnt/pmem0/pmem.dat
	echo 3 > /proc/sys/vm/drop_caches
	python3 main.py -r -m $model
	rm /mnt/pmem0/pmem.dat
	echo 3 > /proc/sys/vm/drop_caches
done
sleep 5s
python3 result.py
