#!/bin/bash

python3 build_csv.py

for model in {'DPN92','VGG','S_DLA'}
do
	python3 main_sata.py -m $model
	echo 3 > /proc/sys/vm/drop_caches
	python3 main_nvme.py -m $model
	echo 3 > /proc/sys/vm/drop_caches
	python3 main_nvm.py -m $model
	echo 3 > /proc/sys/vm/drop_caches
done
