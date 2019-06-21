#!/bin/bash
a=''
folders="$(ls)"
for folder in $folders
do
	for folderb in $folders
	do
		echo -e "$folder -> $folderb,"$(diff -U 0 $folder $folderb | grep -v ^@ | wc -l) >> difference.csv
	done
done
