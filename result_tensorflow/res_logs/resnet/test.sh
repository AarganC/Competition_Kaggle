#!/bin/bash
a=''
folders="$(ls)"
for folder in $folders
do
	#echo $folder
	a+=$(echo "$folder=$folder,")
	#echo "a = $a"
done
tensorboard --logdir $a
