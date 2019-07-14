#!/bin/bash
a=''
i=0
folders="$(ls)"
for folder in $folders
do
    ((i+=1))
	a+=$(echo "$folder=$folder,")
done
echo -e "Show logs of $i models"
tensorboard --logdir $a
