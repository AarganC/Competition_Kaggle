#!/usr/bin/env bash

while IFS=";" read -r test_name model bash_size epoch lr activation layer filtre
do
   python Template_Final.py $test_name $model $bash_size $epoch $lr $activation $layer $filtre
done < /Users/aargancointepas/Documents/ESGI-4IBD/MachineLearning/CompetitionKaggle/HyperParam/file_test.csv