#!/usr/bin/env bash

python3.5 --version

while IFS=";" read -r test_name model bash_size epoch lr activation layer nb_filtre
do
   echo "<----------------------------------------------------------------------------->\n"
   echo "<-------------------------------- $test_name --------------------------------->\n"
   echo "<----------------------------------------------------------------------------->\n"
   nvidia-smi
   python3.5 Template_Final.py $test_name $model $bash_size $epoch $lr $activation $layer $nb_filtre
   sleep 60
   echo "<----------------------------------------------------------------------------->"
   echo "<------------------------------------ FIN ------------------------------------>"
   echo "<----------------------------------------------------------------------------->"
done < ../HyperParam/file_test.csv
