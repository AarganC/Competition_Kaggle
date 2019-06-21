#!/usr/bin/env bash

python3.5 --version

while IFS=";" read -r test_name model bash_size epoch lr activation layer nb_filtre nb_dropout_flag nb_dropout_value
do
   echo "<----------------------------------------------------------------------------->\n"
   echo "<-------------------------------- $test_name --------------------------------->\n"
   echo "<----------------------------------------------------------------------------->\n"
   nvidia-smi
   python3.5 Template_Final.py $test_name $model $bash_size $epoch $lr $activation $layer $nb_filtre $nb_dropout_flag $nb_dropout_value
   sleep 60
   echo "<----------------------------------------------------------------------------->"
   echo "<------------------------------------ FIN ------------------------------------>"
   echo "<----------------------------------------------------------------------------->"
done < ../HyperParam/file_test2.csv
