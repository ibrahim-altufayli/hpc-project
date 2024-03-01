#!/bin/bash

echo "Sequential code evaluation using different optimization flags"

echo "Delete all build files"

rm ./builds/exc*


echo "Delete all previously produced images"
ls ./imgs | grep -P 'img_(?!ref).+' | xargs -d"\n" -I% rm ./imgs/%


echo "COMPILE TIME"

echo "Compile with o2"
icc -std=c++11 ./code/mandelbrot.cpp -diag-disable=10441 -O2 -o ./builds/exc_o2



echo "RUN EXECUTABLES"

for i in 1000 2000 3000 4000 5000
do
  for j in 1000 2000 3000 4000 5000
  do
	echo "Executing the following resolution /$i/ with the following iterations /$j/"
	./builds/exc_o2 $i $j
	echo "-------------------------------"
  done  
done

