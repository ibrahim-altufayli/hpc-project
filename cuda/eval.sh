#!/bin/bash

echo "CUDA code evaluation using different number of threads per blocks"

echo "Delete all build files"

rm ./builds/exc*


echo "Delete all previously produced images"

ls ./imgs | grep -P 'img_(?!ref).+' | xargs -d"\n" -I% rm ./imgs/%

echo "COMPILE TIME"

echo "Compile with o2"
nvc++ -O2 code/mandelbrot.cu -o builds/exc.o

for i in 1000 2000 3000 4000 5000
do
  for j in 256 128 64 32 24 16 8 4 2 1
  do
	echo "Executing the following iterations /$i/ with the following threads /$j/ per block "
	./builds/exc.o 1000 $i $j
	echo "-------------------------------"
  done  
done
