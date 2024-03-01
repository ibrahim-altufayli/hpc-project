#!/bin/bash

echo "CUDA code evaluation using different number of threads per blocks"

echo "Delete all build files"

rm ./builds/exc*



echo "COMPILE TIME"

echo "Compile with o2"
nvc++ -O2 code/heat.cu -o builds/exc.o

for i in 1000 5000 10000 30000
do
  for j in 256 128 64 32 24 16 8 4 2 1
  do
	echo "Executing the following DIM /$i * $i/ with the following threads /$j/ per block "
	./builds/exc.o $i $i $j
	echo "-------------------------------"
  done  
done
