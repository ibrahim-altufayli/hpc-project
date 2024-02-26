#!/bin/bash

echo "OPENMP code evaluation using different optimization flags"

echo "Delete all build files"

rm ./builds/exc*


echo "Delete all previously produced images"

ls ./imgs | grep -P 'img_(?!ref).+' | xargs -d"\n" -I% rm ./imgs/%

echo "COMPILE TIME"

echo "Compile with o3"
mpiicc -std=c++11 -O3 -qopenmp -xSSE2 code/mandelbrot.cpp -o builds/exc.o

for i in 1000 2000 3000
do
  for j in 512 64 32 16 8 4 2 1
  do
	echo "Executing the following resolution /$i/ with the following processes /$j/"
	mpiexec -hostfile list -nolocal -perhost 1 -np $j ./builds/exc.o $i 1000
  done  
done
