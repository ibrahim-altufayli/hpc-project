#!/bin/bash

echo "OPENMP code evaluation using different optimization flags"

echo "Delete all build files"

rm ./builds/exc*


echo "Delete all previously produced images"

ls ./imgs | grep -P 'img_(?!ref).+' | xargs -d"\n" -I% rm ./imgs/%

echo "COMPILE TIME"

echo "Compile with o2 and static scheduling"
icc -std=c++11 -O2 -qopenmp -xSSE2  ./code/mandelbrot_static.cpp -o ./builds/exc_static.o

echo "Compile with o2 and dynamic scheduling"
icc -std=c++11 -O2 -qopenmp -xSSE2  ./code/mandelbrot_dynamic.cpp -o ./builds/exc_dynamic.o

echo "Compile with o2 and guided scheduling"
icc -std=c++11 -O3 -qopenmp -xSSE2  ./code/mandelbrot_guided.cpp -o ./builds/exc_guided.o


for i in 1000 2000 3000
do
  for j in 512 256 128 64 32 16 8 4 2 1
  do
	echo "Executing the following resolution /$i/ with the following threads /$j/ - static scheduling"
	./builds/exc_static.o $i 1000 $j
	echo "-------------------------------"

	echo "Executing the following resolution /$i/ with the following threads /$j/ - dynamic scheduling"
	./builds/exc_dynamic.o $i 1000 $j
	echo "-------------------------------"

	echo "Executing the following resolution /$i/ with the following threads /$j/ - guided scheduling"
	./builds/exc_guided.o $i 1000 $j
	echo "-------------------------------"


  done  
done
