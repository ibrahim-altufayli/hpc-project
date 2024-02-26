#!/bin/bash

echo "Sequential code evaluation using different optimization flags"

echo "Delete all build files"

rm ./builds/exc*


echo "Delete all previously produced images"
rm ./imgs/img*


echo "COMPILE TIME"

echo "Compile with o0"
icc -std=c++11 ./code/mandelbrot.cpp -O0 -xSSE2 -o ./builds/exc_o0

echo "Compile with o1"
icc -std=c++11 ./code/mandelbrot.cpp -O1 -xSSE2 -o ./builds/exc_o1

echo "Compile with o2"
icc -std=c++11 ./code/mandelbrot.cpp -O2 -xSSE2 -o ./builds/exc_o2

echo "Compile with o3"
icc -std=c++11 ./code/mandelbrot.cpp -O3 -xSSE2 -o ./builds/exc_o3


echo "RUN EXECUTABLES"
for i in 0 1 2 3
do 
echo "Running executable with optimization level $i"
./builds/exc_o$i 
done 
