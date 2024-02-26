#!/bin/bash

echo "Sequential code evaluation using different optimization flags"

echo "Delete all build files"

rm ./builds/exc*


echo "Delete all previously produced images"
ls ./imgs | grep -P 'img_(?!ref).+' | xargs -d"\n" -I% rm ./imgs/%


echo "COMPILE TIME"

echo "Compile with o0"
icc -std=c++11 ./code/mandelbrot-rmse.cpp -diag-disable=10441 -O0 -o ./builds/exc_o0

echo "Compile with o1"
icc -std=c++11 ./code/mandelbrot-rmse.cpp -diag-disable=10441 -O1 -o ./builds/exc_o1

echo "Compile with o2"
icc -std=c++11 ./code/mandelbrot-rmse.cpp -diag-disable=10441 -O2 -o ./builds/exc_o2

echo "Compile with o3"
icc -std=c++11 ./code/mandelbrot-rmse.cpp -diag-disable=10441 -O3 -o ./builds/exc_o3

echo "Compile with -xHost"
icc -std=c++11 ./code/mandelbrot-rmse.cpp -diag-disable=10441 -xHost -o ./builds/exc_xhost

echo "Compile with -fast"
icc -std=c++11 ./code/mandelbrot-rmse.cpp -diag-disable=10441 -fast -o ./builds/exc_fast

echo "Compile with -xSSE3"
icc -std=c++11 ./code/mandelbrot-rmse.cpp -diag-disable=10441 -xSSE3 -o ./builds/exc_xsse3

echo "Compile with -fast -xSSE3"
icc -std=c++11 ./code/mandelbrot-rmse.cpp -diag-disable=10441 -fast -xSSE3 -o ./builds/exc_fast_xsse3


echo "RUN EXECUTABLES"

echo "Running executable with optimization level 0"
./builds/exc_o0

echo "Running executable with optimization level 1"
./builds/exc_o1


echo "Running executable with optimization level 2"
./builds/exc_o2


echo "Running executable with optimization level 3"
./builds/exc_o3


echo "Running executable with optimization -xHost"
./builds/exc_xhost


echo "Running executable with optimization -fast"
./builds/exc_fast

echo "Running executable with optimization -xSEE3"
./builds/exc_xsse3

echo "Running executable with optimization -fast -xSEE3"
./builds/exc_fast_xsse3



