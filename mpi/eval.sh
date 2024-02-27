#!/bin/bash

echo "OPENMP code evaluation using different optimization flags"

echo "Delete all build files"

rm ./builds/exc*


echo "Delete all previously produced images"

ls ./imgs | grep -P 'img_(?!ref).+' | xargs -d"\n" -I% rm ./imgs/%

echo "COMPILE TIME"

echo "Compile with MPI o2"
mpiicc -std=c++11 -O2 -qopenmp code/mandelbrot_mpi.cpp -o builds/exc_mpi.o

echo "Compile with MPI-OPENMP o2"
mpiicc -std=c++11 -O2 -qopenmp code/mandelbrot_mpi_openmp.cpp -o builds/exc_mpi_openmp.o

echo "********MPI on different Machines********" >> ./builds/time_results.txt
for i in 1000 2000 3000
do
  for j in 512 256 128 64 32 16 8 4 2 1
  do
	echo "Executing the following resolution /$i/ with the following processes /$j/ in all machines"
	mpiexec -hostfile list -nolocal -perhost 1 -np $j ./builds/exc_mpi.o $i 1000
  done  
done

echo "********MPI-OpenMP on different Machines********" >> ./builds/time_results.txt

for i in 1000 2000 3000 
do
  for j in 512 256 128 64 32 16 8 4 2 1
  do
	echo "Executing the following resolution /$i/ with the following processes /$j/"
	mpiexec -hostfile list -nolocal -perhost 1 -np $j ./builds/exc_mpi_openmp.o $i 1000
  done  
done

echo "********MPI on one Machine********" >> ./builds/time_results.txt

for i in 1000 2000 3000
do
  for j in 512 256 128 64 32 16 8 4 2 1
  do
	echo "Executing the following resolution /$i/ with the following processes /$j/ in all machines"
	mpiexec -hostfile list-one-machine -nolocal -perhost 1 -np $j ./builds/exc_mpi.o $i 1000
  done  
done

echo "********MPI-OpenMP on one Machine********" >> ./builds/time_results.txt

for i in 1000 2000 3000 
do
  for j in 512 256 128 64 32 16 8 4 2 1
  do
	echo "Executing the following resolution /$i/ with the following processes /$j/"
	mpiexec -hostfile list-one-machine -nolocal -perhost 1 -np $j ./builds/exc_mpi_openmp.o $i 1000
  done  
done

