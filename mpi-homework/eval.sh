#!/bin/bash



echo "Delete all build files"


echo "COMPILE TIME"

echo "Compile with MPI o2"
mpiicc  -O2 -qopenmp pi_homework.cpp -o ./exc.o

for i in 10000000000 100000000000 1000000000000
do
  for j in 1024 512 256 128 64 32 16 8 4 2 1
  do
	echo "Executing the following iterations /$i/ with the following processes /$j/ in all machines"
	mpiexec -hostfile list -nolocal -perhost 1 -np $j ./exc.o $i
  done  
done

