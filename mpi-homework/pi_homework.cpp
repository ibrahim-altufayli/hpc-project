#include <stdio.h>
#include <iostream>
#include <time.h>
#include <string.h>
#include <stdlib.h>
#include <mpi.h>
#include <fstream>


               
#define PI25DT 3.141592653589793238462643

using namespace std;

int main(int argc, char **argv)
{
    long int intervals = 100000000000;
    if(argc > 1){
        char *p;
        intervals = strtol(argv[1], &p, 10);
        if (*p != '\0'){
            cout << "Please use only integer values for intervals" << endl;
            return -1;
        }
    }

    int myid, numprocs;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

     

    long dutySize =  intervals / numprocs;
    long int startPos = myid * dutySize + 1;
    long int endPos = startPos + dutySize;

    if(myid == 0){
        printf("Number of intervals: %ld\n", intervals);
        std::cout<<"Number of Processes: "<<numprocs<<std::endl;
        std::cout<<"Duty Size for each Process is: "<<dutySize<<std::endl;
    }

    long int i;
    double x, dx, f, sum, globalSum, pi;
    double time2;
    
    time_t time1 = clock();


    sum = 0.0;
    dx = 1.0 / (double) intervals;

    for (i = startPos; i < endPos; i++) {
        x = dx * ((double) (i - 0.5));
        f = 4.0 / (1.0 + x*x);
        sum = sum + f;
    }
    
    MPI_Reduce(&sum, &globalSum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if(myid == 0){
        pi = dx*globalSum;
        time2 = (clock() - time1) / (double) CLOCKS_PER_SEC;
        printf("Computed PI %.24f\n", pi);
        printf("The true PI %.24f\n\n", PI25DT);
        printf("Elapsed time (s) = %.2lf\n", time2);
        ofstream results_out;
        results_out.open("time_results.txt", ios::app);
        
        if(!results_out.is_open()){
            results_out.open("time_results.txt", ios::trunc);

        
        }
          results_out<<time2
         << ","<<intervals<<","<<numprocs<<endl;
    
        results_out.close();
    }

   
    MPI_Finalize();
    return 0;
}