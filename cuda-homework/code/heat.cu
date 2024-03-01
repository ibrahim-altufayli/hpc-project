#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <chrono>

// Simple define to index into a 1D array from 2D space
#define I2D(num, c, r) ((r)*(num)+(c))

/*
 * `step_kernel_mod` is currently a direct copy of the CPU reference solution
 * `step_kernel_ref` below. Accelerate it to run as a CUDA kernel.
 */
using namespace std;

__global__ void step_kernel_mod(int ni, int nj, float fact, float* temp_in, float* temp_out)
{
  int i00, im10, ip10, i0m1, i0p1;
  float d2tdx2, d2tdy2;

  //evaluate i and j 
  // loop over all points in domain (except boundary)
  int i = blockIdx.x * blockDim.x + threadIdx.x; //column 
  int j = blockIdx.y * blockDim.y + threadIdx.y; //row
  if (i < ni-1 && i > 0 && j > 0 && j< nj-1){
      // find indices into linear memory
      // for central point and neighbours
      //this way of finding the exact locations is not perfict since wee need to calculate the derivatives 
      /*i00 = j * gridDim.x * blockDim.x + i;
      im10 = j * gridDim.x * blockDim.x + (i-1);
      ip10 = j * gridDim.x * blockDim.x + (i+1);
      i0m1 = (j-1) * gridDim.x * blockDim.x + i;
      i0p1 = (j+1) * gridDim.x * blockDim.x + i;*/
      i00 = I2D(ni, i, j);
      im10 = I2D(ni, i-1, j);
      ip10 = I2D(ni, i+1, j);
      i0m1 = I2D(ni, i, j-1);
      i0p1 = I2D(ni, i, j+1);

      // evaluate derivatives
      d2tdx2 = temp_in[im10]-2*temp_in[i00]+temp_in[ip10];
      d2tdy2 = temp_in[i0m1]-2*temp_in[i00]+temp_in[i0p1];

      // update temperatures
      temp_out[i00] = temp_in[i00]+fact*(d2tdx2 + d2tdy2);
    }
}

void step_kernel_ref(int ni, int nj, float fact, float* temp_in, float* temp_out)
{
  int i00, im10, ip10, i0m1, i0p1;
  float d2tdx2, d2tdy2;


  // loop over all points in domain (except boundary)
  for ( int j=1; j < nj-1; j++ ) {
    for ( int i=1; i < ni-1; i++ ) {
      // find indices into linear memory
      // for central point and neighbours
      i00 = I2D(ni, i, j);
      im10 = I2D(ni, i-1, j);
      ip10 = I2D(ni, i+1, j);
      i0m1 = I2D(ni, i, j-1);
      i0p1 = I2D(ni, i, j+1);

      // evaluate derivatives
      d2tdx2 = temp_in[im10]-2*temp_in[i00]+temp_in[ip10];
      d2tdy2 = temp_in[i0m1]-2*temp_in[i00]+temp_in[i0p1];

      // update temperatures
      temp_out[i00] = temp_in[i00]+fact*(d2tdx2 + d2tdy2);
    }
  }
}

int main(int argc, char **argv)
{
  // Specify our 2D dimensions
  int ni = 1000;
  int nj = 1000;

  if(argc > 1){
        char *p;
        nj = strtol(argv[1], &p, 10);
        if (*p != '\0'){
            cout << "Please use only integer values for RESOLUTION" << endl;
            return -1;
        }
        if(argc > 2){
            p = NULL;
            ni = strtol(argv[2], &p, 10);
            if(*p != '\0'){
                cout << "Please use only integer values for ITERATIONS" << endl;
                return -1;
            }

        }
    }
    cout<< "PROBLEM DIM 1:"<< nj<<endl;
    cout<< "PROBLEM DIM 2:"<< ni <<endl;
  int istep;
  int nstep = 200; // number of time steps

  
  float tfac = 8.418e-5; // thermal diffusivity of silver

  float *temp1_ref, *temp2_ref, *temp1, *temp2, *temp_tmp, *temp1_dev, *temp2_dev;

  const int size = ni * nj * sizeof(float);

  temp1_ref = (float*)malloc(size);
  temp2_ref = (float*)malloc(size);
  temp1 = (float*)malloc(size);
  temp2 = (float*)malloc(size);
  cudaMalloc((void**)&temp1_dev, size);
  cudaMalloc((void**)&temp2_dev, size);

  // Initialize with random data
  for( int i = 0; i < ni*nj; ++i) {
    temp1_ref[i] = temp2_ref[i] = temp1[i] = temp2[i] = (float)rand()/(float)(RAND_MAX/100.0f);
  }

  // Execute the CPU-only reference version
  for (istep=0; istep < nstep; istep++) {
    step_kernel_ref(ni, nj, tfac, temp1_ref, temp2_ref);

    // swap the temperature pointers
    temp_tmp = temp1_ref;
    temp1_ref = temp2_ref;
    temp2_ref= temp_tmp;
  }

  int nthreads = 16;
    if(argc > 3 && strlen(argv[3]) > 0){
        char* p;
        nthreads = strtol(argv[3], &p, 10);
        if (*p != '\0') {
            return 1; // In main(), returning non-zero means failure
        }
    }

  const auto start = chrono::steady_clock::now();
  // Execute the modified version using same data
  //design the grid of blocks of threads
  dim3 blocksPerGrid((ni + nthreads -1)/nthreads,(nj + nthreads -1)/nthreads);
  dim3 threadsPerBlock(nthreads, nthreads);
  cout<<"GRID DIM_1: "<<blocksPerGrid.x<<" GRID DIM_2: "<<blocksPerGrid.y<<endl;
  cout<<"BLOCK DIM 1: "<<threadsPerBlock.x<<" BLOCK DIM_2: "<<threadsPerBlock.y<<endl;

  //copy the intialization of tempreture values to device
    cudaMemcpy(temp1_dev, temp1, size, cudaMemcpyHostToDevice);
    cudaMemcpy(temp2_dev, temp2, size, cudaMemcpyHostToDevice);

  for (istep=0; istep < nstep; istep++) {
    
    step_kernel_mod<<<blocksPerGrid, threadsPerBlock>>>(ni, nj, tfac, temp1_dev, temp2_dev);
    cudaDeviceSynchronize();
    // swap the temperature pointers
    temp_tmp = temp1_dev;
    temp1_dev = temp2_dev;
    temp2_dev= temp_tmp;
  }
  cudaMemcpy(temp1, temp1_dev, size, cudaMemcpyDeviceToHost);
  cudaMemcpy(temp2, temp2_dev, size, cudaMemcpyDeviceToHost);

  const auto end = chrono::steady_clock::now();
    cout << "Time elapsed: "
         << chrono::duration_cast<chrono::milliseconds>(end - start).count()
         << " milliseconds." << endl;

ofstream results_out;
        results_out.open("builds/time_results.txt", ios::app);
        if(!results_out.is_open()){
            results_out.open("builds/time_results.txt", ios::trunc);
        }

  float maxError = 0;
  // Output should always be stored in the temp1 and temp1_ref at this point
  for( int i = 0; i < ni*nj; ++i ) {
    if (abs(temp1[i]-temp1_ref[i]) > maxError) { maxError = abs(temp1[i]-temp1_ref[i]); }
  }

  results_out<<chrono::duration_cast<chrono::milliseconds>(end - start).count()
         << ","<< nj<<","<<ni<<','<<blocksPerGrid.x<<','<<blocksPerGrid.y<<','<<threadsPerBlock.x<<','<<threadsPerBlock.y<<','<<maxError<<endl;

      results_out.close();
  // Check and see if our maxError is greater than an error bound
  if (maxError > 0.0005f)
    printf("Problem! The Max Error of %.5f is NOT within acceptable bounds.\n", maxError);
  else
    printf("The Max Error of %.5f is within acceptable bounds.\n", maxError);

  free( temp1_ref );
  free( temp2_ref );
  free( temp1 );
  free( temp2 );
  cudaFree(temp1_dev); 
  cudaFree(temp2_dev); 

  return 0;
}
