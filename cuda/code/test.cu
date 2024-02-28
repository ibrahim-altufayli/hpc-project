#include <stdio.h>
#include <sys/time.h>
#include <iostream>
#define N (32 * 1024)
using namespace std;

__global__ void add( int *a, int *b, int *c ) {
int tid = blockIdx.x*blockDim.x + threadIdx.x;
if (tid < N) c[tid] = a[tid] + b[tid];
}


int main( void ) {
int *a, *b, *c, *dev_a, *dev_b, *dev_c;
dim3 threads(32);
dim3 blocks ( (N+threads.x-1)/threads.x );

a = (int*)malloc( N * sizeof(int) ); 
b = (int*)malloc( N * sizeof(int) );
c = (int*)malloc( N * sizeof(int) );//the same for b and c
cudaMalloc( (void**)&dev_a, N * sizeof(int)  );
cudaMalloc( (void**)&dev_b, N * sizeof(int)  );
cudaMalloc( (void**)&dev_c, N * sizeof(int)  );

//the same for dev_b and dev_c
for (int i=0; i<N; i++) { a[i] = i; b[i] = 2 * i; }

// copy the arrays 'a' and 'b' to the GPU
cudaMemcpy( dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice
);
cudaMemcpy( dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice );

add<<<blocks,threads>>>( dev_a, dev_b, dev_c );
// copy the array 'c' back from the GPU to the CPU
cudaMemcpy( c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);
for (int i=0; i<N; i++) { cout<<c[i]<<endl;}
cudaDeviceSynchronize();


// free the memory
cudaFree( dev_a );
free( a ); //the same for b and c
return 0;
}