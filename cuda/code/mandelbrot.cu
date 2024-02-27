#include <iostream>
#include <fstream>
#include <cuComplex.h>
#include <chrono>

// Ranges of the set
#define MIN_X -2
#define MAX_X 1
#define MIN_Y -1
#define MAX_Y 1

// Image ratio
#define RATIO_X (MAX_X - MIN_X)
#define RATIO_Y (MAX_Y - MIN_Y)

// Image size
#define RESOLUTION 1000
#define WIDTH (RATIO_X * RESOLUTION)
#define HEIGHT (RATIO_Y * RESOLUTION)

#define STEP ((double)RATIO_X / WIDTH)

#define DEGREE 2        // Degree of the polynomial
#define ITERATIONS 1000 // Maximum number of iterations

using namespace std;

__global__ void calcMandelbrotPxl(int* image, int dutySize){
    //I should determine which portion of positions I should deal with
    int startPos = threadIdx.x + blockIdx.x * blockDim.x;
    int endPos = startPos + dutySize;
    for (int pos = startPos; pos < endPos; pos++)
    {
        if (pos < WIDTH * HEIGHT){

        image[pos] = 0;

        int row = pos / WIDTH;
        int col = pos % WIDTH;
        cuDoubleComplex c = make_cuDoubleComplex(col * STEP + MIN_X, row * STEP + MIN_Y);

        // z = z^2 + c
        cuDoubleComplex z= make_cuDoubleComplex(0, 0);
        for (int i = 1; i <= ITERATIONS; i++)
        {
            z = cuCadd(cuCmul(z, z), c);

            // If it is convergent
            if (cuCabs(z) >= 2)
            {
                image[pos] = i;
                break;
            }
        }
        }

    }
}

int main(int argc, char **argv)
{
    int N = WIDTH * HEIGHT;
    int *const image = new int[N];
    int *dev_image;

    const auto start = chrono::steady_clock::now();
    dim3 threads(32);
    dim3 blocks ( (N+threads.x-1)/threads.x );
    int dutySize = N/(threads.x * blocks.x);

    cudaMalloc( (void**)&dev_image, N * sizeof(int) );
    cudaMemcpy(dev_image, image, N * sizeof(int), cudaMemcpyHostToDevice);
    calcMandelbrotPxl<<<blocks,threads>>>( dev_image, dutySize );
    // copy the array 'c' back from the GPU to the CPU
    cudaMemcpy(image, dev_image, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();



    const auto end = chrono::steady_clock::now();
    cout << "Time elapsed: "
         << chrono::duration_cast<chrono::seconds>(end - start).count()
         << " seconds." << endl;

    // Write the result to a file
    ofstream matrix_out;

    if (argc < 2)
    {
        cout << "Please specify the output file as a parameter." << endl;
        return -1;
    }

    matrix_out.open(argv[1], ios::trunc);
    if (!matrix_out.is_open())
    {
        cout << "Unable to open file." << endl;
        return -2;
    }

    for (int row = 0; row < HEIGHT; row++)
    {
        for (int col = 0; col < WIDTH; col++)
        {
            matrix_out << image[row * WIDTH + col];

            if (col < WIDTH - 1)
                matrix_out << ',';
        }
        if (row < HEIGHT - 1)
            matrix_out << endl;
    }
    matrix_out.close();

    delete[] image; // It's here for coding style, but useless
    return 0;
}