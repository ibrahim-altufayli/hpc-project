#include <iostream>
#include <fstream>
#include <cuComplex.h>
#include <chrono>
#include <string.h>
#include <cmath>

// Ranges of the set
#define MIN_X -2
#define MAX_X 1
#define MIN_Y -1
#define MAX_Y 1



#define DEGREE 2        // Degree of the polynomial

using namespace std;

__global__ void calcMandelbrotPxl(int* image, int width, int height, int step, int min_x, int min_y, int iterations, int dutySize){
    //I should determine which portion of positions I should deal with
    int startPos = threadIdx.x + blockIdx.x * blockDim.x;
    int endPos = startPos + dutySize;
    for (int pos = startPos; pos < endPos; pos++)
    {

        if (pos < width * height){

        image[pos] = pos;

        /*int row = pos / width;
        int col = pos % width;
        cuDoubleComplex c = make_cuDoubleComplex(col * step + min_x, row * step + min_y);

        // z = z^2 + c
        cuDoubleComplex z= make_cuDoubleComplex(0, 0);
        for (int i = 1; i <= iterations; i++)
        {
            z = cuCadd(cuCmul(z, z), c);

            // If it is convergent
            if (cuCreal(z)*cuCreal(z) + cuCimag(z)*cuCimag(z) >= 4)
            {
                image[pos] = i;
                break;
            }
        }*/
        }

    }
}

double calc_rmse(int * imageGen, string refImagePath){
    ifstream file(refImagePath.c_str()); 
    int row = 0;
    int width = 0;
    string line;
    string delimiter = ",";
    int sumOfDiff = 0;
    while (std::getline(file, line))
    {
        size_t delimiterPos = 0;
        int pxlCol = 0;
        while ((delimiterPos = line.find(delimiter)) != string::npos) {
            string pxlValueStr = line.substr(0, delimiterPos);
            char* p;
            int pxlValue = strtol(pxlValueStr.c_str(), &p, 10);
            line.erase(0, delimiterPos + delimiter.length());
            int pxl1DPos = row * width + pxlCol;
            sumOfDiff += pow(pxlValue - imageGen[pxl1DPos], 2);
            pxlCol +=1;
        }
        //handel last pixel
        string pxlValueStr = line;
        char* p;
        int pxlValue = strtol(pxlValueStr.c_str(), &p, 10);
        int pxl1DPos = row * width + pxlCol;
        sumOfDiff += pow(pxlValue - imageGen[pxl1DPos], 2);
        pxlCol +=1;
        if(width == 0){
            width = pxlCol;
        }
        row += 1;
    }
    int height = row;
    file.close();
    cout<<"width: "<<width<<" height: "<<height<<endl;
    return sqrt(sumOfDiff/(height * width));
}

int main(int argc, char **argv)
{
    int ITERATIONS = 1000, RESOLUTION = 1000;
    // Image ratio
    int RATIO_X = MAX_X - MIN_X;
    int RATIO_Y = MAX_Y - MIN_Y;


    if(argc > 1){
        char *p;
        RESOLUTION = strtol(argv[1], &p, 10);
        if (*p != '\0'){
            cout << "Please use only integer values for RESOLUTION" << endl;
            return -1;
        }
        if(argc > 2){
            p = NULL;
            ITERATIONS = strtol(argv[2], &p, 10);
            if(*p != '\0'){
                cout << "Please use only integer values for ITERATIONS" << endl;
                return -1;
            }

        }
    }
    cout<< "Image Resolution: "<< RESOLUTION<<endl;
    cout<< "#Iterations: "<< ITERATIONS <<endl;
    

    int WIDTH = RATIO_X * RESOLUTION;
    int HEIGHT = RATIO_Y * RESOLUTION;

    double STEP = ((double)RATIO_X / WIDTH);

    
    int N = WIDTH * HEIGHT;
    int * image = (int*)malloc( N * sizeof(int) );
    int *dev_image;

    const auto start = chrono::steady_clock::now();
    dim3 threads(32);
    dim3 blocks ( (N+threads.x-1)/threads.x );
    int dutySize = N/(threads.x * blocks.x);

    cudaMalloc( (void**)&dev_image, N * sizeof(int) );
    cudaMemcpy(dev_image, image, N * sizeof(int), cudaMemcpyHostToDevice);

    calcMandelbrotPxl<<<blocks,threads>>>( dev_image, WIDTH, HEIGHT, STEP, MIN_X, MIN_Y, ITERATIONS, dutySize);
    

    cudaDeviceSynchronize();
    cudaMemcpy(image, dev_image, N * sizeof(int), cudaMemcpyDeviceToHost);




      const auto end = chrono::steady_clock::now();
    cout << "Time elapsed: "
         << chrono::duration_cast<chrono::milliseconds>(end - start).count()
         << " milliseconds." << endl;

    
    
        ofstream results_out;
        results_out.open("builds/time_results.txt", ios::app);
        if(!results_out.is_open()){
            results_out.open("builds/time_results.txt", ios::trunc);
        }

	       double rmse = calc_rmse(image, "imgs/img_ref_" + to_string(RESOLUTION));
		  cout<<"RMSE: "<<rmse<<endl;

        results_out<<chrono::duration_cast<chrono::milliseconds>(end - start).count()
         << ","<< RESOLUTION<<","<<ITERATIONS<< ','<<rmse<<endl;

        results_out.close();
    

    // Write the result to a file
    ofstream matrix_out;

    
    string imgFilePath = "imgs/img_";
    imgFilePath.append(to_string(RESOLUTION)+"_"+to_string(ITERATIONS));

    cout<<imgFilePath<<endl;
    matrix_out.open(imgFilePath.c_str(), ios::trunc);
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

