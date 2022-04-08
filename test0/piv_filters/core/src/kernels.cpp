#include <cmath>
#include <iostream>
#include <vector>
#include <iomanip>

#include "lib/constants.h"
#include "lib/kernels.h"

std::vector<float> kernels::gaussian(int kernel_size, float sigma)
{
    std::vector<float> kernel(kernel_size*kernel_size,0);
    
    if (sigma <=0 ){
        sigma = 0.3*((kernel_size-1)*0.5 - 1) + 0.8;
    }
    float s = 2.0 * sigma * sigma;

    // sum is for normalization
    float sum = 0.0;

    // generating nxn kernel
    int i,j;
    float mean = kernel_size/2;
    for (i=0 ; i<kernel_size ; i++) {
        for (j=0 ; j<kernel_size ; j++) {
            kernel[(i*kernel_size)+j] =exp( -0.5 * (pow((i-mean)/sigma, 2.0) + pow((j-mean)/sigma,2.0)) ) / (2 * constants::PI * sigma * sigma);
            sum += kernel[(i*kernel_size)+j];
        }
    }

    // normalising the Kernel
    for (int i = 0; i < kernel.size(); ++i){
        kernel[i] /= sum;
    }

    return kernel;
}

std::vector<float> kernels::box(int kernel_size, float dummy)
{
    std::vector<float> kernel(kernel_size*kernel_size,0);

    // generating nxn kernel
    int i, j, sum = 0;
    for (i=0 ; i<kernel_size ; i++) {
        for (j=0 ; j<kernel_size ; j++) {
            kernel[(i*kernel_size)+j] = 1;
            sum += 1;
        }
    }

    // normalising the Kernel
    for (int i = 0; i < kernel.size(); ++i){
        kernel[i] /= sum;
    }

    return kernel;
}

std::function<std::vector<float>(int, float)> kernels::get_kernel_type(int kernel_type)
{
   switch(kernel_type)
   {
      case 0: return &kernels::gaussian;
      case 1: return &kernels::box;
      default: std::runtime_error("Invalid kernel type. Supported kernels: '0': gaussian, '1': box");
   }
}

float kernels::apply_kernel(
   const float* input,
   const std::vector<float>& kernel,
   int row, int col, int step, 
   int kernel_size
){
   int k_ind{0};
   float sum{0};
   for (int i{-kernel_size / 2}; i <= (kernel_size / 2); ++i)
   {
      for (int j{-kernel_size / 2}; j <= (kernel_size / 2); ++j)
      {
         // The operation should be done on images with range [0,1]
         sum += kernel[k_ind] * (input[step * (row + i) + (col + j)]);
         ++k_ind;
      }
   }
   return sum;
}