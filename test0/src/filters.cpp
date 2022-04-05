#include <cmath>
#include <iostream>
#include <vector>
#include <iomanip>
#include <iterator>
#include <functional>

#include "lib/kernels.h"

void apply_filter_lowpass(
   float* output,
   float* input,
   std::vector<float>& kernel,
   int img_rows, int img_cols, 
   int kernel_size
){
   int step{ img_cols };
   for (int row{kernel_size / 2}; row < (img_rows - kernel_size / 2); ++row)
   {
      for (int col{kernel_size / 2}; col < (img_cols - kernel_size / 2); ++col)
      {
         output[step * row + col] = kernels::apply_kernel(input, kernel, row, col, step, kernel_size);
//         std::cout << row << ' ' << col << ' ' << output[img_rows * row + col] << '\n';
      }
   }
}

void local_variance_norm(
   float* output,
   float* input,
   float* buffer,
   int img_rows, int img_cols, 
   int kernel_size,
   float sigma1,
   float sigma2
){
   /*
   Calculate variance and mean via two gaussian filters and normalize the array.
   */
   auto GKernel1 = kernels::gaussian(kernel_size, sigma1);
   auto GKernel2 = kernels::gaussian(kernel_size, sigma2);
   int k_ind{0}, step{ img_cols };
   float sum{0.f}, den{0.f}, invalid_denom{0.f};
   
   // highpass
   for (int row{kernel_size / 2}; row < (img_rows - kernel_size / 2); ++row)
   {
      for (int col{kernel_size / 2}; col < (img_cols - kernel_size / 2); ++col)
      {
         buffer[step * row + col] = input[step * row + col] - kernels::apply_kernel(input, GKernel1, row, col, step, kernel_size);
      }
   }
   // variance, mean, and normalize
   for (int row{kernel_size / 2}; row < (img_rows - kernel_size / 2); ++row)
   {
      for (int col{kernel_size / 2}; col < (img_cols - kernel_size / 2); ++col)
      {
         sum = 0.f;
         k_ind = 0;
         for (int i{-kernel_size / 2}; i <= (kernel_size / 2); ++i)
         {
            for (int j{-kernel_size / 2}; j <= (kernel_size / 2); ++j)
            {
               // The operation should be done on images with range [0,1]
               sum += GKernel2[k_ind] * pow(buffer[step * (row + i) + (col + j)], 2);
               ++k_ind;
            }
         }
         den = pow(sum, 0.5f);
         output[step * row + col] = (den != invalid_denom) ? (buffer[step * row + col] / den) : 0;
      }
   }  
   // normalize
   float max_val{ 0.f };
   for (int i{ 0 }; i < (img_rows * img_cols); ++i)
      max_val = (output[i] > max_val) ? output[i] : max_val;
   
   for (int i{ 0 }; i < (img_rows * img_cols); ++i)
      output[i] /= max_val;
}