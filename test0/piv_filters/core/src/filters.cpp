#include <cmath>
#include <iostream>
#include <vector>
#include <iomanip>
#include <iterator>
#include <functional>
#include <numeric>

#include "lib/kernels.h"

void intensity_cap_filter(
   float* output,
   float* input,
   int N_M,
   float std_mult
){
   float sum{}, mean{}, std_{}, upper_limit{};
   
   // calculate mean and std
   for (int i{}; i < N_M; ++i)
   {
      sum += input[i];
      std_ += input[i]*input[i]; // temp
   }
   mean = sum / N_M;
   std_ = sqrt( (std_ / N_M) + (mean*mean) - (2*mean*mean) );
   
   // calculate cap
   upper_limit = mean + std_mult * std_;
   
   // perform intensity capping
   for (int i{}; i < N_M; ++i)
      output[i] = (input[i] < upper_limit) ? input[i] : upper_limit;
}

void binarize_filter(
   float* output,
   float* input,
   int N_M,
   float threshold
){
   
   // perform binarization, assuming pixel intensity range of [0..1]
   for (int i{}; i < N_M; ++i)
      output[i] = (input[i] > threshold) ? 1.f : 0.f;
}

void apply_kernel_lowpass(
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
      }
   }
}

void apply_kernel_highpass(
   float* output,
   float* input,
   std::vector<float>& kernel,
   int img_rows, int img_cols, 
   int kernel_size,
   bool clip_at_zero = false
){   
   int step{ img_cols };
   float den{ 0.f }, invalid_set{ 0.f };
   
   for (int row{kernel_size / 2}; row < (img_rows - kernel_size / 2); ++row)
   {
      for (int col{kernel_size / 2}; col < (img_cols - kernel_size / 2); ++col)
      {         
         den = input[step * row + col] - kernels::apply_kernel(input, kernel, row, col, step, kernel_size);
         
         if(clip_at_zero && (den < invalid_set)) // clip invalid pixels
            den = invalid_set;
         
         output[step * row + col] = den;
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
   float sigma2,
   bool clip_at_zero = false
){
   /*
   Calculate variance and mean via two gaussian filters and normalize the array.
   */
   auto GKernel1 = kernels::gaussian(kernel_size, sigma1);
   auto GKernel2 = kernels::gaussian(kernel_size, sigma2);
   
   int k_ind{0}, step{ img_cols };
   float sum{0.f}, den{0.f}, invalid_set{0.f};
   
   // highpass
   apply_kernel_highpass(buffer, input, GKernel1, img_cols, img_rows, kernel_size, false);
   
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
         output[step * row + col] = (den != invalid_set) ? (buffer[step * row + col] / den) : 0;
      }
   }  
   
   // normalize
   float max_val{ 0.f };
   for (int i{ 0 }; i < (img_rows * img_cols); ++i)
      max_val = (output[i] > max_val) ? output[i] : max_val;
   
   for (int i{ 0 }; i < (img_rows * img_cols); ++i)
      output[i] /= max_val;
      
  // clip pixel values less than zero if necessary
   if (clip_at_zero) 
   {
      for (int i{ 0 }; i < (img_rows * img_cols); ++i)
      {
         if (output[i] < invalid_set)
            output[i] = invalid_set;
      } 
   }    
}