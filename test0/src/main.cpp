
#include <cmath>
#include <iostream>
#include <vector>
#include <iomanip>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

//#include "kernels.h"
namespace constants
{
   constexpr float PI = 22/7;
}

namespace kernels
{
   std::vector<float> gaussian(int kernel_size, float sigma)
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
}

float apply_kernel(
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

void apply_filter_lowpass(
   float* output,
   float* input,
   std::vector<float>& kernel,
   int img_rows, int img_cols, 
   int kernel_size
){
   for (int row{kernel_size / 2}; row < (img_rows - kernel_size / 2); ++row)
   {
      for (int col{kernel_size / 2}; col < (img_cols - kernel_size / 2); ++col)
      {
         output[img_cols * row + col] = apply_kernel(input, kernel, row, col, img_cols, kernel_size);
//         std::cout << row << ' ' << col << ' ' << output[img_rows * row + col] << '\n';
      }
   }
}

void apply_filter_highpass(
   float* output,
   float* input,
   std::vector<float>& kernel,
   int img_rows, int img_cols, 
   int kernel_size
){
   for (int row{kernel_size / 2}; row < (img_rows - kernel_size / 2); ++row)
   {
      for (int col{kernel_size / 2}; col < (img_cols - kernel_size / 2); ++col)
      {
         output[img_rows * row + col] = input[img_rows * row + col] - apply_kernel(input, kernel, row, col, img_rows, kernel_size);
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
   auto GKernel1 = kernels::gaussian(kernel_size, sigma1);
   auto GKernel2 = kernels::gaussian(kernel_size, sigma2);
   int k_ind{0}, den{0};
   float sum{0}, invalid_denom{0.f};
   
   apply_filter_highpass(
      buffer,
      input, 
      GKernel1,
      img_cols, img_rows,
      kernel_size
   );
   
   for (int row{kernel_size / 2}; row < (img_rows - kernel_size / 2); ++row)
   {
      for (int col{kernel_size / 2}; col < (img_cols - kernel_size / 2); ++col)
      {
         sum = 0;
         k_ind = 0;
         for (int i{-kernel_size / 2}; i <= (kernel_size / 2); ++i)
         {
            for (int j{-kernel_size / 2}; j <= (kernel_size / 2); ++j)
            {
               // The operation should be done on images with range [0,1]
               sum += GKernel2[k_ind] * pow(buffer[img_rows * (row + i) + (col + j)], 2);
               ++k_ind;
            }
         }
         den = pow(sum, 0.5);
         output[img_rows * row + col] = (den != invalid_denom) ? (buffer[img_rows * row + col] / den) : 0;
      }
   }  
}
std::function<std::vector<float>(int, float)> get_kernel_type(std::string kernel_s)
{
   if (kernel_s == "gaussian")
      return &kernels::gaussian;
   else
      std::runtime_error("Invalid kernel type. Supported kernels: 'gaussian'");
}

// Interface
namespace py = pybind11;

// wrap C++ function with NumPy array IO
py::array_t<float> apply_conv_filter_wrapper(
   py::array_t<float> input,
   int kernel_size = 3,
   float sigma = 1
){
   // check input dimensions
   if ( input.ndim() != 2 )
      throw std::runtime_error("Input should be 2-D NumPy array");

   auto buf1 = input.request();
   
   int N = input.shape()[0], M = input.shape()[1];

   py::array_t<float> result = py::array_t<float>(buf1.size);
   auto buf2 = result.request();
   
   float* ptr_in  = (float*) buf1.ptr;
   float* ptr_out = (float*) buf2.ptr;

   auto GKernel = get_kernel_type("gaussian")(kernel_size, sigma);
//   std::cout << N << ' ' << M << '\n';
   // call pure C++ function
   apply_filter_lowpass(
      ptr_out,
      ptr_in,
      GKernel,
      N, M, 
      kernel_size
   );
   result.resize({N,M});
   return result;
}

py::array_t<float> local_variance_norm_wrapper(
   py::array_t<float> input,
   int kernel_size = 3,
   float sigma1 = 2,
   float sigma2 = 2
){
   // check input dimensions
   if ( input.ndim() != 2 )
      throw std::runtime_error("Ouput should be 2-D NumPy array");

   auto buf1 = input.request();

   int N = input.shape()[0], M = input.shape()[1];
   
   py::array_t<float> result = py::array_t<float>(buf1.size);
   auto buf2 = result.request();
   
   py::array_t<float> temp_buffer = py::array_t<float>(buf1.size);
   auto buf3 = temp_buffer.request();
   
   float* ptr_out = (float*) buf2.ptr;
   float* ptr_in  = (float*) buf1.ptr;
   float* ptr_buf = (float*) buf3.ptr;
   
   // call pure C++ function
   local_variance_norm(
      ptr_out,
      ptr_in,
      ptr_buf,
      N, M, 
      kernel_size,
      sigma1,
      sigma2
   );
   result.resize({N,M});
   return result;
}

PYBIND11_MODULE(example_filters_bindings,m) {
   m.doc() = "Bindings for convolution filters written in c++.";
   m.def("gaussian_filter", &apply_conv_filter_wrapper, "Apply a gaussian filter to a 2D array");
   m.def("local_variance_normalization", &local_variance_norm_wrapper, "Apply a local variance normalization filter to a 2D array");
}
