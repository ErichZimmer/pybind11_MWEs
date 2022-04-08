#include <cmath>
#include <iostream>
#include <vector>
#include <iomanip>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "lib/kernels.h"
#include "lib/filters.h"

// Interface
namespace py = pybind11;

// wrap C++ function with NumPy array IO
py::array_t<float> low_pass_filter_wrapper(
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
   
   int kernel_type = 0; // gaussian kernel
   auto GKernel = kernels::get_kernel_type(kernel_type)(kernel_size, sigma);

   // call pure C++ function
   apply_kernel_lowpass(
      ptr_out,
      ptr_in,
      GKernel,
      N, M, 
      kernel_size
   );
   
   result.resize({N,M});
   
   return result;
}

py::array_t<float> high_pass_filter_wrapper(
   py::array_t<float> input,
   int kernel_size = 3,
   float sigma = 1,
   py::bool_ clip_at_zero = false
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
   
   int kernel_type = 0; // gaussian kernel
   auto GKernel = kernels::get_kernel_type(kernel_type)(kernel_size, sigma);

   // call pure C++ function
   apply_kernel_highpass(
      ptr_out,
      ptr_in,
      GKernel,
      N, M, 
      kernel_size,
      clip_at_zero
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
      throw std::runtime_error("Input should be 2-D NumPy array");

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

void mult_scal(
   float* output,
   float* input,
   const int constant,
   int N, int M
){
   int step = M;
   for (int i = 0; i < N; ++i)
   {
      for (int j = 0; j < M; ++j)
      {
         output[step * i + j] = input[step * i + j] * constant;
      }
   }
}

py::array_t<float> test_wrapper(
   py::array_t<float> input,
   int testConst
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

   // call pure C++ function
   mult_scal(
      ptr_out,
      ptr_in,
      N, M, 
      testConst
   );
   
   result.resize({N,M});
   
   return result;
}


PYBIND11_MODULE(piv_filters_core,m) {
   m.doc() = "Bindings for convolution filters written in c++.";
   m.def("test_wrapper", &test_wrapper, "Test wrapper by multiplying a scalar to an array.");
   m.def("lowpass_filter", &low_pass_filter_wrapper, "Apply a low pass filter to a 2D array");
   m.def("highpass_filter", &high_pass_filter_wrapper, "Apply a high pass filter to a 2D array");
   m.def("local_variance_normalization", &local_variance_norm_wrapper, "Apply a local variance normalization filter to a 2D array");
}
