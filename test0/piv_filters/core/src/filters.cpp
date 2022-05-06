#include <cmath>
#include <vector>
#include <iterator>
#include <functional>
#include <numeric>
#include <cstdint>

#include "lib/kernels.h"
#include "lib/utils.h"

void parallel_bulk( // for future parallel
   std::function<void(std::size_t)>& lambda, 
   std::size_t img_rows,
   std::size_t kernel_size,
   std::size_t thread_count = 4
){
   /* Perform bulk processing due to costs of creating/maintaining queues */
   // get chunk size and starting row
   std::size_t chunk_size = img_rows/thread_count, row = kernel_size / 2;
   // allocate vector of shuck sizes
   std::vector<size_t> chunk_sizes( thread_count, chunk_size );
   // fix rounding errors to remove undefined behavior
   chunk_sizes.back() = (img_rows - 2*(kernel_size / 2) - (thread_count-1)*chunk_size);
      
   for ( const auto& chunk_size_ : chunk_sizes )
   {
      auto processor = [row, chunk_size_, &lambda] ()
      {
         for ( std::size_t j=row; j<row + chunk_size_; ++j )
            lambda(j);
      };
      processor(); // would be multi-threaded later...
      row += chunk_size_;
   }
}

void intensity_cap_filter(
   float* input,
   int N_M,
   float std_mult = 2.f
){
   float upper_limit{};
   
   // calculate mean and std
   auto mean_std{ buffer_mean_std(input, N_M) };
      
   // calculate cap
   upper_limit = mean_std[0] + std_mult * mean_std[1];
   
   // perform intensity capping
   buffer_clip(input, 0.f, upper_limit, N_M);
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
   
   // setup lambda function for column processing
   auto process_row = [
      output, 
      input,
      &kernel,
      &img_cols, 
      &step, 
      &kernel_size
   ]( int _row ) mutable 
   {
      for (int col{kernel_size / 2}; col < (img_cols - kernel_size / 2); ++col)        
         output[step * _row + col] = kernels::apply_conv_kernel(
            input, 
            kernel,
            _row, col, step, 
            kernel_size
         );         
   };
   
   // process rows in parallel
   parallel_bulk(
      process_row, 
      img_rows,
      kernel_size,
      4 // temp
  );
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
   
   // setup lambda function for column processing
   auto process_row = [
      output, 
      input,
      &kernel,
      &img_cols, 
      &step, 
      &kernel_size
   ]( int _row ) mutable
   {
      for (int col{kernel_size / 2}; col < (img_cols - kernel_size / 2); ++col)        
         output[step * _row + col] = input[step * _row + col] - kernels::apply_conv_kernel(
            input, 
            kernel,
            _row, col, step, 
            kernel_size
         );         
   };
   
   // process rows in parallel
   parallel_bulk(
      process_row, 
      img_rows,
      kernel_size,
      4 // temp
  );
}
   // clip pixel values less than zero if necessary
   if (clip_at_zero) 
      buffer_clip(output, 0.f, 1.f, img_rows * img_cols);
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
   buffer_p_norm(output, img_rows * img_cols);
     
   // clip pixel values less than zero if necessary
   if (clip_at_zero) 
   {
      buffer_clip(output, 0.f, 1.f, img_rows * img_cols);
   }     
}