#ifndef FILTER_H
#define FILTER_H

#include <vector>

void intensity_cap_filter(
   float*,
   int,
   float
);

void binarize_filter(
   float*,
   float*,
   int,
   float
);

void apply_kernel_lowpass(
   float*,
   float*,
   std::vector<float>&,
   int, int, 
   int
);

void apply_kernel_highpass(
   float*,
   float*,
   std::vector<float>&,
   int, int, 
   int,
   bool
);

void local_variance_norm(
   float*,
   float*,
   float*,
   int, int, 
   int,
   float,
   float,
   bool
);

#endif