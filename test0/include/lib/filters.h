#ifndef FILTER_H
#define FILTER_H

#include <vector>

void apply_filter_lowpass(
   float*,
   float*,
   std::vector<float>&,
   int, int, 
   int
);

void apply_filter_highpass(
   float*,
   float*,
   std::vector<float>&,
   int, int, 
   int
);

void local_variance_norm(
   float*,
   float*,
   float*,
   int, int, 
   int,
   float,
   float
);

#endif