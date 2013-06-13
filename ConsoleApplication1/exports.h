#ifndef EXPORTS_H
#define EXPORTS_H

#ifdef COMPILE_NEURAL_NETWORKS_LIB
  #define EXPORT_ __declspec(dllexport)
#else
  #define EXPORT_ __declspec(dllimport)
#endif

#endif