#ifndef CONST_H
#define CONST_H
#define debug 1
#define FEATURE_NUM 162500
#include <cuda_runtime.h>
typedef struct feature
{
    char feature_type;
    char i;
    char j;
    char w;
    char h;

} feature;
extern feature *features_info;
__device__ feature *d_features_info; // Device pointer

typedef struct window
{
    int x;
    int y;
    int w;
    int h;
} window;
#endif