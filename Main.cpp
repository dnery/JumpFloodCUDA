#include "cuda_runtime.h"


#include <cstdio>


/**
 * Get rid of false error squiggly lines inside VS.
 * See: https://stackoverflow.com/a/27992604
 */
#ifdef __INTELLISENSE__
#define KERNEL_2ARGS(grid, block)
#define KERNEL_3ARGS(grid, block, sh_mem)
#define KERNEL_4ARGS(grid, block, sh_mem, stream)
#else
#define KERNEL_2ARGS(grid, block) <<< grid, block >>>
#define KERNEL_3ARGS(grid, block, sh_mem) <<< grid, block, sh_mem >>>
#define KERNEL_4ARGS(grid, block, sh_mem, stream) <<< grid, block, sh_mem, stream >>>
#endif


cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

int main()
{
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}
