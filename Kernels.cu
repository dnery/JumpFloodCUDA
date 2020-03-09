// CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Helpers
#include "helper_cuda.h"

// Natives
#include <cstdio>


//Work around false error squiggly lines inside VS; see: https://stackoverflow.com/a/27992604
#ifdef __INTELLISENSE__
#define KERNEL_2ARGS(grid, block)
#define KERNEL_3ARGS(grid, block, sh_mem)
#define KERNEL_4ARGS(grid, block, sh_mem, stream)
#else
#define KERNEL_2ARGS(grid, block) <<< grid, block >>>
#define KERNEL_3ARGS(grid, block, sh_mem) <<< grid, block, sh_mem >>>
#define KERNEL_4ARGS(grid, block, sh_mem, stream) <<< grid, block, sh_mem, stream >>>
#endif

__device__ uchar3 toColor(float2 texCoords, int imageW, int imageH)
{
    if (texCoords.x > 0.f && texCoords.y > 0.f)
    {
        //printf("Tex2D valid lookup: %f %f\n", texCoords.x, texCoords.y);
		return uchar3{
			int(255.99f * texCoords.x / float(imageW)),
			int(255.99f * texCoords.y / float(imageH)),
			int(255.99f *                         .3f)
		};
    }
    return uchar3{ 0,0,0, };
}

__global__ void jumpFloodSync(uchar3* outputCanvas, float2* transientCanvas, int diagramXDim, int diagramYDim, cudaTextureObject_t texture)
{
	// calculate non-normalized texture coordinates
	unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x > diagramXDim || y > diagramYDim) return;

	unsigned outputIdx = y * diagramXDim + x;
	float r, g;
#ifndef __INTELLISENSE__
	r = tex2D<float>(texture, 2 * x, y);
	g = tex2D<float>(texture, 2 * x + 1, y);
#endif

	transientCanvas[outputIdx] = { r, g };
	outputCanvas[outputIdx] = toColor({ r, g }, diagramXDim, diagramYDim);
}

__global__ void jumpFloodKernel(uchar3* outputCanvas, float2* transientCanvas, int diagramXDim, int diagramYDim, int passIndex, cudaTextureObject_t texture)
{
	// calculate non-normalized texture coordinates
	unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned y = blockIdx.y * blockDim.y + threadIdx.y;

	// Ignore out-of-bounds
	if (x > diagramXDim || y > diagramYDim) return;

	unsigned outputIdx = y * diagramXDim + x;

	// JFA step computations
	float maximalDim = fmaxf(diagramXDim, diagramYDim);
	int step = powf(2.f, (log2f(maximalDim) - passIndex - 1));
	for (int gridY = 0; gridY < 3; ++gridY)
	{
		for (int gridX = 0; gridX < 3; ++gridX)
		{
			int xLookup = x - step + gridX * step;
			int yLookup = y - step + gridY * step;

			// Ignore out-of-bounds
			if (xLookup < 0.f || xLookup > diagramXDim || yLookup < 0.f || yLookup > diagramYDim) continue;

			float rTheirs, gTheirs;
#ifndef __INTELLISENSE__
			rTheirs = tex2D<float>(texture, 2 * xLookup, yLookup);
			gTheirs = tex2D<float>(texture, 2 * xLookup + 1, yLookup);
#endif

			// Ignore non-seed pixels
			if (!(rTheirs + gTheirs > 0.f)) continue;

			float rMe, gMe;
#ifndef __INTELLISENSE__
			rMe = tex2D<float>(texture, 2 * x, y);
			gMe = tex2D<float>(texture, 2 * x + 1, y);
#endif

			// No closest seed yet, adopt it
			if (!(rMe + gMe) > 0.f)
			{
				transientCanvas[outputIdx] = { rTheirs, gTheirs };
				outputCanvas[outputIdx] = toColor({ rTheirs, gTheirs }, diagramXDim, diagramYDim);
				continue;
			}

			// Calculate distance only if really needed
			float curDistance = sqrtf((rMe - x) * (rMe - x) + (gMe - y) * (gMe - y));
			float newDistance = sqrtf((rTheirs - x) * (rTheirs - x) + (gTheirs - y) * (gTheirs - y));
			if (newDistance < curDistance)
			{
				transientCanvas[outputIdx] = { rTheirs, gTheirs };
				outputCanvas[outputIdx] = toColor({ rTheirs, gTheirs }, diagramXDim, diagramYDim);
			}
		}
	}
#ifndef __INTELLISENSE__
	__syncthreads();
#endif
}

void jumpFloodWithCuda(unsigned char* hostResultCanvas, float2* hostSourceCanvas, int diagramXDim, int diagramYDim)
{
	// For sanity
	checkCudaErrors(cudaSetDevice(0));

	// Allocate output
    int uintChannels = 3;
	uchar3* deviceOutputCanvas;
	size_t outputCanvasSize = diagramXDim * diagramYDim * uintChannels * sizeof(unsigned char);
	checkCudaErrors(cudaMalloc((void**)&deviceOutputCanvas, outputCanvasSize));

	// Allocate transient space
    int floatChannels = 2;
	float2* deviceTransientCanvas;
	size_t transientCanvasSize = diagramXDim * diagramYDim * floatChannels * sizeof(float);
	cudaMalloc((void**)&deviceTransientCanvas, transientCanvasSize);


	// Allocate image space; see: https://stackoverflow.com/a/16217548
	cudaArray* deviceInputCanvas;
	cudaChannelFormatDesc channelDescription = cudaCreateChannelDesc<float>();
	checkCudaErrors(cudaMallocArray(&deviceInputCanvas, &channelDescription, diagramXDim * floatChannels, diagramYDim));
    checkCudaErrors(cudaMemcpy2DToArray(
        deviceInputCanvas,                                              // Dest data cudaArray
        0,
        0,
        hostSourceCanvas,                                               // Source data pointer
        diagramXDim * floatChannels * sizeof(float),                    // Pitch/alignment for this allocated memory
        diagramXDim * floatChannels * sizeof(float),                    // Copy span width (bytes)
        diagramYDim,                                                    // Copy span height (elements)
        cudaMemcpyHostToDevice
    ));

	// Texture resource
	cudaResourceDesc textureResource;
	memset(&textureResource, 0, sizeof(cudaResourceDesc));
	textureResource.resType         = cudaResourceTypeArray;
	textureResource.res.array.array = deviceInputCanvas;

	// Texture description (actually important); see: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-object-api
	cudaTextureDesc textureDescription;
	memset(&textureDescription, 0, sizeof(cudaTextureDesc));
	textureDescription.addressMode[0]   = cudaAddressModeClamp;         // Clamp over-index lookups across X (default)
	textureDescription.addressMode[1]   = cudaAddressModeClamp;         // Clamp over-index lookups across Y (default)
    textureDescription.filterMode       = cudaFilterModePoint;          // Do not interpolate, pick the actual closest value during lookup
	textureDescription.readMode         = cudaReadModeElementType;      // Do not convert the returned lookup data
	textureDescription.normalizedCoords = 0;                            // Use tex coords, not coords between [0,1)

	// Texture object (allocated and bound during runtime, as opposed to the texture reference)
	cudaTextureObject_t texture = 0;
	checkCudaErrors(cudaCreateTextureObject(&texture, &textureResource, &textureDescription, NULL));

	// Define kernel dimensions & sync
	dim3 bDim(8, 8, 1);
	dim3 gDim(diagramXDim / bDim.x, diagramYDim / bDim.y, 1);
	jumpFloodSync KERNEL_2ARGS(gDim, bDim) (deviceOutputCanvas, deviceTransientCanvas, diagramXDim, diagramYDim, texture);
	checkCudaErrors(cudaDeviceSynchronize());

	float maximalDim = fmaxf(diagramXDim, diagramYDim);
	// We perform log2(N) steps, N being the maximal dims of our image
	for (int passIndex = 0; passIndex <= log2f(maximalDim); ++passIndex)
	{
		// Launch main computations kernel
		jumpFloodKernel KERNEL_2ARGS(gDim, bDim)(deviceOutputCanvas, deviceTransientCanvas, diagramXDim, diagramYDim, passIndex, texture);
		checkCudaErrors(cudaDeviceSynchronize());

		// Update the texture
		checkCudaErrors(cudaMemcpy2DToArray(
			deviceInputCanvas,                                              // Dest data cudaArray
			0,
			0,
			deviceTransientCanvas,                                          // Source data pointer
			diagramXDim * floatChannels * sizeof(float),                    // Pitch/alignment for this allocated memory
			diagramXDim * floatChannels * sizeof(float),                    // Copy span width (bytes)
			diagramYDim,                                                    // Copy span height (elements)
			cudaMemcpyDeviceToDevice
		));
	}

	// Sanity checks
	getLastCudaError("Kernel launch failed!");

	// Copy back
    checkCudaErrors(cudaMemcpy2D(
        hostResultCanvas,                                               // Dest data pointer
        diagramXDim * uintChannels * sizeof(unsigned char),             // Dest mem alignment
        deviceOutputCanvas,                                             // Source data pointer
        diagramXDim * uintChannels * sizeof(unsigned char),             // Source mem alignment
        diagramXDim * uintChannels * sizeof(unsigned char),             // Copy span width (bytes)
        diagramYDim,                                                    // Copy span height (elements)
        cudaMemcpyDeviceToHost
    ));

	// Cleanup
	checkCudaErrors(cudaFree(deviceOutputCanvas));
	checkCudaErrors(cudaFree(deviceTransientCanvas));
	checkCudaErrors(cudaFreeArray(deviceInputCanvas));
	checkCudaErrors(cudaDestroyTextureObject(texture));
	checkCudaErrors(cudaDeviceReset());
}
