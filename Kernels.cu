// CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Helpers
#include "helper_cuda.h"


// Work around false error squiggly lines inside VS; see: https://stackoverflow.com/a/27992604
#ifdef __INTELLISENSE__
#define KERNEL_2ARGS(grid, block)
#define KERNEL_3ARGS(grid, block, sh_mem)
#define KERNEL_4ARGS(grid, block, sh_mem, stream)
#else
#define KERNEL_2ARGS(grid, block) <<< grid, block >>>
#define KERNEL_3ARGS(grid, block, sh_mem) <<< grid, block, sh_mem >>>
#define KERNEL_4ARGS(grid, block, sh_mem, stream) <<< grid, block, sh_mem, stream >>>
#endif


/**
 * Get an arbitrary 3D color vec from a 2D numeric input.
 */
__device__ uchar3 toColor(float2 texCoords, int imageW, int imageH)
{
	return uchar3{
		int(255.99f * texCoords.x / float(imageW)),
		int(255.99f * texCoords.y / float(imageH)),
		int(255.99f * .3f)
	};
}


__global__ void jumpFloodKernel(uchar3* pixelCanvas, float2* numericCanvas, int diagramXDim, int diagramYDim)
{
	// calculate non-normalized texture coordinates
	unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned y = blockIdx.y * blockDim.y + threadIdx.y;

	// Ignore out-of-bounds index
	if (x > diagramXDim || y > diagramYDim) return;

	float maximalDim = fmaxf(diagramXDim, diagramYDim);

	// JFA pass(es) loop
	for (int passIndex = 0; passIndex < log2f(maximalDim); ++passIndex)
	{
		float step = powf(2.f, (log2f(maximalDim) - passIndex - 1.f));

		// At first, the best candidate is ourselves
		unsigned selfIdx = y * diagramXDim + x;
		float2 closestCandidate = numericCanvas[selfIdx];
		float closestDistance = float(INT_MAX);

		// JFA pass computations
		for (int gridY = 0; gridY < 3; ++gridY)
		{
			for (int gridX = 0; gridX < 3; ++gridX)
			{
				float xLookup = x - step + gridX * step;
				float yLookup = y - step + gridY * step;

				// Ignore out-of-bounds
				if (xLookup < 1e-6f || xLookup > diagramXDim || yLookup < 1e-6f || yLookup > diagramYDim) continue;

				int lookupIdx = yLookup * diagramXDim + xLookup;
				float2 otherCandidate = numericCanvas[lookupIdx];

				if (otherCandidate.x + otherCandidate.y > 1e-6f)
				{
					float otherDistance = sqrtf(
						(otherCandidate.x - x) * (otherCandidate.x - x)
						 + (otherCandidate.y - y) * (otherCandidate.y - y)
					);
					if (otherDistance < closestDistance)
					{
						closestCandidate = otherCandidate;
						closestDistance = otherDistance;
					}
				}

				// Abandoned idea using texture objects...
				#if 0
				#ifndef __INTELLISENSE__
								rTheirs = tex2D<float>(texture, 2 * xLookup, yLookup);
								gTheirs = tex2D<float>(texture, 2 * xLookup + 1, yLookup);
				#endif
				#endif
				#if 0
				#ifndef __INTELLISENSE__
								rMe = tex2D<float>(texture, 2 * x, y);
								gMe = tex2D<float>(texture, 2 * x + 1, y);
				#endif
				#endif
			}
		}

		pixelCanvas[selfIdx] = toColor(closestCandidate, diagramXDim, diagramYDim);
		numericCanvas[selfIdx] = closestCandidate;

#ifndef __INTELLISENSE__
		__syncthreads();
#endif
	}
}

void jumpFloodWithCuda(unsigned char* hostPixelCanvas, float2* hostNumericCanvas, int diagramXDim, int diagramYDim)
{
	// For sanity
	checkCudaErrors(cudaSetDevice(0));

	int pixelChannels = 3;
	int numericChannels = 2;

	// Allocate device numeric canvas
	float2* deviceNumericCanvas;
	size_t numericCanvasSize = diagramXDim * diagramYDim * numericChannels * sizeof(float);
	checkCudaErrors(cudaMalloc((void**)&deviceNumericCanvas, numericCanvasSize));
	checkCudaErrors(cudaMemset(deviceNumericCanvas, 0, numericCanvasSize));


	// Allocate device pixel canvas
	uchar3* devicePixelCanvas;
	size_t pixelCanvasSize = diagramXDim * diagramYDim * pixelChannels * sizeof(unsigned char);
	checkCudaErrors(cudaMalloc((void**)&devicePixelCanvas, pixelCanvasSize));
	checkCudaErrors(cudaMemset(devicePixelCanvas, 0, pixelCanvasSize));

#if 0
	// Allocate image space; see: https://stackoverflow.com/a/16217548
	cudaArray* deviceInputCanvas;
	cudaChannelFormatDesc channelDescription = cudaCreateChannelDesc<float>();
	checkCudaErrors(cudaMallocArray(&deviceInputCanvas, &channelDescription, diagramXDim * numericChannels, diagramYDim));
    checkCudaErrors(cudaMemcpy2DToArray(
        deviceInputCanvas,                                             // Dest data cudaArray
        0,
        0,
        hostNumericCanvas,                                             // Source data pointer
        diagramXDim * numericChannels * sizeof(float),                 // Pitch/alignment for this allocated memory
        diagramXDim * numericChannels * sizeof(float),                 // Copy span width (bytes)
        diagramYDim,                                                   // Copy span height (elements)
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
#endif

	// Copy into
	checkCudaErrors(cudaMemcpy2D(
		deviceNumericCanvas,
		diagramXDim * numericChannels * sizeof(float),
		hostNumericCanvas,
		diagramXDim * numericChannels * sizeof(float),
		diagramXDim * numericChannels * sizeof(float),
		diagramYDim,
		cudaMemcpyHostToDevice
	));


	// Define dimensions & launch kernel
	dim3 bDim(32, 32, 1);
	dim3 gDim(diagramXDim / bDim.x, diagramYDim / bDim.y, 1);
	jumpFloodKernel KERNEL_2ARGS(gDim, bDim)(devicePixelCanvas, deviceNumericCanvas, diagramXDim, diagramYDim);

	// Sanity checks
	checkCudaErrors(cudaDeviceSynchronize());
	getLastCudaError("Kernel launch failed!");

	// Copy back
	checkCudaErrors(cudaMemcpy2D(
		hostPixelCanvas, // Dest data pointer
		diagramXDim * pixelChannels * sizeof(unsigned char), // Dest mem alignment
		devicePixelCanvas, // Source data pointer
		diagramXDim * pixelChannels * sizeof(unsigned char), // Source mem alignment
		diagramXDim * pixelChannels * sizeof(unsigned char), // Copy span width (bytes)
		diagramYDim, // Copy span height (elements)
		cudaMemcpyDeviceToHost
	));

	// Cleanup
	checkCudaErrors(cudaFree(devicePixelCanvas));
	checkCudaErrors(cudaFree(deviceNumericCanvas));
	//checkCudaErrors(cudaFreeArray(deviceInputCanvas));
	//checkCudaErrors(cudaDestroyTextureObject(texture));
	checkCudaErrors(cudaDeviceReset());
}
