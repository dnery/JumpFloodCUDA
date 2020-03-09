
#include "cuda_runtime.h"
#include "helper_string.h"
#include "helper_image.h"

#include "ft2build.h"
#include FT_FREETYPE_H

#include <cstdio>
#include <cmath>


/**
 * Get rid of false error squiggly lines inside VS.
 * See: https://stackoverflow.com/a/27992604
 */
#ifdef __INTELLISENSE__
#define KERNEL_2ARGS(gridSize, blockSize)
#define KERNEL_3ARGS(gridSize, blockSize, sh_mem)
#define KERNEL_4ARGS(gridSize, blockSize, sh_mem, stream)
#else
#define KERNEL_2ARGS(gridSize, blockSize) <<< gridSize, blockSize >>>
#define KERNEL_3ARGS(gridSize, blockSize, sh_mem) <<< gridSize, blockSize, sh_mem >>>
#define KERNEL_4ARGS(gridSize, blockSize, sh_mem, stream) <<< gridSize, blockSize, sh_mem, stream >>>
#endif


void jumpFloodWithCuda(unsigned char* hostResultCanvas, float2* hostSourceCanvas, int diagramXDim, int diagramYDim);

int main()
{

#if 0
	/**
	 * Idea 1: 
	 *	1. Read and rasterize a truetype font using the freetype lib
	 *  2. Calculate the signed-distance-function (usually stored in the alpha channel for post processing) of a large resolution raster
	 */
	const char* fontFaceFile = "TruenoLt.otf";
	// Scan some default dirs to find the resource (inside ../../data, in this case)
	const char* fontFacePath = sdkFindFilePath(fontFaceFile, 0);
	if (fontFacePath == NULL)
	{
		fprintf(stderr, "Font face not found: %s\n", fontFaceFile);
		return 1;
	}

	FT_Library ftLibraryHandle;
	if (FT_Init_FreeType(&ftLibraryHandle))
	{
		fprintf(stderr, "Error initializing FreeType library handle\n");
		return 1;
	}

	FT_Face fontFaceHandle;
	if (FT_New_Face(ftLibraryHandle, fontFacePath, 0, &fontFaceHandle))
	{
		fprintf(stderr, "Error loading font face: %s\n", fontFacePath);
		return 1;
	}


	FT_Set_Pixel_Sizes(fontFaceHandle, 0, 512); // Set one of the dims as 0 and it will replicate the other
	if (FT_Load_Char(fontFaceHandle, 'g', FT_LOAD_RENDER)) // Load default AND render
	{
		fprintf(stderr, "Error loading character 'g'\n");
		return 1;
	}

	char outputFilename[1024];
	strcpy(outputFilename, fontFacePath);
	strcpy(outputFilename + strlen(fontFacePath) - 4, "_out.pgm");
	sdkSavePGM(
		outputFilename,
		fontFaceHandle->glyph->bitmap.buffer,
		fontFaceHandle->glyph->bitmap.width,
		fontFaceHandle->glyph->bitmap.rows
	);
	printf("Wrote '%s'\n", outputFilename);

    char* terminator = "";
    for (int iRow = 0; iRow < fontFaceHandle->glyph->bitmap.rows; iRow++)
    {
        printf("%s", terminator);
		for (int iCol = 0; iCol < fontFaceHandle->glyph->bitmap.width; iCol++)
		{
            size_t charIndex = iRow * fontFaceHandle->glyph->bitmap.width + iCol;
            printf("%u ", fontFaceHandle->glyph->bitmap.buffer[charIndex]);
		}
        terminator = "\n";
    }
#endif

	/**
	 * Idea 2: generate a Voronoi fill using the Jump-Flood algorithm, computed on the GPU.
	 */
	const int seeds = 3;
	const int diagramXDim = 128;
	const int diagramYDim = 128;
	const int channels = 3;

	/**
	 * 1. Set some hardcoded seed pixel locations.
	 */
	float2 voronoiSeedsUV[seeds] = {	// UV coordinates (between 0 and 1)
		{.2f, .2f},				// p1: upper left
		{.5f, .8f},				// p2: lower middle
		{.8f, .2f},				// p3: upper right
	};
	float2 voronoiSeeds[seeds];			// Texture coordinates (sub-pixel measurement, similar to gl_TexCoord)
	for (int iSeed = 0; iSeed < seeds; ++iSeed)
	{
		voronoiSeeds[iSeed] = float2{
			.49f + floor(voronoiSeedsUV[iSeed].x * float(diagramXDim)),
			.49f + floor(voronoiSeedsUV[iSeed].y * float(diagramYDim))
		};
	}

	/**
	 * 2. Set the R,G values of the seed pixels to their own tex coordinates.
	 */
	float2 voronoiCanvas[diagramXDim * diagramYDim];
	float2* voronoiCanvasFill = voronoiCanvas;
	for (int iRow = 0; iRow < diagramYDim; iRow++)
	{
		for (int iCol = 0; iCol < diagramXDim; iCol++)
		{
			for (int iSeed = 0; iSeed < seeds; ++iSeed)
			{
				if (iCol == int(voronoiSeeds[iSeed].x) && iRow == int(voronoiSeeds[iSeed].y))
				{
					printf("New seed at %f %f\n", voronoiSeeds[iSeed].x, voronoiSeeds[iSeed].y);
					voronoiCanvasFill->x = voronoiSeeds[iSeed].x;
					voronoiCanvasFill->y = voronoiSeeds[iSeed].y;
					goto canvasIterationDone;
				}
			}

			voronoiCanvasFill->x = 0.f;
			voronoiCanvasFill->y = 0.f;

		canvasIterationDone:
			voronoiCanvasFill++;
		}
	}

	// Allocate GPU resources, launch the computation kernel
	unsigned char* voronoiOutputImage = (unsigned char*)malloc(diagramXDim * diagramYDim * channels * sizeof(unsigned char));
	jumpFloodWithCuda(voronoiOutputImage, voronoiCanvas, diagramXDim, diagramYDim);

	char* diagramFileName = "./data/Diagram.ppm";
	__savePPM(diagramFileName, voronoiOutputImage, diagramXDim, diagramYDim, 3);
	printf("Wrote '%s'\n", diagramFileName);

	free(voronoiOutputImage);
	return 0;
}
