#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
using namespace std;

#define GAUSS_WIDTH 5
#define SOBEL_WIDTH 3

typedef struct images {
	char pType[3];
	int width;
	int height;
	int maxValColor;
	unsigned char *data;
} image;

/**
	Reads the input file formatted as pnm. The actual implementation
	supports only P5 type pnm images (grayscale).
*/
void readInput(const char *fileName, image &img) {

	FILE *inputImageFile;

	inputImageFile = fopen(fileName, "rb");

	fgets(img.pType, sizeof(img.pType), inputImageFile);

	char c = getc(inputImageFile);
	while(c == '#') {
		while(getc(inputImageFile) != '\n');
		c = getc(inputImageFile);
	}

	ungetc(c, inputImageFile);

	fscanf(inputImageFile, "%d%d", &img.width, &img.height);

	fscanf(inputImageFile, "%d", &img.maxValColor);

	while (fgetc(inputImageFile) != '\n') ;

	if (img.pType[1] == '5') {
		img.data = (unsigned char*)malloc(img.height * img.width);
		fread(img.data, sizeof(unsigned char), img.height * img.width, inputImageFile);
	}

	fclose(inputImageFile);
}

/**
	Writes an image to the output file.
*/
void writeData(const char *fileName, image img) {

	FILE *outputImageFile;

	outputImageFile = fopen (fileName, "wb");

	fprintf(outputImageFile, "%s\n", img.pType);

	fprintf(outputImageFile, "%d %d\n", img.width, img.height);

	fprintf(outputImageFile, "%d\n", img.maxValColor);

	fwrite(img.data, sizeof(unsigned char), img.height * img.width, outputImageFile);

	fclose(outputImageFile);
}

void copyPropertiesToImage(image i, image &o) {

	o.pType = i.pType;
	o.width = i.width;
	o.height = i.height;
	o.maxValColor = i.maxValColor;
}

__global__ void applyGaussianFilter(unsigned char *input, unsigned char *output, float *kernel, int iHeight, int iWidth, int kWidth) {

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	float sum = 0.0;

	int halvedKW = kWidth / 2;

	for (int i = -halvedKW; i <= halvedKW; i++) {
		for (int j = -halvedKW; j <= halvedKW; j++) {
			if ((x + j) < iWidth && (x + j) >= 0 && (y + i) < iHeight && (y + i) >= 0) {
				int kPosX = (j + halvedKW);
				int kPosY = (i + halvedKW);
				sum += (float) input[(y + i) * iWidth + (x + j)] * kernel[kPosY * kWidth + kPosX];
			}
		}
	}

	sum = min(sum, 255);

	output[y * iWidth + x] = (unsigned char) sum;
}

// __global__ void applySobelFilter(unsigned char *input, unsigned char *output, float *sobelX, float *sobelY, int iHeight, int iWidth, int kWidth) {

// 	int x = blockIdx.x * blockDim.x + threadIdx.x;
// 	int y = blockIdx.y * blockDim.y + threadIdx.y;

// 	int halvedKW = kWidth / 2;
// 	float gx = 0.0, gy = 0.0;

// 	if (x > 0 && (x + 1) < iWidth && y > 0 && (y + 1) < iHeight) {
// 		for (int i = -halvedKW; i <= halvedKW; i++) {
// 			for (int j = -halvedKW; j <= halvedKW; j++) {
// 				int kPosX = (j + halvedKW);
// 				int kPosY = (i + halvedKW);
// 				gx += (float) input[(y + i) * iWidth + (x + j)] * sobelX[kPosY * kWidth + kPosX];
// 				gx += (float) input[(y + i) * iWidth + (x + j)] * sobelY[kPosY * kWidth + kPosX];
// 			}
// 		}
// 		output[y * iWidth + x] = (unsigned char) sqrtf(gx * gx + gy * gy);
// 	}
// }

__global__ void applySobelFilter(unsigned char in, unsigned char out, int ih, int iw) {

	int x = blockIdx.x + blockDim.x + threadIdx.x;
	int y = blockIdx.y + blockDim.y + threadIdx.y;

	float gx, gy;

	if (x > 0 && x + 1 < iw && y > 0 && y + 1 < ih) {
		gx = 
			1.0 * in[(y - 1) * iw + (x - 1)] + (-1.0) * in[(y - 1) * iw + (x + 1)] +
			2.0 * in[y * iw + (x - 1)]			 + (-2.0) * in[y * iw + (x + 1)] 			 +
			1.0 * in[(y + 1) * iw + (x - 1)] + (-1.0) * in[(y + 1) * iw + (x + 1)] ;

		gy =
			1.0    * in[(y - 1) * iw + (x - 1)] +    2.0 * in[(y - 1) * iw + x] +    1.0 * in[(y - 1) * iw + (x + 1)] +
			(-1.0) * in[(y + 1) * iw + (x - 1)] + (-2.0) * in[(y + 1) * iw + x] + (-1.0) * in[(y + 1) * iw + (x + 1)];
		
		output[y * iw + x] = (unsigned char) sqrtf(gx * gx + gy * gy);
	}
}

int main(int argc, char *argv[]) {
	
	image input;

	readInput(argv[1], input);

	float gaussianKernel[25] = {
		1.f/273.f,  4.f/273.f,  7.f/273.f,  4.f/273.f, 1.f/273.f, 
		4.f/273.f, 16.f/273.f, 26.f/273.f, 16.f/273.f, 4.f/273.f, 
		7.f/273.f, 26.f/273.f, 41.f/273.f, 26.f/273.f, 7.f/273.f, 
		4.f/273.f, 16.f/273.f, 26.f/273.f, 16.f/273.f, 4.f/273.f, 
		1.f/273.f,  4.f/273.f,  7.f/273.f,  4.f/273.f, 1.f/273.f,
	};

	// float sobelKernelX[9] = {
	// 	1.f, 0.f, -1.f,
	// 	2.f, 0.f, -2.f,
	// 	1.f, 0.f, -1.f,
	// };

	// float sobelKernelY[9] = {
	// 	1.f, 2.f, 1.f,
	// 	0.f, 0.f, 0.f,
	// 	-1.f, -2.f, -1.f,
	// };

	int *d_gaussInput, *d_gaussOutput, *d_gaussKernel, *d_sobelKernelX, *d_sobelKernelY, *d_sobelOutput;
	int imgRes = input.height * img.height;

	dim3 blocks (input.width / 16, input.height / 16);
	dim3 threads(16, 16);

	cudaMalloc(&d_gaussInput, imgRes);
	cudaMalloc(&d_gaussOutput, imgRes);
	cudaMalloc(&d_gaussKernel, GAUSS_WIDTH * GAUSS_WIDTH);
	cudaMemcpy(d_gaussInput, input, imgRes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_gaussKernel, gaussianKernel, GAUSS_WIDTH * GAUSS_WIDTH * sizeof(float), cudaMemcpyHostToDevice);

	applyGaussianFilter <<< blocks, threads >>> (d_gaussInput, d_gaussOutput, d_gaussKernel, input.height, input.width, GAUSS_WIDTH);

	cudaThreadSynchronize();

	cudaFree(d_gaussKernel);
	cudaFree(d_gaussInput);

	// cudaMalloc(&d_sobelKernelX, SOBEL_WIDTH * SOBEL_WIDTH);
	// cudaMalloc(&d_sobelKernelY, SOBEL_WIDTH * SOBEL_WIDTH);
	// cudaMemcpy(d_sobelKernelX, sobelX, SOBEL_WIDTH * SOBEL_WIDTH, sizeof(float), cudaMemcpyHostToDevice);
	// cudaMemcpy(d_sobelKernelY, sobelY, SOBEL_WIDTH * SOBEL_WIDTH, sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc(&d_sobelOutput, imgRes);

	// applySobelFilter <<< blocks, threads >>> (d_gaussOutput, d_sobelOutput, d_sobelKernelX, d_sobelKernelY, input.height, input.width, SOBEL_WIDTH);

	applySobelFilter <<< blocks, threads >>> (d_gaussOutput, d_sobelOutput, input.height, input.width);

	cudaThreadSynchronize();

	cudaMemcpy(output.data, d_sobelOutput, imgRes, cudaMemcpyDeviceToHost);

	cudaFree(d_gaussOutput);
	cudaFree(d_sobelKernelX);
	cudaFree(d_sobelKernelY);
	cudaFree(d_sobelOutput);

	copyPropertiesToImage(input, output);

	writeData(argv[2], output);

	return 0;
}