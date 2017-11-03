#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <string.h>
using namespace std;

#define GAUSS_WIDTH 5
#define SOBEL_WIDTH 3

typedef struct images {
	char *pType;
	int width;
	int height;
	int maxValColor;
	unsigned char *data;
} image;

/**
Reads the input file formatted as pnm. The actual implementation
supports only P5 type pnm images (grayscale).
*/
void readInput(char *fileName, image &img) {

	FILE *inputImageFile;

	inputImageFile = fopen(fileName, "rb");

	img.pType = new char[3];
	fgets(img.pType, sizeof(img.pType), inputImageFile);

	char c = getc(inputImageFile);
	while (c == '#') {
		while (getc(inputImageFile) != '\n');
		c = getc(inputImageFile);
	}

	ungetc(c, inputImageFile);

	fscanf(inputImageFile, "%d%d", &img.width, &img.height);

	fscanf(inputImageFile, "%d", &img.maxValColor);

	while (fgetc(inputImageFile) != '\n');

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

	outputImageFile = fopen(fileName, "wb");

	fprintf(outputImageFile, "%s\n", img.pType);

	fprintf(outputImageFile, "%d %d\n", img.width, img.height);

	fprintf(outputImageFile, "%d\n", img.maxValColor);

	fwrite(img.data, sizeof(unsigned char), img.height * img.width, outputImageFile);

	fclose(outputImageFile);
}

/**
Copies generic data from the input image to output image
*/
void copyPropertiesToImage(image i, image &o) {

	o.pType = new char[3];
	strcpy(o.pType, "P5");
	o.width = i.width;
	o.height = i.height;
	o.maxValColor = i.maxValColor;
}

__global__ void applyGaussianFilter(unsigned char *input, unsigned char *output, float *kernel, int iHeight, int iWidth, int kWidth) {

	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	double sum = 0.0;

	int halvedKW = kWidth / 2;

	for (int i = -halvedKW; i <= halvedKW; i++) {
 		for (int j = -halvedKW; j <= halvedKW; j++) {
			if ((x + j) < iWidth && (x + j) >= 0 && (y + i) < iHeight && (y + i) >= 0) {
				int kPosX = (j + halvedKW);
				int kPosY = (i + halvedKW);
				sum = sum + (float)(input[(y + i) * iWidth + (x + j)]) * kernel[kPosY * kWidth + kPosX];
			}
		}
	}

	if (sum > 255.0)
		sum = 255.0;

	output[y * iWidth + x] = (unsigned char)sum;
}

__global__ void applySobelFilter(unsigned char *in, unsigned char *intensity, float *direction, int ih, int iw) {

	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	int gx, gy;

	if (x > 0 && x + 1 < iw && y > 0 && y + 1 < ih) {
		gx =
			1 * in[(y - 1) * iw + (x - 1)] + (-1) * in[(y - 1) * iw + (x + 1)] +
			2 * in[y * iw + (x - 1)]	   + (-2) * in[y * iw + (x + 1)] +
			1 * in[(y + 1) * iw + (x - 1)] + (-1) * in[(y + 1) * iw + (x + 1)];

		gy =
			   1 * in[(y - 1) * iw + (x - 1)] +    2 * in[(y - 1) * iw + x] +    1 * in[(y - 1) * iw + (x + 1)] +
			(-1) * in[(y + 1) * iw + (x - 1)] + (-2) * in[(y + 1) * iw + x] + (-1) * in[(y + 1) * iw + (x + 1)];

		intensity[y * iw + x] = (unsigned char)sqrt((float)(gx) * (float)(gx) + (float)(gy) * (float)(gy));
		direction[y * iw + x] = atan2((float)gy, (float)gx);
	}
}

int main(int argc, char *argv[]) {

	cout << argv[1] << endl;

	image input, output;

	readInput(argv[1], input);

	float gaussianKernel[] = {
		1.0 / 273.0,  4.0 / 273.0,  7.0 / 273.0,  4.0 / 273.0, 1.0 / 273.0,
		4.0 / 273.0, 16.0 / 273.0, 26.0 / 273.0, 16.0 / 273.0, 4.0 / 273.0,
		7.0 / 273.0, 26.0 / 273.0, 41.0 / 273.0, 26.0 / 273.0, 7.0 / 273.0,
		4.0 / 273.0, 16.0 / 273.0, 26.0 / 273.0, 16.0 / 273.0, 4.0 / 273.0,
		1.0 / 273.0,  4.0 / 273.0,  7.0 / 273.0,  4.0 / 273.0, 1.0 / 273.0
	};

	unsigned char *d_gaussInput, *d_gaussOutput, *d_sobelOutput;
	float *d_gaussKernel, *d_gradDirections;
	int imgRes = input.height * input.width;

	dim3 blocks(input.width / 16, input.height / 16);
	dim3 threads(16, 16);

	cudaMalloc(&d_gaussInput, imgRes);
	cudaMalloc(&d_gaussOutput, imgRes);
	cudaMalloc(&d_gaussKernel, GAUSS_WIDTH * GAUSS_WIDTH * sizeof(float));
	cudaMemcpy(d_gaussInput, input.data, imgRes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_gaussKernel, gaussianKernel, GAUSS_WIDTH * GAUSS_WIDTH * sizeof(float), cudaMemcpyHostToDevice);

	applyGaussianFilter <<< blocks, threads >>> (d_gaussInput, d_gaussOutput, d_gaussKernel, input.height, input.width, GAUSS_WIDTH);

	cudaMalloc(&d_sobelOutput, imgRes);
	cudaMalloc(&d_gradDirections, imgRes * sizeof(float));
	cudaMemcpy(d_sobelOutput, d_gaussOutput, imgRes, cudaMemcpyDeviceToDevice);
	
	applySobelFilter <<< blocks, threads >>> (d_gaussOutput, d_sobelOutput, d_gradDirections, input.height, input.width);

	copyPropertiesToImage(input, output);
	output.data = (unsigned char*)malloc(output.height * output.width);
	cudaMemcpy(output.data, d_sobelOutput, imgRes, cudaMemcpyDeviceToHost);

	cudaFree(d_gaussKernel);
	cudaFree(d_gaussInput);
	cudaFree(d_gaussOutput);
	cudaFree(d_sobelOutput);
	cudaFree(d_gradDirections);

	writeData(argv[2], output);

	system("pause");

	return 0;
}