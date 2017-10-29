#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
using namespace std;

#define GAUSS_SIZE 25

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

/**
	Returns the greatest divisor of the argument.
	Useful at computing the number of threads per block.
*/
int getGreatestDivisor(int n)
{
	int res = n;

	for (int i = 2; i <= sqrt(n); i++) {
		while (res % i == 0) {
			if (res <= 1024)
				return res;
			res /= i;
		}
		if (res <= 1024)
			return res;
	}
	
	return res;
}

__global__ void applyGaussianFilter(unsigned char *input, unsigned char *output, double *kernel, int imgHeight, int imgWidth) {

	int kernelSize = 5;

	

}

int main(int argc, char *argv[]) {
	
	image input;

	// readInput(argv[1], input);
	// writeData(argv[2], input);

	double gaussianKernel[][5] = {
		{1, 4, 7, 4, 1}, 
		{4, 16, 26, 16, 4}, 
		{7, 26, 41, 26, 7}, 
		{4, 16, 26, 16, 4}, 
		{1, 4, 7, 4, 1}
	};

	for (int i = 0; i < 5; ++i)
		for (int j = 0; j < 5; ++j)
			gaussianKernel[i][j] /= 273.0;

	int *d_gaussInput, *d_gaussOutput, *d_gaussKernel;
	int imgRes = input.height * img.height;

	cudaMalloc(&d_gaussInput, imgRes);
	cudaMalloc(&d_gaussOutput, imgRes);
	cudaMalloc(&d_gaussKernel, GAUSS_SIZE);
	cudaMemcpy(d_gaussInput, input, imgRes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_gaussKernel, gaussianKernel, GAUSS_SIZE, cudaMemcpyHostToDevice);



	return 0;
}