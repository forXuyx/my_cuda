#include <stdio.h>
#include "../tool/common.cuh"

__global__ void addMat(int *a, int *b, int *c, const int nx, const int ny) {
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = ix + iy * nx;
    if (ix < nx && iy < ny) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {

    // 1.Set GPU
    setGPU();

    // 2.Allocating memory in host
    int nx = 8;
    int ny = 4;
    int nxy = nx * ny;
    size_t byteCount = nxy * sizeof(int);

    int *hostA, *hostB, *hostC;
    hostA = (int *)malloc(byteCount);
    hostB = (int *)malloc(byteCount);
    hostC = (int *)malloc(byteCount);
    if (hostA != nullptr && hostB != nullptr && hostC != nullptr) {
        for (int i = 0; i < nxy; i ++ ) {
            hostA[i] = i;
            hostB[i] = i + 10;
        }
        memset(hostC, 0, byteCount);
    } else {
        printf("Fail to allocating memory in host\n");
        exit(-1);
    }

    // 3.Allocating memory in device
    int *deviceA, *deviceB, *deviceC;
    errorCheck(cudaMalloc(&deviceA, byteCount), __FILE__, __LINE__);
    errorCheck(cudaMalloc(&deviceB, byteCount), __FILE__, __LINE__);
    errorCheck(cudaMalloc(&deviceC, byteCount), __FILE__, __LINE__);

    if (deviceA != nullptr && deviceB != nullptr && deviceC != nullptr) {
        errorCheck(cudaMemcpy(deviceA, hostA, byteCount, cudaMemcpyHostToDevice), __FILE__, __LINE__);
        errorCheck(cudaMemcpy(deviceB, hostB, byteCount, cudaMemcpyHostToDevice), __FILE__, __LINE__);
        errorCheck(cudaMemcpy(deviceC, hostC, byteCount, cudaMemcpyHostToDevice), __FILE__, __LINE__);
    } else {
        printf("Fail to allocating memory in device\n");
        free(hostA);
        free(hostB);
        free(hostC);
        exit(-1);
    }

    // 4.Calculate on GPU
    dim3 block(4, 4);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    addMat<<<grid, block>>>(deviceA, deviceB, deviceC, nx, ny);
    errorCheck(cudaDeviceSynchronize(), __FILE__, __LINE__);

    errorCheck(cudaMemcpy(hostC, deviceC, byteCount, cudaMemcpyDeviceToHost), __FILE__, __LINE__);

    for (int i = 0; i < 10; i ++ ) {
        printf("id=%d, matrix_A=%d, matrix_B=%d, result=%d\n", i+1, hostA[i], hostB[i], hostC[i]);
    }

    free(hostA);
    free(hostB);
    free(hostC);

    errorCheck(cudaFree(deviceA), __FILE__, __LINE__);
    errorCheck(cudaFree(deviceB), __FILE__, __LINE__);
    errorCheck(cudaFree(deviceC), __FILE__, __LINE__);

    errorCheck(cudaDeviceReset(), __FILE__, __LINE__);

    return 0;
}