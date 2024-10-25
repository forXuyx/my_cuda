#include <iostream>

#include <stdio.h>

__global__ void hello_gpu() {
    printf("hello gpu\n");
}

int main() {
    hello_gpu<<<2, 2>>>();
    cudaDeviceSynchronize();
    return 0;
}