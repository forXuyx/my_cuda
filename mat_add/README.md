# Matrix index calculation
## 2D * 2D (nx * ny)
```c
ix = threadIdx.x + blockDim.x * blockIdx.x;
iy = threadIdx.y + blockDim.y * blockIdx.y;
idx = iy * nx + ix;
```