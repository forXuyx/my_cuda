# Matrix index calculation
## 2D * 2D (nx * ny)
```c
ix = threadIdx.x + blockDim.x * blockIdx.x;
iy = threadIdx.y + blockDim.y * blockIdx.y;
idx = iy * nx + ix;
```
## 2D * 1D (nx * ny)
```c
ix = threadIdx.x + blockDim.x * blockIdx.x;
iy = blockIdx.y;
idx = iy * nx + ix;
```
## 1D * 1D (nx * ny)
```c
ix = threadIdx.x + blockDim.x * blockIdx.x;
if (ix < nx) {
    for (int iy = 0; iy < ny; iy++) {
        idx = iy * nx + ix;
        // do something by idx
    }
}
```