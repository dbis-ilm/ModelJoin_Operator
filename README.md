Repository for standalone ModelJoin operator described in paper (LINK). ModelJoin allows to perform ML model inference in a column store based on a relational model representation.

# Dependencies
- [Intel MKL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html#gs.gof3fc)
- (optional) [CUDA](https://developer.nvidia.com/cuda-downloads)
- (optional) [cuBLAS](https://docs.nvidia.com/cuda/cublas/index.html)

# Build instructions

Set `USECUDA` to `ON` or `OFF` in CMakeLists.txt and set `CMAKE_CUDA_ARCHITECTURES` according to your GPU if needed. Activating CUDA requires CUDA and cuBLAS to be installed.

```
BUILDMODE=<DEBUG/RELEASE> . build.sh
```

Run Tests:
```
cd debug/release
make test
```
