Repository for standalone ModelJoin operator described in paper (LINK). ModelJoin allows to perform ML model inference in a column store based on a relational model representation.

# Build instructions

Set `USECUDA` to `ON` or `OFF` in CMakeLists.txt and set `CMAKE_CUDA_ARCHITECTURES` according to your GPU if needed.

```
BUILDMODE=<DEBUG/RELEASE> . build.sh
```

Run Tests:
```
cd debug/release
make test
```
