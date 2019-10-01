#Dependencies
* OpenCL 1.2 (Tested on MacOS)

# To Run
    clang++ -std=c++0x -framework OpenCL main.cpp -o main.out

# References
* https://rageandqq.github.io/blog/2018/03/09/opencl-mac-cpp.html
* https://www.eriksmistad.no/using-the-cpp-bindings-for-opencl/
* http://simpleopencl.blogspot.com/2013/06/tutorial-simple-start-with-opencl-and-c.html
* https://github.com/Dakkers/OpenCL-examples/

* https://cnugteren.github.io/tutorial/pages/page1.html [single-precision matrix-multiplication (SGEMM)]

# Codes

### 2-large-vector-addition
* This example compares the timings of adding vectors on the CPU versus adding vectors on the GPU, the latter of which has different implementations.
* The code runs the following implementations of adding large vectors (131072 elements; 8 * 32 * 512). The vectors are added together 10000 times.

- CPU
- GPU, where 1024 threads are spawned and each thread thus gets 128 elements to calculate; there are two implementations of this:
  - (Version 1) each thread gets 128 sequential elements (thread 0 gets 0-127, 1 gets 128-255, ...)
  - (Version 2) each thread gets 128 elements, but coalescing happens (thread 0 gets 0,128,256..., thread 1 gets 1,129,257...)