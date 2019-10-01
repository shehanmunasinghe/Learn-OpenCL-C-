// kernel.cl
// Multiply two matrices A * B = C
// Device code.

// OpenCL Kernel
__kernel void
matrixMul(__global float *C,
          __global float *A,
          __global float *B,
          int wA, int wB)
{

   int i = get_global_id(0);
   int k = get_global_id(1);
   // if (i > num_rows_A || k > wB)
   //    return;

   // Sum is on the register(local to each thread)
   float sum = 0;

   // This iterate a lot on the global memory 2*j times
   for (int j = 0; j < wA; j++)
   {
      // A[i][j] == A[i*wA+j]
      // B[j][k] == B[j*wB+k]
      //sum += A[i][j]*B[j][k];
      sum += A[i * wA + j] * B[j * wB + k];
   }

   // And now one more time
   C[i * wB + k] = sum;
}
