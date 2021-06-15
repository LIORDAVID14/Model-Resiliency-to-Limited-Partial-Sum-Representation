#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>
#include <vector>

#define DIVCEIL(a,b) (((a)%(b)!=0)?(a/b+1):(a/b))
//psum size
#define blocksize 32

template <typename scalar_t>
__global__ void gemm_kernel(
  const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> A,
  const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> B,
  torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> C,
  int bits_allowed,
  int bits_to_all) {
  __shared__ scalar_t As[blocksize][blocksize];
  __shared__ scalar_t Bs[blocksize][blocksize];

  int bx = blockIdx.x;
  int by = blockIdx.y;

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int a_begin_y = blocksize * by;
  int b_begin_x = blocksize * bx;

  //compute into C
  int sum = 0;
  int A_width = A.size(1)/blocksize;

  //bits_to_all - max psum bits
  //bits_allowed - number of high digits to save

  int number_all = pow(2,bits_to_all-1);
  int max_all = number_all - 1;
  int min_all = -number_all;
  int number_not_allowed = pow(2,bits_to_all-1-(bits_allowed-1));
  int max_not_allowed = number_not_allowed;
  int min_not_allowed = -number_not_allowed;
  int dynamic_factor = pow(2,bits_allowed-1);
  int dynamic_to_decrease = 1; //1 FOR NOT REDUCE - FIRST TIME


  for (int i=0; i<A_width; i++){
    As[ty][tx] = A[a_begin_y + ty][i*blocksize + tx];
    Bs[ty][tx] = B[b_begin_x + tx][i*blocksize + ty];
    __syncthreads();


    //Original way to compute the matrix
/*
    for (int t=0; t<blocksize; t++){
      sum += As[ty][t] * Bs[t][tx];
    }
*/

    //FIRST METHOD:
    //limit total bits
/*
    int number_allowed = pow(2,bits_allowed-1);
    int max_num = number_allowed - 1;
    int min_num = -number_allowed;
    for (int t=0; t<blocksize; t++){
      sum += As[ty][t] * Bs[t][tx];
      if (sum >= max_num){
        sum = max_num;
      }
      else if (sum <= min_num){
        sum = min_num;
      }
    }
*/

    //SECOND METHOD:
    //Save only MSB
/*
    int mask = -pow(2,(bits_to_all-bits_allowed));
    for (int t=0; t<blocksize; t++){
      int compute = As[ty][t] * Bs[t][tx];
      if (number_not_allowed != 1){
        if (compute >= 0){
            compute = compute & mask;
        }
        else{
            compute = (-compute) & mask;
            compute = (compute ^ -1) + 1;
        }
      }

      sum += compute;
      if (sum >= max_all){
        sum = max_all;
      }
      else if (sum <= min_all){
        sum = min_all;
      }
    }
*/

    //THIRD METHOD:
    //Dynamic LSB Reduce

    for (int t=0; t<blocksize; t++){
          int compute = As[ty][t] * Bs[t][tx];

          if (number_not_allowed != 1){
                if (dynamic_to_decrease != 1){
                    if (compute >= 0){
                        compute = compute & (-dynamic_to_decrease);
                    }
                    else{
                        compute = (-compute) & (-dynamic_to_decrease);
                        compute = (compute ^ -1) + 1;
                    }
                }
                sum += compute;

                while (sum > dynamic_factor - 1 || sum < -dynamic_factor) {
                    if (dynamic_factor < number_all) {         //check for not passing max psum with the dynamic
                        dynamic_to_decrease = dynamic_to_decrease * 2;
                        dynamic_factor = dynamic_factor * 2;
                    }
                    else {
                        break;
                    }
                }
                if (sum >= 0){
                    sum = sum & (-dynamic_to_decrease);
                }
                else{
                    sum = (-sum) & (-dynamic_to_decrease);
                    sum = (sum ^ -1) + 1;
                }
          }
          else{ //like non dynamic
             sum += compute;
          }

          if (sum >= max_all){
            sum = max_all;
          }
          else if (sum <= min_all){
            sum = min_all;
          }
    }



    __syncthreads();
  }

  C[by*blocksize + ty][bx*blocksize + tx] = sum;
  return;
}


std::vector<torch::Tensor> cu_gemm(
	torch::Tensor a,
	torch::Tensor b,
	int bits_allowed,
	int bits_to_all) {
  //printf("%f \n",compute);
  //printf("inside KERNEL cude \n");

  torch::Device device = torch::kCUDA;
  auto output = torch::zeros({a.size(0), b.size(0)}, device);

  const dim3 threads(blocksize, blocksize);
  const dim3 grid(DIVCEIL(output.size(1), threads.x), DIVCEIL(output.size(0), threads.y));

  AT_DISPATCH_FLOATING_TYPES(a.scalar_type(), "_gemm", ([&] {
    gemm_kernel<scalar_t><<< grid, threads >>>(
      a.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
      b.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
      output.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
      bits_allowed,
      bits_to_all
    );
  }));

  cudaError_t code = cudaGetLastError();
  if (code != cudaSuccess) {
    const char * errorMessage = cudaGetErrorString(code);
    fprintf(stderr, "CUDA error: (%d) %s\n", code, errorMessage);
  }

  return {output};
}