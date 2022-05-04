#include <cuda_runtime.h>
#include <cufft.h>
#include <helper_cuda.h>
#include <benchmark/benchmark.h>
#include <math.h>
#include <chrono>

#define REAL 0
#define IMAG 1

void generate_signal(cufftComplex *signal, const int N)
{
  int i;
  for (i = 0; i < N; ++i)
  {
    double theta = (double)i / (double)N * M_PI;
    signal[i].x = 1.0 * cos(10.0 * theta) + 0.5 * cos(25.0 * theta);
    signal[i].y = 1.0 * sin(10.0 * theta) + 0.5 * sin(25.0 * theta);
  }
}

static void cu_fft_single_2d(benchmark::State &state)
{
  int N = state.range(0);

  // Allocate host memory for the signal and result
  cufftComplex *h_signal = (cufftComplex *)malloc(sizeof(cufftComplex) * N * N);
  cufftComplex *h_fft = (cufftComplex *)malloc(sizeof(cufftComplex) * N * N);

  //  Allocate complex signal GPU device memory
  cufftComplex *d_signal;
  checkCudaErrors(cudaMalloc((void **)&d_signal, N * N * sizeof(cufftComplex)));

  //  Init CUFFT Plan
  cufftHandle plan;
  checkCudaErrors(cufftPlan2d(&plan, N, N, CUFFT_C2C));

  //  Generate signal
  generate_signal(h_signal, N);

  for (auto _ : state)
  {

    // Copy host memory to device
    checkCudaErrors(cudaMemcpy(d_signal, h_signal, N * N * sizeof(cufftComplex), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaDeviceSynchronize());
    //  Start iteration timer
    // auto start = std::chrono::high_resolution_clock::now();
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // Transform signal to fft (inplace)
    checkCudaErrors(cufftExecC2C(plan, (cufftComplex *)d_signal, (cufftComplex *)d_signal, CUFFT_FORWARD));

    // checkCudaErrors(cudaDeviceSynchronize());
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop);

    //  Calculate elapsed time
    // auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - start);

    // Copy device memory to host fft
    // checkCudaErrors(cudaMemcpy(h_fft, d_signal, N*N * sizeof(cufftComplex), cudaMemcpyDeviceToHost));

    //  Set Iteration Time
    // state.SetIterationTime(elapsed_seconds.count());
    state.SetIterationTime(elapsed / 1000.0f);
  }

  // Destroy CUFFT context
  checkCudaErrors(cufftDestroy(plan));

  // Cleanup memory
  checkCudaErrors(cudaFree(d_signal));
  free(h_signal);
  free(h_fft);

  //  Save statistics
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * N * N);
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * N * N * sizeof(cufftComplex));
  state.SetComplexityN(N);
}
BENCHMARK(cu_fft_single_2d)->RangeMultiplier(2)->Range(1 << 5, 1 << 15)->Complexity()->UseManualTime();
BENCHMARK_MAIN();
