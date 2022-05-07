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

static void cu_fft_single(benchmark::State &state)
{
  int N = state.range(0);

  // Allocate host memory for the signal and result
  cufftComplex *h_signal = (cufftComplex *)malloc(sizeof(cufftComplex) * N);
  cufftComplex *h_fft = (cufftComplex *)malloc(sizeof(cufftComplex) * N);

  //  Allocate complex signal GPU device memory
  cufftComplex *d_signal;

  //  Generate signal
  generate_signal(h_signal, N);

  for (auto _ : state)
  {

    checkCudaErrors(cudaMalloc((void **)&d_signal, N * sizeof(cufftComplex)));
    //  Init CUFFT Plan
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cufftHandle plan;
    cudaEventRecord(start, 0);
    checkCudaErrors(cufftPlan1d(&plan, N, CUFFT_C2C, 1));
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    // Copy host memory to device
    checkCudaErrors(cudaMemcpy(d_signal, h_signal, N * sizeof(cufftComplex), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaDeviceSynchronize());
    //  Start iteration timer
    // auto start = std::chrono::high_resolution_clock::now();

    // Transform signal to fft (inplace)
    checkCudaErrors(cufftExecC2C(plan, (cufftComplex *)d_signal, (cufftComplex *)d_signal, CUFFT_FORWARD));

    // checkCudaErrors(cudaDeviceSynchronize());

    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop);
    //  Calculate elapsed time
    // auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - start);

    // Copy device memory to host fft
    // checkCudaErrors(cudaMemcpy(h_fft, d_signal, N * sizeof(cufftComplex), cudaMemcpyDeviceToHost));

    //  Set Iteration Time
    // state.SetIterationTime(elapsed_seconds.count());
    state.SetIterationTime(elapsed / 1000.0f);
    checkCudaErrors(cudaFree(d_signal));
    // Destroy CUFFT context
    checkCudaErrors(cufftDestroy(plan));
  }

  // Cleanup memory

  free(h_signal);
  free(h_fft);

  //  Save statistics
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * N);
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * N * sizeof(cufftComplex));
  state.SetComplexityN(N);
}
BENCHMARK(cu_fft_single)->RangeMultiplier(2)->Range(1 << 10, 1 << 30)->Complexity()->UseManualTime();
BENCHMARK_MAIN();
