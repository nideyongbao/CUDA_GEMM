// softmax bench — cudaEvent timing + effective HBM bandwidth (softmax is memory
// bound). Ideal traffic = read x + write y = 2*M*N*4 bytes; GB/s = ideal/time.
// Fewer passes / better coalescing -> higher effective GB/s. H20 HBM3 ~4000 GB/s.
// Peak-BW divisor defaults to H20 (4000) but is overridable via env PEAK_BW_GBS,
// so the same binary reports honest %-of-peak on any card (single source: gpu_specs.py).
//   usage: bench <id> [M N]   (default 8192 8192)
#include "softmax.h"
#include <cstdlib>

int main(int argc, char** argv) {
  const char* pk = getenv("PEAK_BW_GBS");
  double peak_bw = pk ? atof(pk) : 4000.0;   // H20 HBM3 default
  if (argc < 2) { softmax_print_registry(); return 0; }
  int id = atoi(argv[1]);
  int M = argc > 2 ? atoi(argv[2]) : 8192;
  int N = argc > 3 ? atoi(argv[3]) : 8192;
  const SoftmaxCase* c = softmax_find(id);
  if (!c) { printf("bad id %d\n", id); softmax_print_registry(); return 1; }

  size_t n = (size_t)M * N;
  std::vector<float> hx(n);
  for (size_t i = 0; i < n; ++i) hx[i] = (float)((i * 1103515245u + 12345u) % 1000) / 100.f - 5.f;
  float *dx, *dy;
  CHECK_CUDA(cudaMalloc(&dx, n * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dy, n * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(dx, hx.data(), n * sizeof(float), cudaMemcpyHostToDevice));

  for (int i = 0; i < 5; ++i) c->fn(dx, dy, M, N, 0);  // warmup
  CHECK_CUDA(cudaDeviceSynchronize());

  cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
  const int iters = 50;
  cudaEventRecord(s);
  for (int i = 0; i < iters; ++i) c->fn(dx, dy, M, N, 0);
  cudaEventRecord(e); cudaEventSynchronize(e);
  float ms; cudaEventElapsedTime(&ms, s, e); ms /= iters;

  double ideal_bytes = 2.0 * (double)n * sizeof(float);
  double gbps = ideal_bytes / (ms * 1e-3) / 1e9;
  printf("%-20s M=%d N=%d  time=%.4f ms  eff_BW=%.1f GB/s  (%.1f%% of %.0f)\n",
         c->name, M, N, ms, gbps, gbps / peak_bw * 100.0, peak_bw);
  cudaFree(dx); cudaFree(dy);
  return 0;
}
