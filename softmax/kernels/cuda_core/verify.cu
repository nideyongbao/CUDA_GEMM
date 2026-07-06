// softmax verify — run kernel <id>, compare against the double-precision CPU safe
// softmax. allclose: |a-b| <= atol + rtol*|b|, atol=rtol=1e-4 (fp32 softmax).
//   usage: verify <id> [M N]   (default 1024 2048)
#include "softmax.h"

int main(int argc, char** argv) {
  if (argc < 2) { softmax_print_registry(); return 0; }
  int id = atoi(argv[1]);
  int M = argc > 2 ? atoi(argv[2]) : 1024;
  int N = argc > 3 ? atoi(argv[3]) : 2048;
  const SoftmaxCase* c = softmax_find(id);
  if (!c) { printf("bad id %d\n", id); softmax_print_registry(); return 1; }

  size_t n = (size_t)M * N;
  std::vector<float> hx(n), hy(n), ref(n);
  for (size_t i = 0; i < n; ++i) hx[i] = (float)((i * 2654435761u + 7u) % 2000) / 100.f - 10.f;
  softmax_cpu_ref(hx.data(), ref.data(), M, N);

  float *dx, *dy;
  CHECK_CUDA(cudaMalloc(&dx, n * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dy, n * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(dx, hx.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  c->fn(dx, dy, M, N, 0);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(hy.data(), dy, n * sizeof(float), cudaMemcpyDeviceToHost));

  double atol = 1e-4, rtol = 1e-4, max_abs = 0, max_rel = 0;
  size_t bad = 0;
  for (size_t i = 0; i < n; ++i) {
    double a = hy[i], b = ref[i], d = fabs(a - b);
    max_abs = d > max_abs ? d : max_abs;
    double rel = d / (fabs(b) + 1e-30); max_rel = rel > max_rel ? rel : max_rel;
    if (d > atol + rtol * fabs(b)) ++bad;
  }
  printf("%-20s M=%d N=%d  max_abs=%.3e max_rel=%.3e bad=%zu/%zu  %s\n",
         c->name, M, N, max_abs, max_rel, bad, n, bad == 0 ? "PASS" : "FAIL");
  cudaFree(dx); cudaFree(dy);
  return bad == 0 ? 0 : 1;
}
