#include <omp.h>
#include <stdlib.h>

int main() {
  int *id = (int*)malloc(omp_get_max_threads() * sizeof(int));
#pragma omp parallel
  {
    id[omp_get_thread_num()] = omp_get_thread_num();
  }
  free(id);
  return 0;
}
