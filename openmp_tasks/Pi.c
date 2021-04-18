#include <iostream>
#include <random>
#include <omp.h>
#include <stdlib.h>
#include <chrono>

int main() {
    int max_threads = omp_get_max_threads();
    unsigned int seeds [max_threads];
    int tid;
    unsigned int seed;


#pragma omp parallel private(tid) num_threads(max_threads)
    {
        tid = omp_get_thread_num();
        unsigned seed = (unsigned int) time(NULL);
        seed = (seed & 0xFFFFFFF0) | (tid + 1);
        seeds[tid] = seed;
    }

    int throws = 100000000, insideCircle = 0;
    double pi, randX, randY;
    int i;
    double start_time, end_time;

    start_time = omp_get_wtime();

#pragma omp parallel private(tid,randX, randY, seed)  num_threads(max_threads) reduction(+:insideCircle)
{
    tid = omp_get_thread_num();
    seed = seeds[tid];
    srand(seed);
    #pragma omp for
    for (i = 0; i < throws; ++i) {
        randX = (double) rand_r(&seed) / RAND_MAX;
        randY = (double) rand_r(&seed) / RAND_MAX;

        if ((randX * randX + randY * randY) < 1.0) ++insideCircle;
    }
}
    end_time = omp_get_wtime();
    printf("Time: %f \n", end_time-start_time);
    printf("%d\n", insideCircle);
    pi = 4.0 * ((double) insideCircle / throws);
    printf("pi = %f\n", pi);

    return 0;
}
