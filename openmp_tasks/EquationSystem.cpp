#include<stdio.h>
#include"omp.h"
#include<stdlib.h>
#include<time.h>
#include <math.h>
int n=50;
double* x;
double* y;
double a,b, calc_a, calc_b;


void Solve(){
    double mean_x=0.0, mean_y=0.0, mean_xx = 0.0, mean_xy = 0.0;

#pragma omp parallel for shared(n,x,y) reduction(+:mean_x, mean_y, mean_xx, mean_xy)
    for(int i = 0;i < n;i++){
        mean_x += x[i];
        mean_y += y[i];
        mean_xx += x[i]*x[i];
        mean_xy += x[i]*y[i];
    }
    mean_x/=n;
    mean_y/=n;
    mean_xy/=n;
    mean_xx/=n;

    calc_a = (mean_xy - mean_x*mean_y) / (mean_xx - mean_x * mean_x);
    calc_b = mean_y - calc_a * mean_x;

}


int main(){
    int i;
    x = (double *) malloc(n * sizeof(double));
    y=  (double *) malloc(n * sizeof(double));

    srand(time(NULL));

    a=4.5;
    b=-1;

    int max_threads = omp_get_max_threads();
    unsigned int seeds[max_threads];

    int tid;
#pragma omp parallel private(tid) num_threads(max_threads)
    {
        tid = omp_get_thread_num();
        unsigned seed = (unsigned int) time(NULL);
        seed = (seed & 0xFFFFFFF0) | (tid + 1);
        seeds[tid] = seed;
    }

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        unsigned seed = seeds[tid];
#pragma omp for
        for(i=0;i<n;i++){
            x[i] = ((double )rand_r(&seed))/RAND_MAX;
            double noise = (rand_r(&seed)%RAND_MAX)/10000000000;
            y[i]= a*x[i]+b+noise;
        }
    }

    double time1 = omp_get_wtime();
    Solve();

    double time2 = omp_get_wtime() - time1;
    printf("Real a=%f\n",a);
    printf("Real b=%f\n",b);

    printf("Calc a=%f\n",calc_a);
    printf("Calc b=%f\n",calc_b);

    printf("Time of running: %f\n",time2);

    free(x);
    free(y);
    return 0;
}
