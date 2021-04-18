#include <iostream>
#include <cmath>
#include <omp.h>
#include <chrono>

const int N = 1000;
double EPS = 0.000000001;


double ** malloc_matrix(size_t N)
{
    double ** matrix = (double **)malloc(N * sizeof(double *));

    for (int i = 0; i < N; ++i)
    {
        matrix[i] = (double *)malloc(N * sizeof(double));
    }

    return matrix;
}

void free_matrix(double ** matrix, size_t N)
{
    for (int i = 0; i < N; ++i)
    {
        free(matrix[i]);
    }

    free(matrix);
}


bool error(double* previous, double*  current)
{
    double norm = 0;
    for (int i = 0; i < N; i++)
        norm += (current[i] - previous[i]) * (current[i] - previous[i]);
    return (sqrt(norm) < EPS);
}


void gauss_seidel_omp(double **a, double *b, double *current){

        double previous[N];

    for (int i = 0; i < N; i++)
        previous[i] = 0;

    do {
        double t;
        for (int i = 0; i < N; i++)
            previous[i] = current[i];

#pragma omp parallel for private(t)
        for (int i = 0; i < N; i++) {
            t = 0;
            for (int j = 0; j < i; j++)
                t += a[i][j] * current[j];
            for (int j = i + 1; j < N; j++)
                t += a[i][j] * previous[j];
            current[i] = (b[i] - t) / a[i][i];
        }
    }
    while (!error(current, previous));
}


void gauss_seidel(double **a, double *b, double *current){

    double previous[N];

    for (int i = 0; i < N; i++)
        previous[i] = 0;

    do {
        double t;
        for (int i = 0; i < N; i++)
            previous[i] = current[i];


        for (int i = 0; i < N; i++) {
            t = 0;
            for (int j = 0; j < i; j++)
                t += a[i][j] * current[j];
            for (int j = i + 1; j < N; j++)
                t += a[i][j] * previous[j];
            current[i] = (b[i] - t) / a[i][i];
        }
    }
    while (!error(current, previous));

}


int main()
{
    double start_time, end_time;
    double **a;
    double *b;
    double *current;

    a = malloc_matrix(N);
    b = (double *)malloc(N * sizeof(double));
    current = (double *)malloc(N * sizeof(double));

    for (int i=0;i < N; ++i){
        for (int j=0; j < N;++j){
            if (i==j)
                a[i][j]=N;
            else
                a[i][j]=1;
        }
        b[i]=i;
        current[i] = 0;
    }

    start_time = omp_get_wtime();
    gauss_seidel(a, b, current);
    end_time = omp_get_wtime();
    std::cout << end_time - start_time << std::endl;

    for (int i = 0; i < N; i++)
        current[i] = 0;
    start_time = omp_get_wtime();
    gauss_seidel_omp(a, b, current);
    end_time = omp_get_wtime();
    std::cout << end_time - start_time << std::endl;
}


