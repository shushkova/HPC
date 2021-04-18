#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <chrono>

void zero_init_matrix(double ** matrix, size_t N)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            matrix[i][j] = 0.0;
        }
    }
}

void rand_init_matrix(double ** matrix, size_t N)
{
    srand(time(NULL));

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            matrix[i][j] = rand() / RAND_MAX;
        }
    }
}

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

int main()
{
    const size_t N = 1000; // size of an array
 
    double ** A, ** B, ** C; // matrices

    printf("Starting:\n");

    A = malloc_matrix(N);
    B = malloc_matrix(N);
    C = malloc_matrix(N);    

    rand_init_matrix(A, N);
    rand_init_matrix(B, N);
    zero_init_matrix(C, N);
  
    
    auto begin = std::chrono::high_resolution_clock::now();
 
    for (int i = 0; i < N; ++i)
      for (int j = 0; j < N; ++j)
        for (int c = 0; c < N; ++c)
          C[i][j] += A[i][c] * B[c][j];


    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);

    printf("Time elapsed (ijn): %ld seconds.\n", elapsed.count());

    zero_init_matrix(C, N);
    
    begin = std::chrono::high_resolution_clock::now();
 
    int i, j, k;


#pragma omp parallel for shared(A, B, C) collapse(3)
    for(i = 0; i < N; i++)
        for(j = 0; j < N; j++)
            for(k = 0; k < N; k++)
                C[i][j] += A[i][k] *  B[k][j];

    end = std::chrono::high_resolution_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);

    printf("Time elapsed (ijn): %ld seconds.\n", elapsed.count());


    free_matrix(A, N);
    free_matrix(B, N);
    free_matrix(C, N);

    return 0;
}
