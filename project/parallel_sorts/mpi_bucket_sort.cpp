#include <stdio.h>
#include <vector>
#include <iterator>
#include <ctime>
#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <limits>
#include <mpi.h>
#include <cmath>

using std::vector;

#define BLOCK_SIZE 50

void insertion_sort(vector<int>::iterator vec_beg, vector<int>::iterator vec_end);

void print_vector(vector<int>::iterator vec_beg, vector<int>::iterator vec_end) {
    for (auto i = vec_beg; i < vec_end; ++i) {
        std::cout << *i << " ";
    }
    std::cout << std::endl;
}

vector<int> merge(vector<int>::iterator v_beg, vector<int>::iterator v_mid, vector<int>::iterator v_end) {
    auto v1_beg = v_beg, v1_end = v_mid, v2_beg = v_mid, v2_end = v_end;
    vector<int> res(v1_end - v1_beg + v2_end - v2_beg);
    int i = 0;
    while (v1_end - v1_beg && v2_end - v2_beg) {
        if (*v1_beg >= *v2_beg) {
            res[i] = *v2_beg;
            ++v2_beg;
        } else {
            res[i] = *v1_beg;
            ++v1_beg;
        }
        ++i;
    }
    if (v1_beg != v1_end) {
        while (v1_beg != v1_end) {
            res[i] = *v1_beg;
            ++v1_beg;
            ++i;
        }
    }
    if (v2_beg != v2_end) {
        while (v2_beg != v2_end) {
            res[i] = *v2_beg;
            ++v2_beg;
            ++i;
        }
    }
    return res;
}

void merge_sort_serial(vector<int>::iterator vec_beg, vector<int>::iterator vec_end) {
    int size_ = vec_end - vec_beg;
    if (size_ < BLOCK_SIZE) {
        insertion_sort(vec_beg, vec_end);
        return;
    }
    int t = (vec_end - vec_beg) / 2;
    vector<int>::iterator m = vec_beg + t;
    merge_sort_serial(vec_beg, m);
    merge_sort_serial(m, vec_end);
    vector<int> v = merge(vec_beg, m, vec_end);
    std::copy(v.begin(), v.end(), vec_beg);
}

void bucket_sort(vector<int>::iterator vec_beg, vector<int>::iterator vec_end) {
    double time1;
    int count = 0;
    int prank;
    int psize;
    MPI_Status status;
    MPI_Comm_rank(MPI_COMM_WORLD, &prank);
    MPI_Comm_size(MPI_COMM_WORLD, &psize);

    time1 = MPI_Wtime();
    int* counters = new int[psize]();
    int* displ = new int[psize]();

    int n = vec_end - vec_beg;

    vector<int> bucket;

    MPI_Bcast((void*) &*vec_beg, n, MPI_INT, 0, MPI_COMM_WORLD);

    int m = RAND_MAX - 1;
    vector<int>::iterator tmp;
    for (int i = 0; i < n; i++)
    {
        tmp = std::next(vec_beg, i);
        int value = *tmp;
        if (prank * m / psize <= value && value <  ((double)(prank + 1) / psize) * m) {
            bucket.push_back(value);
            ++count;
        }

    }

    merge_sort_serial(bucket.begin(), bucket.end());

    MPI_Gather(&count, 1, MPI_INT, counters, 1, MPI_INT, 0, MPI_COMM_WORLD);

    displ[0] = 0;
    for (int i = 0; i < psize - 1; i++)
    {
        displ[i + 1] = displ[i] + counters[i];
    }

    vector<int> v(vec_end - vec_beg);
    MPI_Gatherv((void*) &bucket[0], bucket.size(), MPI_INT, (void*) &v[0], counters, displ, MPI_INT, 0, MPI_COMM_WORLD);
    if (prank == 0)
        std::copy(v.begin(), v.end(), vec_beg);
}

void insertion_sort(vector<int>::iterator vec_beg, vector<int>::iterator vec_end)
{
    if(vec_beg == vec_end)
        return;
    for(auto i = vec_beg + 1; i < vec_end; ++i)
    {
        auto k = *i;
        auto j = i - 1;
        while(j >= vec_beg && *j > k)
        {
            *(j + 1) = *j;
            j--;
        }
        *(j + 1) = k;
    }
}


int main(int argc, char* argv[]) {
    std::srand(time(0));
    int len = 10000000;
    std::vector<int> numbers_bucket(len);

    for (int i = 0; i != len; ++i) {
        int t = std::rand();
        numbers_bucket[i] = t;
    }

    vector<int> c(len);
    std::copy(numbers_bucket.begin(), numbers_bucket.end(), c.begin());
    std::sort(c.begin(), c.end());

    int num_threads = *argv[1] - '0';

    MPI_Init(&argc, &argv);
    int prank;
    MPI_Comm_rank(MPI_COMM_WORLD, &prank);

    double start_time_bucket = MPI_Wtime();
    bucket_sort(numbers_bucket.begin(), numbers_bucket.end());
    if (prank == 0) {
        double end_time_bucket = MPI_Wtime();
        double time_bucket = end_time_bucket - start_time_bucket;

        std::cout << "Parallel finished" << '\n';
        std::cout << "len = " << len << '\n';
        std::cout << "num threads = " << num_threads << '\n';
        std::cout << "Time of parallel mpi bucket sort = " << time_bucket << "\n";
        std::cout << "Result vector is sorted: " << std::is_sorted(numbers_bucket.begin(), numbers_bucket.end()) << std::endl;
        std::cout << "Result vector equals sorted initial vector: " << (c == numbers_bucket) << std::endl;
    }

    MPI_Finalize();
}
