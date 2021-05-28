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

void merge_sort(vector<int>::iterator vec_beg, vector<int>::iterator vec_end) {
    int prank;
    int psize;
    MPI_Status status;
    MPI_Comm_rank(MPI_COMM_WORLD, &prank);
    MPI_Comm_size(MPI_COMM_WORLD, &psize);

    int sub_size = (vec_end - vec_beg) / psize;

    vector<int> sub_array(sub_size);
    MPI_Scatter((void*) &*vec_beg, sub_size, MPI_INT, (void*) &sub_array[0], sub_size, MPI_INT, 0, MPI_COMM_WORLD);

    merge_sort_serial(sub_array.begin(), sub_array.end());

    int size = sub_size;
    for (int i = 0; i < std::log2(psize); ++i) {
        int range = std::pow(2, i + 1);

        if (prank % range == 0 && prank + (range / 2) < psize) {
            vector<int> recv_vec(size);
            MPI_Recv((void*) &recv_vec[0], size, MPI_INT, prank + (range / 2), 1, MPI_COMM_WORLD, &status);
            sub_array.insert(sub_array.end(), recv_vec.begin(), recv_vec.end());
        } else if (prank % range != 0) {
            MPI_Send((void*) &sub_array[0], size, MPI_INT,
                     prank - (range / 2), 1, MPI_COMM_WORLD);
            break;
        }

        sub_array = merge(sub_array.begin(), sub_array.begin() + size , sub_array.end());
        size *= 2;
    }


    if (prank == 0) {
        std::copy(sub_array.begin(), sub_array.end(), vec_beg);
    }
    MPI_Barrier(MPI_COMM_WORLD);
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

void print_vector(vector<int>::iterator vec_beg, vector<int>::iterator vec_end) {
    for (auto i = vec_beg; i < vec_end; ++i) {
        std::cout << *i << " ";
    }
    std::cout << std::endl;
}


int main(int argc, char* argv[]) {
    std::srand(time(0));
    int len = 10000000;
    std::vector<int> numbers_merge(len);

    for (int i = 0; i != len; ++i) {
        int t = (std::rand() - std::rand()) * (std::rand() / 1000);
        numbers_merge[i] = t;
    }

    vector<int> c(len);
    std::copy(numbers_merge.begin(), numbers_merge.end(), c.begin());
    std::sort(c.begin(), c.end());

    int num_threads = *argv[1] - '0';

    MPI_Init(&argc, &argv);
    int prank;
    MPI_Comm_rank(MPI_COMM_WORLD, &prank);

    double start_time_merge = MPI_Wtime();
    merge_sort(numbers_merge.begin(), numbers_merge.end());
    if (prank == 0) {
        double end_time_merge = MPI_Wtime();
        double time_merge = end_time_merge - start_time_merge;

        std::cout << "Parallel finished" << '\n';
        std::cout << "len = " << len << '\n';
        std::cout << "num threads = " << num_threads << '\n';
        std::cout << "Time of parallel mpi merge sort = " << time_merge << "\n";
        std::cout << "Result vector is sorted: " << std::is_sorted(numbers_merge.begin(), numbers_merge.end()) << std::endl;
        std::cout << "Result vector equals sorted initial vector: " << (c == numbers_merge) << std::endl;
    }

    MPI_Finalize();
}

