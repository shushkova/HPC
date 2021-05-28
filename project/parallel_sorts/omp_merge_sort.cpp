#include <stdio.h>
#include <vector>
#include <iterator>
#include <ctime>
#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <limits>
#include"omp.h"

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

void merge_sort(vector<int>::iterator vec_beg, vector<int>::iterator vec_end, int threads) {
    int size_ = vec_end - vec_beg;
    if (size_ < BLOCK_SIZE) {
        insertion_sort(vec_beg, vec_end);
        return;
    }
    if (threads == 1) {
        merge_sort_serial(vec_beg, vec_end);
        return;
    }
    int t = (vec_end - vec_beg) / 2;
    vector<int>::iterator m = vec_beg + t;
    int t1 = threads / 2;
    int t2 = threads  - t1;
#pragma omp parallel
    {
#pragma omp single
        {
#pragma omp task
            {
                merge_sort(vec_beg, m, t1);
            }
#pragma omp task
            {
                merge_sort(m, vec_end, t2);
            }
        }
    }
    vector<int> v = merge(vec_beg, m, vec_end);
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
    std::vector<int> numbers_merge(len);

    for (int i = 0; i != len; ++i) {
        int t = (std::rand() - std::rand()) * (std::rand() / 1000);
        numbers_merge[i] = t;
    }

    vector<int> c(len);
    std::copy(numbers_merge.begin(), numbers_merge.end(), c.begin());
    std::sort(c.begin(), c.end());

    int num_threads = *argv[1] - '0';

    omp_set_num_threads(num_threads);

    unsigned int start_time_merge = omp_get_wtime();
    merge_sort(numbers_merge.begin(), numbers_merge.end(), num_threads);
    unsigned int end_time_merge = omp_get_wtime();
    int time_merge = end_time_merge - start_time_merge;


    std::cout << "Parallel finished" << '\n';
    std::cout << "len = " << len << '\n';
    std::cout << "num threads = " << num_threads << '\n';
    std::cout << "Time of parallel omp merge sort = " << time_merge << "\n";
    std::cout << "Result vector is sorted: " << std::is_sorted(numbers_merge.begin(), numbers_merge.end()) << std::endl;
    std::cout << "Result vector equals sorted initial vector: " << (c == numbers_merge) << std::endl;
}

