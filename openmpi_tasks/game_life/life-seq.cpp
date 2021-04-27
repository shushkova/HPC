#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <chrono>

using namespace std;

int rule(int number) {
    if (number == 0)
        return 0;
    else if (number == 1)
        return 1;
    else if (number == 10)
        return 1;
    else if (number == 11)
        return 1;
    else if (number == 100)
        return 1;
    else if (number == 101)
        return 0;
    else if (number == 110)
        return 0;
    else if (number == 111)
        return 0;
    return -1;
}

int getNext(int *arr) {
    int res = 0;
    int order = 1;
    for (int i = 2; i >= 0; --i) {
        res += arr[i] * order;
        order *= 10;
    }
    return rule(res);
}

void print_array(int* array, int size) {
    for (int i = 0; i < size; i++)
        if (array[i] == 1)
            cout << array[i] << " ";
        else
            cout << " " << " ";

    cout << endl;
}

int main(int argc, char ** argv) {
    int data[] = { 0, 0, 0,0, 0, 0,0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0};
    int n = sizeof(data) / sizeof(data[0]);

    int local[n+2];
    copy(data, data + n, local + 1);

    int result[n];
    auto t1 = chrono::high_resolution_clock::now();
    for (int j = 0; j < 20; ++j) {
        local[0] = local[n];
        local[n+1] = local[1];

        for (int i = 0; i < n; ++i)
            result[i] = getNext(&local[i]);
        print_array(result, n);
        copy(result, result + n, local + 1);
    }
    auto t2 = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();

    cout << duration << endl;
    return 0;
}