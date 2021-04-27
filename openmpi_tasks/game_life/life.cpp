#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <mpi.h>
#define TAG 30

using namespace std;

void print_array(int* array, int size, int prank) {
    for (int i = 0; i < size; i++)
        cout << array[i] << " ";
    cout << "prank: " << prank<< endl;
}

int rule(int number) {
    if (number == 0)
        return 1;
    else if (number == 1)
        return 0;
    else if (number == 10)
        return 0;
    else if (number == 11)
        return 1;
    else if (number == 100)
        return 1;
    else if (number == 101)
        return 1;
    else if (number == 110)
        return 0;
    else if (number == 111)
        return 1;
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

int main(int argc, char ** argv) {
    int psize;
    int prank;
    double t;
    int data[] = { 0, 0, 0,0, 0, 0,0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0};

    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &prank);
    MPI_Comm_size(MPI_COMM_WORLD, &psize);
    t = MPI_Wtime();

    int size = sizeof(data) / sizeof(data[0]);
    int l = (size / psize) + (size % psize == 0 ? 0 : 1);
    int copy_data[l*psize];
    copy(data, data + size, copy_data);
    int rbuf[l];

    MPI_Scatter(copy_data, l, MPI_INT, rbuf, l, MPI_INT, 0, MPI_COMM_WORLD);

    int localBuf[l+2];
    l = prank == psize - 1 ? size % l : l;
    copy(rbuf, rbuf + l, localBuf + 1);

    int left, right;
    int for_l = (prank - 1 + psize) % psize;
    int for_r = (prank + 1) % psize;

    int result[l];

    for (int j = 0; j < 2; ++j) {
        left = localBuf[1], right = localBuf[l];

        MPI_Send(&left, 1, MPI_INT, for_l , TAG, MPI_COMM_WORLD);
        MPI_Send(&right, 1, MPI_INT, for_r, TAG + 1, MPI_COMM_WORLD);
        MPI_Recv(&right, 1, MPI_INT, for_r, TAG, MPI_COMM_WORLD, &status);
        MPI_Recv(&left, 1, MPI_INT, for_l, TAG + 1, MPI_COMM_WORLD, &status);

        localBuf[0] = left;
        localBuf[l+1] = right;

        for (int i = 0; i < l; ++i)
            result[i] = getNext(&localBuf[i]);
        print_array(result, l, prank);
        copy(result, result + l, localBuf + 1);
    }

    l = (size / psize) + (size % psize == 0 ? 0 : 1);
    MPI_Gather(result, l, MPI_INT, copy_data, l, MPI_INT, 0, MPI_COMM_WORLD);

    if (prank == 0)
        print_array(copy_data, size, prank);

    t = MPI_Wtime() - t;
    cout << t << " - prank:" << prank << endl;

    MPI_Finalize();
    return 0;
}