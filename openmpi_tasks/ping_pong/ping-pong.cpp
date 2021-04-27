#include <mpi.h>
#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <vector>
#include <algorithm>
#include <cstdlib>

#define TAG 30

using namespace std;

void updateData(vector<int>& data, int prank) {
    data[0]++;
    data[data[0]] = prank;
}

int pass(vector<int>& data, int psize) {
    srand(time(NULL));
    int ind = data[0];

    if (ind == psize)
        return -1;

    vector<bool> isSend (psize, false);
    for (int i = 1; i <= ind; ++i)
        isSend[data[i]] = true;

    vector<int> available;
    for (int i = 0; i < psize; ++i) {
        if (!isSend[i])
            available.push_back(i);
    }

    int index = rand() % available.size();
    return available[index];
}

int main(int argc, char ** argv) {
    int psize;
    int prank;
    double t;

    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &prank);
    MPI_Comm_size(MPI_COMM_WORLD, &psize);
    t = MPI_Wtime();

    int size = psize + 1;

    vector<int> data (psize + 1, -1);
    data[0] = 0;

    if (prank == 0)
    {
        updateData(data, prank);
        MPI_Ssend((void*) &data[0], size, MPI_INT, 1, TAG, MPI_COMM_WORLD);
    }
    else
    {
        MPI_Recv((void*) &data[0], size, MPI_INT, MPI_ANY_SOURCE, TAG, MPI_COMM_WORLD, &status);
        updateData(data, prank);

        int dstProcess = pass(data, psize);
        if (dstProcess != -1)
            MPI_Ssend((void*) &data[0], size, MPI_INT, dstProcess, TAG, MPI_COMM_WORLD);
    }

    t = MPI_Wtime() - t;
    cout << t << endl;

    MPI_Finalize();
    return 0;
}