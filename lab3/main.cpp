#define CHECK_RES
// #define SAVE_MATRIX // For saving result matirx in file

#include <iostream>
#include "Matrix.h"
#include <mpi.h>

using namespace std;

int main(int argc, char** argv)
{

    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        cout << "Lab3: by Dolzhikov D.A. 6212-100503D\n\n";
    }

    ifstream fin("input.txt");
    if (!fin.is_open())
        throw std::exception("Failed open file");

    ofstream fout;
    if (rank == 0) {
        fout.open("output.txt", ios::app);
        cout << "Starting successful with" << size << "cores\n";
    }

    while (!fin.eof())
    {
        if (rank == 0) cout << "Working...\n";
        size_t n = 0;
        if (rank == 0) fin >> n;

        MPI_Bcast(&n, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

        if (n == 0) break;

        Matrix<int> a(n, n);
        Matrix<int> b(n, n);

        int* tmp = new int[n * n];
        if (rank == 0) {
            for (size_t i = 0; i < n * n; ++i) fin >> tmp[i];
            for (size_t i = 0; i < n * n; ++i) a.data()[i] = tmp[i];
            for (size_t i = 0; i < n * n; ++i) fin >> tmp[i];
            for (size_t i = 0; i < n * n; ++i) b.data()[i] = tmp[i];
        }

        MPI_Bcast(a.data(), n * n, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(b.data(), n * n, MPI_INT, 0, MPI_COMM_WORLD);

        stats<int> res = multiply_matrix(a, b);

        if (rank == 0) {
            fout << res;
            res.to_plot();
        }

        delete[] tmp;
    }

    if (rank == 0) {
        fout.close();
        fin.close();
        cout << "Complete!\n";
    }
    return 0;
}
