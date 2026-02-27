#define CHECK_RES

#include <iostream>
#include "Matrix.h"

using namespace std;

int main()
{
    cout << "Lab1: by Dolzhikov D.A. 6212-100503D\n\n";
    ifstream fin;
    fin.open("input.txt");

    if (!fin.is_open())
    {
        throw std::exception("Failed open file");
    }

    ofstream fout;
    fout.open("output.txt");
    cout << "Starting successful\n";
    while (!fin.eof())
    {
        cout << "Working...\n";
        size_t n = 0;
        fin >> n;
        if (n == 0)
        {
            break;
        }
        int* tmp = new int[n*n];
        for (size_t i = 0; i < n*n; i++)
        {
            fin >> tmp[i];
        }
        Matrix<int> a(n, n, tmp);
        for (size_t i = 0; i < n*n; i++)
        {
            fin >> tmp[i];
        }
        Matrix<int> b(n, n, tmp);

        stats<int> res = multiply_matrix(a, b);

        fout << res;

        res.to_plot();

        delete[] tmp;

    }

    fout.close();
    fin.close();
    cout << "Copmlite!\n";
    return 0;
}
