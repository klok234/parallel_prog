#define CHECK_RES      
// #define SAVE_MATRIX       

#include <iostream>
#include <fstream>
#include <vector>
#include "Matrix.h"

using namespace std;

int main(int argc, char** argv)
{
    cout << "Lab4 (CUDA): by Dolzhikov D.A. 6212-100503D\n\n";

    ifstream fin("input.txt");
    if (!fin.is_open())
        throw std::exception("Failed open input.txt");

    ofstream fout("output.txt", ios::app);
    ofstream csv("experiments.csv", ios::trunc);
    csv << "size,time_us,block_x,block_y,shared_mem\n";

    while (!fin.eof())
    {
        size_t n = 0;
        fin >> n;
        if (n == 0) break;

        vector<int> tmp(n * n);
        for (size_t i = 0; i < n * n; ++i) fin >> tmp[i];
        Matrix<int> a(n, n, tmp.data());
        for (size_t i = 0; i < n * n; ++i) fin >> tmp[i];
        Matrix<int> b(n, n, tmp.data());

        cout << "Processing matrix " << n << "x" << n << "...\n";

        vector<dim3> blocks = { dim3(8,8), dim3(16,16), dim3(32,32) };
        vector<bool> shared_flags = { true, false };

        for (dim3 block : blocks)
        {
            for (bool shared : shared_flags)
            {
                cout << "  block " << block.x << "x" << block.y
                    << ", shared=" << (shared ? "yes" : "no") << "... ";
                try {
                    stats<int> res = multiply_matrix_cuda(a, b, block, shared);
                    fout << res;
                    res.to_plot();
                    res.to_csv();
                    cout << "done (" << res.duration.count() << " us)\n";
                }
                catch (const exception& e) {
                    cout << "ERROR: " << e.what() << "\n";
                }
            }
        }
    }

    fin.close();
    fout.close();
    cout << "Complete!\n";
    return 0;
}
