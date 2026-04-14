#ifndef MATRIX
#define MATRIX

#include <iostream>
#include <exception>
#include <fstream>
#include <chrono>
#include <mpi.h>

using namespace std;

template <class T>
class Matrix {
    size_t _rows = 0;
    size_t _cols = 0;
    T* _value = nullptr;

public:
    Matrix() = default;
    Matrix(const Matrix<T>& other)
    {
        _rows = other._rows;
        _cols = other._cols;
        _value = new T[_rows*_cols];
        for (size_t i = 0; i < _rows*_cols; i++)
        {
            _value[i] = other._value[i];
        }
    }

    Matrix(size_t rows, size_t cols, T* values) : _rows(rows), _cols(cols)
    {
        _value = new T[_rows*_cols];
        for (size_t i = 0; i < _rows*_cols; i++)
        {
            _value[i] = values[i];
        }
    }
    Matrix(size_t rows, size_t cols) : _rows(rows), _cols(cols)
    {
        _value = new T[_rows * _cols](0);
    }

    size_t rows() const
    {
        return _rows;
    }
    size_t cols() const
    {
        return _cols;
    }

    T operator()(const size_t rows, const size_t cols) const
    {
        if (rows >= _rows  || cols >= _cols)
        {
            throw std::range_error("Index out of range");
        }
        return _value[rows*_cols + cols];
    }
    T& operator()(const size_t rows, const size_t cols)
    {
        if (rows >= _rows || cols >= _cols)
        {
            throw std::range_error("Index out of range");
        }
        return _value[rows * _cols + cols];
    }

    T* data()
    {
        return _value;
    }

    Matrix<T>& operator=(const Matrix<T>& rhs)
    {
        Matrix<T> tmp(rhs);
        this->_rows = tmp._rows;
        this->_cols = tmp._cols;
        std::swap(_value, tmp._value);
        return *this;
    }

    Matrix<T> operator*(const Matrix<T>& rhs) const
    {
        Matrix<T> result(_rows, _cols);
        if (_rows != rhs._rows && _cols != rhs._cols)
        {
            throw std::length_error("The dimensions are not equal");
        }
        else
        {
            for (int i = 0; i < _rows; i++)
            {
                for (int j = 0; j < rhs._cols; j++)
                {
                    T sum = 0;
                    for (int k = 0; k < _cols; k++)
                    {
                        sum += this->operator()(i, k) * rhs(k, j);
                    }
                    result(i, j) = sum;
                }
            }
        }
        return result;
    }
    ~Matrix()
    {
        delete[] _value;
    }
};

template <typename T>
std::ostream& operator<<(std::ostream& os, const Matrix<T>& mat)
{
    for (size_t i = 0; i < mat.rows(); i++)
    {
        for (size_t j = 0; j < mat.cols(); j++)
        {
            os << mat(i, j) << " ";
        }
        os << "\n";
    }
    return os;
}

template <class T>
struct stats
{
    Matrix<T> matrix;
    std::chrono::milliseconds duration = 0ms;
    int threads = 1;
    bool is_correct = false;

    void to_plot()
    {
        ofstream fout;
        fout.open("to_plot.txt", ios::app);

        if (!fout.is_open())
        {
            throw std::exception("Failed to save result");
        }
        fout << matrix.cols() << " " << duration.count() << " " << threads << "\n";

        fout.close();
    }

};

template <class T>
std::ostream& operator<<(std::ostream& os, const stats<T>& s)
{
#ifdef SAVE_MATRIX
    os << s.matrix;
#endif
    os << "Duration: " << s.duration << "\n";
    os << "Threads: " << s.threads << "\n";
    os << "Correct: " << s.is_correct << "\n";
    os << "Count of matrix elements: " << s.matrix.rows() * s.matrix.cols() << "\n\n";
    return os;
}

template <class T>
stats<T> multiply_matrix(Matrix<T>& a, Matrix<T>& b)
{

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (a.cols() != b.cols() && a.rows() != a.cols() && b.rows() != b.cols())
    {
        throw std::invalid_argument("Matrix must be N*N!");
    }

    stats<T> res;
    if (rank == 0) std::cout << "Start multiply with " << size << " processes\n";
    MPI_Barrier(MPI_COMM_WORLD);

    auto start = chrono::high_resolution_clock::now();

    int rows_per_proc = a.rows() / size;
    int remainder = a.rows() % size;
    int start_row, end_row;


    if (rank < remainder) {
        start_row = rank * (rows_per_proc + 1);
        end_row = start_row + rows_per_proc + 1;
    }
    else {
        start_row = rank * rows_per_proc + remainder;
        end_row = start_row + rows_per_proc;
    }

    int loc_rows = end_row - start_row;
    Matrix<T> loc_result(loc_rows, a.cols());


    for (int i = 0; i < loc_rows; i++)
    {
        int global_i = start_row + i;
        for (int j = 0; j < b.cols(); j++)
        {
            T sum = 0;
            for (int k = 0; k < a.cols(); k++)
            {
                sum += a(global_i, k) * b(k, j);
            }
            loc_result(i, j) = sum;
        }
    }
    Matrix<T> full_result;
    if (rank == 0) full_result = Matrix<T>(a.rows(), b.cols());

    int* recvcounts = nullptr, * displs = nullptr;
    if (rank == 0) {
        recvcounts = new int[size];
        displs = new int[size];
        int offset = 0;
        for (int p = 0; p < size; ++p) {
            int p_rows = (p < remainder) ? rows_per_proc + 1 : rows_per_proc;
            recvcounts[p] = p_rows * b.cols();
            displs[p] = offset;
            offset += recvcounts[p];
        }
    }

    MPI_Gatherv(loc_result.data(), loc_rows * b.cols(), MPI_INT,
        (rank == 0) ? full_result.data() : nullptr,
        recvcounts, displs, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    auto stop = chrono::high_resolution_clock::now();
    if (rank == 0) {
        res.matrix = full_result;
        res.threads = size;
        std::cout << "Finish multiply\n";
        res.duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
#ifdef CHECK_RES
        res.is_correct = compare_matrix(a, b, res.matrix);
#endif
        delete[] recvcounts;
        delete[] displs;
    }
    return res;
}

template <class T>
bool compare_matrix(Matrix<T>& a, Matrix<T>& b, Matrix<T>& res)
{
    ofstream fout;
    cout << "Cheking result\n";
    fout.open("to_cmp.txt");

    fout << a.cols() << "\n";
    for (size_t i = 0; i < a.rows(); i++)
    {
        for (size_t j = 0; j < a.cols(); j++)
        {
            fout << a(i, j) << " ";
        }
        fout << "\n";
    }
    for (size_t i = 0; i < b.rows(); i++)
    {
        for (size_t j = 0; j < b.cols(); j++)
        {
            fout << b(i, j) << " ";
        }
        fout << "\n";
    }
    for (size_t i = 0; i < res.rows(); i++)
    {
        for (size_t j = 0; j < res.cols(); j++)
        {
            fout << res(i, j) << " ";
        }
        fout << "\n";
    }

    fout.close();
    int code = system("python cheker.py to_cmp.txt");

    if (code == 0)
    {
        return true;
    }
    if (code == 1)
    {
        return false;
    }
    else
    {
        throw std::exception("Something went wrong");
    }
}


#endif
