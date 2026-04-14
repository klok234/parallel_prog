#ifndef MATRIX
#define MATRIX

#include <iostream>
#include <exception>
#include <fstream>
#include <chrono>
#include <cuda_runtime.h>


using namespace std;


template <typename T>
__global__ void matmul_kernel_shared(const T* A, const T* B, T* C, size_t N)
{
    extern __shared__ T shared_mem[];
    T* sA = shared_mem;
    T* sB = &shared_mem[blockDim.x * blockDim.y];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;

    T sum = 0;
    for (int tile = 0; tile < (N + blockDim.x - 1) / blockDim.x; ++tile)
    {
        if (row < N && tile * blockDim.x + tx < N)
            sA[ty * blockDim.x + tx] = A[row * N + tile * blockDim.x + tx];
        else
            sA[ty * blockDim.x + tx] = 0;

        if (col < N && tile * blockDim.y + ty < N)
            sB[ty * blockDim.x + tx] = B[(tile * blockDim.y + ty) * N + col];
        else
            sB[ty * blockDim.x + tx] = 0;

        __syncthreads();

        for (int k = 0; k < blockDim.x; ++k)
            sum += sA[ty * blockDim.x + k] * sB[k * blockDim.x + tx];

        __syncthreads();
    }

    if (row < N && col < N)
        C[row * N + col] = sum;
}

template <typename T>
__global__ void matmul_kernel_naive(const T* A, const T* B, T* C, size_t N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N)
    {
        T sum = 0;
        for (int k = 0; k < N; ++k)
            sum += A[row * N + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

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
    std::chrono::microseconds duration = 0us;
    bool is_correct = false;
    dim3 grid_size;
    dim3 block_size;
    bool used_shared_mem = false;

    void to_plot()
    {
        ofstream fout("to_plot.txt", ios::app);
        if (!fout.is_open()) throw std::exception("Failed to save result");
        fout << matrix.cols() << " " << duration.count() << " "
            << block_size.x << "x" << block_size.y << " "
            << (used_shared_mem ? "shared" : "naive") << "\n";
        fout.close();
    }

    void to_csv(const string& filename = "experiments.csv")
    {
        ofstream fout(filename, ios::app);
        if (!fout.is_open()) return;
        fout << matrix.rows() << "," << duration.count() << ","
            << block_size.x << "x" << block_size.y << ","
            << used_shared_mem << "\n";
        fout.close();
    }
};

template <class T>
std::ostream& operator<<(std::ostream& os, const stats<T>& s)
{
#ifdef SAVE_MATRIX
    os << s.matrix;
#endif
    os << "Duration: " << s.duration.count() << " us\n";
    os << "Grid: (" << s.grid_size.x << "," << s.grid_size.y
        << ")  Block: (" << s.block_size.x << "," << s.block_size.y << ")\n";
    os << "Shared memory used: " << (s.used_shared_mem ? "yes" : "no") << "\n";
    os << "Correct: " << s.is_correct << "\n";
    os << "Count of matrix elements: " << s.matrix.rows() * s.matrix.cols() << "\n\n";
    return os;
}

template <typename T>
stats<T> multiply_matrix_cuda(const Matrix<T>& a, const Matrix<T>& b,
    dim3 block_size = dim3(16, 16),
    bool use_shared_mem = true)
{
    if (a.cols() != b.rows() || a.rows() != a.cols() || b.rows() != b.cols())
        throw std::invalid_argument("Only square matrices");

    size_t N = a.rows();
    stats<T> res;
    res.block_size = block_size;
    res.used_shared_mem = use_shared_mem;

    dim3 grid_size((N + block_size.x - 1) / block_size.x,
        (N + block_size.y - 1) / block_size.y);
    res.grid_size = grid_size;

    T* d_A = nullptr, * d_B = nullptr, * d_C = nullptr;
    size_t bytes = N * N * sizeof(T);

    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, b.data(), bytes, cudaMemcpyHostToDevice);

    auto start = chrono::high_resolution_clock::now();

    if (use_shared_mem)
    {
        size_t shared_size = block_size.x * block_size.y * 2 * sizeof(T);
        matmul_kernel_shared<T> << <grid_size, block_size, shared_size >> > (d_A, d_B, d_C, N);
    }
    else
    {
        matmul_kernel_naive<T> << <grid_size, block_size >> > (d_A, d_B, d_C, N);
    }
    cudaDeviceSynchronize();

    Matrix<T> result(N, N);
    cudaMemcpy(result.data(), d_C, bytes, cudaMemcpyDeviceToHost);

    auto stop = chrono::high_resolution_clock::now();
    res.duration = chrono::duration_cast<chrono::microseconds>(stop - start);
    res.matrix = result;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

#ifdef CHECK_RES
    cout << "Checking result with CPU multiplication...\n";
    Matrix<T> cpu_res = a.mul_cpu(b);
    bool ok = true;
    for (size_t i = 0; i < N && ok; ++i)
        for (size_t j = 0; j < N; ++j)
            if (abs(cpu_res(i, j) - result(i, j)) > 1e-5)
            {
                ok = false; break;
            }
    res.is_correct = ok;
#endif

    return res;
}

#endif
