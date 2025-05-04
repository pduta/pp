#include <iostream>
#include <vector>
#include <omp.h>

using namespace std;

void matrixMultiply(const vector<vector<double>> &A,
                    const vector<vector<double>> &B,
                    vector<vector<double>> &C) {
    int n = A.size();
    int m = B[0].size();
    int common = B.size();

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            C[i][j] = 0;
            for (int k = 0; k < common; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

int main() {
    int n = 3, m = 3;

    vector<vector<double>> A = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };

    vector<vector<double>> B = {
        {9, 8, 7},
        {6, 5, 4},
        {3, 2, 1}
    };

    vector<vector<double>> C(n, vector<double>(m, 0));

    matrixMultiply(A, B, C);

    cout << "Result matrix C:" << endl;
    for (const auto &row : C) {
        for (double val : row) {
            cout << val << " ";
        }
        cout << endl;
    }

    return 0;
}
