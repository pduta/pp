#include <iostream>
#include <vector>
#include <algorithm>
#include <omp.h>

using namespace std;

void computeHistogramBins(const vector<int>& data, int numBins, int minVal, int maxVal, vector<int>& binEdges) {
    int range = maxVal - minVal + 1;
    int binWidth = (range + numBins - 1) / numBins;

    binEdges.resize(numBins + 1);
    for (int i = 0; i <= numBins; ++i) {
        binEdges[i] = minVal + i * binWidth;
    }
}

void assignToBins(const vector<int>& data, const vector<int>& binEdges, vector<vector<int>>& bins) {
    int numBins = binEdges.size() - 1;
    int numThreads = omp_get_max_threads();
    vector<vector<vector<int>>> threadBins(numThreads, vector<vector<int>>(numBins));

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        #pragma omp for
        for (int i = 0; i < data.size(); ++i) {
            int value = data[i];
            for (int b = 0; b < numBins; ++b) {
                if (value >= binEdges[b] && value < binEdges[b + 1]) {
                    threadBins[tid][b].push_back(value);
                    break;
                }
            }
        }
    }

    // Merge thread-local bins into final bins
    for (int b = 0; b < numBins; ++b) {
        for (int t = 0; t < numThreads; ++t) {
            bins[b].insert(bins[b].end(), threadBins[t][b].begin(), threadBins[t][b].end());
        }
    }
}

int main() {
    int n, numBins;
    cout << "Enter number of elements: ";
    cin >> n;

    vector<int> data(n);
    cout << "Enter " << n << " integers:\n";
    for (int i = 0; i < n; ++i) {
        cin >> data[i];
    }

    cout << "Enter number of bins: ";
    cin >> numBins;

    int minVal = *min_element(data.begin(), data.end());
    int maxVal = *max_element(data.begin(), data.end());

    vector<int> binEdges;
    computeHistogramBins(data, numBins, minVal, maxVal, binEdges);

    vector<vector<int>> bins(numBins);
    assignToBins(data, binEdges, bins);

    #pragma omp parallel for
    for (int i = 0; i < numBins; ++i) {
        sort(bins[i].begin(), bins[i].end());
    }

    // Combine sorted bins
    vector<int> sortedData;
    for (const auto& bin : bins) {
        sortedData.insert(sortedData.end(), bin.begin(), bin.end());
    }

    cout << "Sorted data: ";
    for (int val : sortedData) {
        cout << val << " ";
    }
    cout << endl;

    return 0;
}
