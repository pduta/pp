#include <iostream>
#include <vector>
#include <queue>
#include <limits>
#include <omp.h>

using namespace std;

const int INF = numeric_limits<int>::max();

void parallel_dijkstra(int start_node, int num_nodes, const vector<vector<pair<int, int>>>& adj, vector<int>& dist) {
    dist.assign(num_nodes, INF);
    dist[start_node] = 0;

    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq;
    pq.push({0, start_node});

    vector<bool> finalized(num_nodes, false);

    while (!pq.empty()) {
        int d = pq.top().first;
        int u = pq.top().second;
        pq.pop();

        if (finalized[u]) {
            continue;
        }
        finalized[u] = true;

        #pragma omp parallel for shared(adj, dist, finalized, pq)
        for (const auto& edge : adj[u]) {
            int v = edge.first;
            int weight = edge.second;
            if (!finalized[v]) {
                if (dist[u] != INF && dist[u] + weight < dist[v]) {
                    #pragma omp critical
                    {
                        dist[v] = dist[u] + weight;
                        pq.push({dist[v], v});
                    }
                }
            }
        }
    }
}

int main() {
    int num_nodes, num_edges, start_node;

    cout << "Enter the number of nodes in the graph: ";
    cin >> num_nodes;

    cout << "Enter the number of edges in the graph: ";
    cin >> num_edges;

    vector<vector<pair<int, int>>> adj(num_nodes);
    cout << "Enter the edges (format: source destination weight):" << endl;
    for (int i = 0; i < num_edges; ++i) {
        int u, v, weight;
        cin >> u >> v >> weight;
        if (u >= 0 && u < num_nodes && v >= 0 && v < num_nodes && weight >= 0) {
            adj[u].push_back({v, weight});
        } else {
            cerr << "Invalid edge input." << endl;
            return 1;
        }
    }

    cout << "Enter the starting node for Dijkstra's algorithm: ";
    cin >> start_node;

    if (start_node < 0 || start_node >= num_nodes) {
        cerr << "Invalid starting node." << endl;
        return 1;
    }

    vector<int> shortest_distances;
    parallel_dijkstra(start_node, num_nodes, adj, shortest_distances);

    cout << "Shortest distances from node " << start_node << ":" << endl;
    for (int i = 0; i < num_nodes; ++i) {
        cout << "To node " << i << ": ";
        if (shortest_distances[i] == INF) {
            cout << "Infinity" << endl;
        } else {
            cout << shortest_distances[i] << endl;
        }
    }

    return 0;
}