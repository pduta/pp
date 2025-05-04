#include <iostream>
#include <vector>
#include <queue>
#include <omp.h>

using namespace std;

void parallel_bfs(int start_node, int num_nodes, const vector<vector<int>>& adj, vector<int>& bfs_order) {
    vector<bool> visited(num_nodes, false);
    queue<int> q;
    vector<int> next_level_nodes;

    visited[start_node] = true;
    q.push(start_node);
    bfs_order.push_back(start_node); // Add the starting node to the order

    while (!q.empty()) {
        int current_level_size = q.size();
        next_level_nodes.clear();
        vector<int> local_bfs_order; // Local buffer for this level's nodes

        #pragma omp parallel for shared(adj, visited, q, next_level_nodes, local_bfs_order)
        for (int i = 0; i < current_level_size; ++i) {
            int u = -1;
            #pragma omp critical
            {
                if (!q.empty()) {
                    u = q.front();
                    q.pop();
                    local_bfs_order.push_back(u);
                }
            }

            if (u != -1) {
                for (int v : adj[u]) {
                    bool was_visited = false;
                    #pragma omp critical
                    {
                        if (!visited[v]) {
                            visited[v] = true;
                            next_level_nodes.push_back(v);
                        }
                    }
                }
            }
        }

        #pragma omp critical
        {
            for (int node : next_level_nodes) {
                q.push(node);
                bfs_order.push_back(node);
            }
        }
    }
}

int main() {
    int num_nodes, num_edges;

    cout << "Enter the number of nodes in the graph: ";
    cin >> num_nodes;

    cout << "Enter the number of edges in the graph: ";
    cin >> num_edges;

    if (num_nodes <= 0 || num_edges < 0) {
        cerr << "Invalid input for the number of nodes or edges." << endl;
        return 1;
    }

    vector<vector<int>> adj(num_nodes);
    cout << "Enter the edges (format: source destination):" << endl;
    for (int i = 0; i < num_edges; ++i) {
        int u, v;
        cin >> u >> v;
        if (u >= 0 && u < num_nodes && v >= 0 && v < num_nodes) {
            adj[u].push_back(v);
            adj[v].push_back(u); // Assuming an undirected graph
        } else {
            cerr << "Invalid edge input. Nodes should be between 0 and " << num_nodes - 1 << "." << endl;
            return 1;
        }
    }

    int start_node;
    cout << "Enter the starting node for BFS: ";
    cin >> start_node;

    if (start_node < 0 || start_node >= num_nodes) {
        cerr << "Invalid starting node. It should be between 0 and " << num_nodes - 1 << "." << endl;
        return 1;
    }

    vector<int> bfs_order;
    parallel_bfs(start_node, num_nodes, adj, bfs_order);

    cout << "Breadth-First Search traversal starting from node " << start_node << ": ";
    for (int node : bfs_order) {
        cout << node << " ";
    }
    cout << endl;

    return 0;
}