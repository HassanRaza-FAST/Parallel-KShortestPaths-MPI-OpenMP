#include <iostream>
#include <vector>
#include <queue>
#include <algorithm>
#include <fstream>
#include <string>
#include <set>
#include <sstream>
#include <mpi.h>
#include <unistd.h>
#include<unordered_map>
#include <iterator> 
#include<cstring>
#include <chrono>
#include <omp.h>
using std::advance;

#define INF 99999

// this version is serial and using file
using namespace std;
void findKShortestPaths(vector<vector<int>>& edges, int totalNodes, int totalEdges, int k, int startNode, int endNode, int numProcesses, int processId, MPI_Datatype MPI_PAIR_INT) {
    
    // Create adjacency list
    vector<vector<pair<int, int>>> graph(totalNodes);
    vector<vector<int>> minPaths(totalNodes, vector<int>(k, INF)); // Initialize distances to infinity

    // Create the graph in the master process
    if (processId == 0) {
        for (int i = 0; i < totalEdges; i++) {
            int source = edges[i][0];
            int destination = edges[i][1];
            int cost = edges[i][2];
            graph[source].push_back({destination, cost});
        }
    }
    // Broadcast the graph to all processes
    for (int i = 0; i < totalNodes; i++) {
        int numEdges = processId == 0 ? graph[i].size() : 0;
        MPI_Bcast(&numEdges, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (processId != 0) {
            graph[i].resize(numEdges);
        }
        MPI_Bcast(graph[i].data(), numEdges, MPI_PAIR_INT, 0, MPI_COMM_WORLD);
    }

    
    
    int verticesPerProcess = (totalNodes + numProcesses - 1) / numProcesses;
    int startVertex = processId * verticesPerProcess;
    int endVertex = (processId + 1) * verticesPerProcess;
    if (endVertex > totalNodes) endVertex = totalNodes;
    // cout << "Process " << processId << ": Assigned vertices " << startVertex << " to " << endVertex << endl;
    MPI_Barrier(MPI_COMM_WORLD);    

    // Vector to store distances
    vector<vector<int>> shortestPaths(totalNodes, vector<int>(k, INF)); // Initialize distances to infinity

    // Initialization of priority queue
    priority_queue<pair<int, int>, vector<pair<int, int>>,
            greater<pair<int, int>>>
            priorityQueue;
    priorityQueue.push({0, startNode}); // Start from node startNode
    shortestPaths[startNode][0] = 0; // Distance from startNode to itself is 0

    // while priorityQueue has elements
    while (!priorityQueue.empty()) {
        int num_elements = priorityQueue.size();
        for(int i = 0; i< num_elements; i++){
            // Storing the node value
            int currentNode = priorityQueue.top().second;
            // Storing the distance value
            int currentDistance = priorityQueue.top().first;
            priorityQueue.pop();
            vector<pair<int, int>>& adjacentNodes = graph[currentNode]; // Use reference to avoid copying
            // Traversing the adjacency list
            for (const auto& edge : adjacentNodes) {
                if (edge.first >= startVertex && edge.first < endVertex) {
                    int destinationNode = edge.first;
                    int edgeCost = edge.second;
                    if(destinationNode != currentNode){
                        // Checking for the cost
                        if (currentDistance + edgeCost < shortestPaths[destinationNode][k - 1]) {
                            shortestPaths[destinationNode][k - 1] = currentDistance + edgeCost;
                            // Sorting the distances
                            sort(shortestPaths[destinationNode].begin(), shortestPaths[destinationNode].end());
                            // Pushing elements to priority queue
                            //cout << "Process " << processId << " Pushing " << destinationNode << " with distance " << (currentDistance + edgeCost) << " from node: " << currentNode << endl;
                            priorityQueue.push({(currentDistance + edgeCost), destinationNode});
                        }
                    }
                }
            }
        }

        // Gather the size of priority queues from all processes
        num_elements = priorityQueue.size();

        vector<int> pq_sizes(numProcesses);
        MPI_Allgather(&num_elements, 1, MPI_INT, pq_sizes.data(), 1, MPI_INT, MPI_COMM_WORLD);
        
         // Determine the total number of elements in all priority queues
        int total_elements = 0;
        for (int i = 0; i < numProcesses; ++i) {
            total_elements += pq_sizes[i];
        }

        // Resize vector to store all elements
        vector<pair<int,int>> all_elements(total_elements);


        // Gather all elements from all processes
        vector<int> displs(numProcesses, 0);
        
        for (int i = 1; i < numProcesses; ++i) {
            displs[i] = displs[i - 1] + pq_sizes[i - 1];
        }

        vector<pair<int, int>> pq_vector;
        while (!priorityQueue.empty()) {
            pq_vector.push_back(priorityQueue.top());
            priorityQueue.pop();
        }


        MPI_Allgatherv(pq_vector.data(), num_elements, MPI_PAIR_INT, all_elements.data(), pq_sizes.data(), displs.data(), MPI_PAIR_INT, MPI_COMM_WORLD);
        
        // // Sort all_elements before adding them to the priority queue
        sort(all_elements.begin(), all_elements.end());

        // put all elements back into the original priority queue
        priorityQueue = priority_queue<pair<int, int>, vector<pair<int, int>>,
        greater<pair<int, int>>>(all_elements.begin(), all_elements.end());
        
    }

    // Gather the 2D array from all processes and find the minimum values
    for (int i = 0; i < totalNodes; ++i) {
        MPI_Allreduce(&shortestPaths[i][0], &minPaths[i][0], k, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    }

    // Print the k shortest paths to endNode
    if(processId==0){
        cout << "K shortest paths to endNode: ";
        for (int i = 0; i < k; i++) {
            cout << minPaths[endNode][i] << " ";
        }
        cout << endl;
    }
}

void findKShortestPaths_Matrix(vector<vector<int>>& edges, int totalNodes, int totalEdges, int k, int startNode, int endNode, int numProcesses, int processId, MPI_Datatype MPI_PAIR_INT) {

    int verticesPerProcess = (totalNodes + numProcesses - 1) / numProcesses;
    int startVertex = processId * verticesPerProcess;
    int endVertex = (processId + 1) * verticesPerProcess;
    if (endVertex > totalNodes) endVertex = totalNodes;
    // cout << "Process " << processId << ": Assigned vertices " << startVertex << " to " << endVertex << endl;

    int noOfcolumns = endVertex - startVertex;


    // make vector with rows = totalnodes and columns noOfcolumns
    vector<vector<int>> graph(totalNodes, vector<int>(noOfcolumns, INF));

    

    // Populate adjacency matrix
    for (int i = 0; i < totalEdges; i++) {
        int destination = edges[i][1];
        if(destination<endVertex&&destination>=startVertex){
            int mapped_dest = destination - startVertex;
            int source = edges[i][0];
            int cost = edges[i][2];
            graph[source][mapped_dest] = cost;
        }
    }

    // Set the cost of traveling from a node to itself to 0 for the partioned graph
    for (int i = startVertex; i < endVertex; i++) {
        graph[i][i-startVertex] = 0;
    }

    
    // Vector to store distances
    vector<vector<int>> shortestPaths(totalNodes, vector<int>(k, INF)); // Initialize distances to infinity

    // Initialization of priority queuee
    priority_queue<pair<int, int>, vector<pair<int, int>>,
            greater<pair<int, int>>>
            priorityQueue;
    priorityQueue.push({0, startNode}); // Start from node startNode
    shortestPaths[startNode][0] = 0; // Distance from startNode to itself is 0

    // while priorityQueue has elements
    while (!priorityQueue.empty()) {
        int num_elements = priorityQueue.size();
        for(int i = 0; i< num_elements; i++){
           // cout << "Process " << processId << " has size: " << priorityQueue.size() << endl; 
            // Storing the node value
            int currentNode = priorityQueue.top().second;
            // Storing the distance value
            int currentDistance = priorityQueue.top().first;
            priorityQueue.pop();
            vector<int>& adjacentNodes = graph[currentNode]; // Use reference to avoid copying
            // Traversing the adjacency matrix column wise
            for (int i = startVertex; i < endVertex; i++) {
                int destinationNode = i;
                if(destinationNode != currentNode){
                    int edgeCost = adjacentNodes[i-startVertex]; // Adjust index to match the partitioned graph
                    // Checking for the cost
                    if (currentDistance + edgeCost < shortestPaths[destinationNode][k - 1]) {
                        shortestPaths[destinationNode][k - 1] = currentDistance + edgeCost;
                        // Sorting the distances
                        sort(shortestPaths[destinationNode].begin(), shortestPaths[destinationNode].end());
                        // Pushing elements to priority queue
                        //cout << "Pushing " << destinationNode << " with distance " << (currentDistance + edgeCost) << endl;
                        priorityQueue.push({(currentDistance + edgeCost), destinationNode});
                    }
                }
            }
        }
        // Gather the size of priority queues from all processes
        num_elements = priorityQueue.size();

        vector<int> pq_sizes(numProcesses);
        MPI_Allgather(&num_elements, 1, MPI_INT, pq_sizes.data(), 1, MPI_INT, MPI_COMM_WORLD);
        
         // Determine the total number of elements in all priority queues
        int total_elements = 0;
        for (int i = 0; i < numProcesses; ++i) {
            total_elements += pq_sizes[i];
        }

        // Resize vector to store all elements
        vector<pair<int,int>> all_elements(total_elements);


         // Gather all elements from all processes
        vector<int> displs(numProcesses, 0);
        for (int i = 1; i < numProcesses; ++i) {
            displs[i] = displs[i - 1] + pq_sizes[i - 1];
        }

        vector<pair<int, int>> pq_vector;
        while (!priorityQueue.empty()) {
            pq_vector.push_back(priorityQueue.top());
            priorityQueue.pop();
        }

        MPI_Allgatherv(pq_vector.data(), num_elements, MPI_PAIR_INT, all_elements.data(), pq_sizes.data(), displs.data(), MPI_PAIR_INT, MPI_COMM_WORLD);
        
        // Sort all_elements before adding them to the priority queue
        sort(all_elements.begin(), all_elements.end());


        // put all elements back into the original priority queue
        priorityQueue = priority_queue<pair<int, int>, vector<pair<int, int>>,
        greater<pair<int, int>>>(all_elements.begin(), all_elements.end());
    }

    // Gather the 2D array from all processes and find the minimum values
    vector<vector<int>> minPaths(totalNodes, vector<int>(k, INF));
    for (int i = 0; i < totalNodes; ++i) {
        MPI_Allreduce(&shortestPaths[i][0], &minPaths[i][0], k, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    }
    shortestPaths = minPaths;

    // Print the k shortest paths to endNode
    if(processId==0){
        cout << "K shortest paths to endNode: ";
    for (int i = 0; i < k; i++) {
        cout << shortestPaths[endNode][i] << " ";
    }
    cout << endl;
    }
    
}

void findKShortestPaths_Serial(vector<vector<int>>& edges, int totalNodes, int totalEdges, int k, int startNode, int endNode) {
    // Create adjacency list
    vector<vector<pair<int, int>>> graph(totalNodes);

    // Populate adjacency list
    for (int i = 0; i < totalEdges; i++) {
        int source = edges[i][0];
        int destination = edges[i][1];
        int cost = edges[i][2];
        graph[source].push_back({destination, cost});
    }

    // Vector to store distances
    vector<vector<int>> shortestPaths(totalNodes, vector<int>(k, INF)); // Initialize distances to infinity

    // Initialization of priority queue
    priority_queue<pair<int, int>, vector<pair<int, int>>,
            greater<pair<int, int>>>
            priorityQueue;
    priorityQueue.push({0, startNode}); // Start from node startNode
    shortestPaths[startNode][0] = 0; // Distance from startNode to itself is 0

    // while priorityQueue has elements
    while (!priorityQueue.empty()) {
        // Storing the node value
        int currentNode = priorityQueue.top().second;
        // Storing the distance value
        int currentDistance = priorityQueue.top().first;
        priorityQueue.pop();
        vector<pair<int, int>>& adjacentNodes = graph[currentNode]; // Use reference to avoid copying
        // Traversing the adjacency list
        for (auto& node : adjacentNodes) {
            int destinationNode = node.first;
            int edgeCost = node.second;
            // Checking for the cost
            if (currentDistance + edgeCost < shortestPaths[destinationNode][k - 1]) {
                shortestPaths[destinationNode][k - 1] = currentDistance + edgeCost;
                // Sorting the distances
                sort(shortestPaths[destinationNode].begin(), shortestPaths[destinationNode].end());
                // Pushing elements to priority queue
                //cout << "Pushing " << destinationNode << " with distance " << (currentDistance + edgeCost) << endl;
                priorityQueue.push({(currentDistance + edgeCost), destinationNode});
            }
        }
    }

    // Print the k shortest paths to endNode
    cout << "K shortest paths to endNode: ";
    for (int i = 0; i < k; i++) {
        cout << shortestPaths[endNode][i] << " ";
    }
    cout << endl;
    
}

void readCSV(string filename, int& totalNodes, int& totalEdges, vector<vector<int>>& edges, unordered_map<string, int>& nodeMap){
    // Read edges from CSV file
    ifstream inputFile(filename);
    if (!inputFile.is_open()) {
        cout << "Failed to open CSV file\n";
        return;
    }

    string line;
    // skip the first line
    getline(inputFile, line);
    while (getline(inputFile, line)) {
        stringstream ss(line);
        string source, target, weightStr;
        int weight;
        // Read the source, target and weight of the edge separated by comma
        getline(ss, source, ',');
        getline(ss, target, ',');
        getline(ss, weightStr, ',');
        weight = stoi(weightStr);
        // If the source node is not in the map, add it
        if (nodeMap.find(source) == nodeMap.end()) {
            nodeMap[source] = totalNodes++;
        }

        // If the target node is not in the map, add it
        if (nodeMap.find(target) == nodeMap.end()) {
            nodeMap[target] = totalNodes++;
        }

        int sourceIndex = nodeMap[source];
        int targetIndex = nodeMap[target];
        edges.push_back({sourceIndex, targetIndex, weight});
        // Add the reverse edge for undirected graph
        edges.push_back({targetIndex, sourceIndex, weight});
        totalEdges += 2; // since we have undirected graph so 2 edges are added
    }
    inputFile.close();
}

void readTxt(string filename, int& totalNodes, int& totalEdges, vector<vector<int>>& edges) {
    ifstream inFile(filename, ifstream::ate | ifstream::binary);
    size_t fileSize = inFile.tellg();
    inFile.close();

    // Read the third line serially
    inFile.open(filename);
    string line;
    for (int i = 0; i < 2; i++) {
        getline(inFile, line);
    }
    getline(inFile, line);
    istringstream iss(line);
    string word;
    iss >> word; // Skip "#"
    for (int i = 0; i < 2; i++) {
        iss >> word; // Skip "Nodes:" and "Edges:"
        if (i == 0) {
            iss >> totalNodes;
        } else {
            iss >> totalEdges;
        }
    }
    size_t startPos = inFile.tellg();
    inFile.close();

    int numThreads = omp_get_max_threads();
    size_t chunkSize = (fileSize - startPos) / numThreads;

    #pragma omp parallel
    {
        int threadNum = omp_get_thread_num();
        size_t start = startPos + threadNum * chunkSize;
        size_t end = (threadNum == numThreads - 1) ? fileSize : start + chunkSize;

        ifstream threadFile(filename);
        threadFile.seekg(start);

        string line;
        while (threadFile.tellg() < end && getline(threadFile, line)) {
            istringstream iss(line);
            int source, destination, cost = 1;
            if (iss >> source >> destination) {
                #pragma omp critical
                edges.push_back({source, destination, cost});
            }
        }
    }
}




int main(int argc, char* argv[]) {
    omp_set_num_threads(16);
    MPI_Init(&argc, &argv);

    // Define MPI_Datatype for pair<int, int>
    MPI_Datatype MPI_PAIR_INT;
    MPI_Type_contiguous(2, MPI_INT, &MPI_PAIR_INT);
    MPI_Type_commit(&MPI_PAIR_INT);

    int numProcesses, processId;
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);
    MPI_Comm_rank(MPI_COMM_WORLD, &processId);

    int totalNodes, totalEdges, K = 3;
    int startNode, endNode; 
    string filename;

    vector<vector<int>> edges;
    unordered_map<string, int> nodeMap;

    int choice;
    if(processId==0){
        cout << "Enter 1 for txt dataset, 2 for csv dataset:  ";
        cin >> choice;

        while(choice!= 1 && choice !=2){
            cout << "Invalid choice. Retry ";
            cin >> choice;
        }
    }

    // Broadcast the choice to all processes
    MPI_Bcast(&choice, 1, MPI_INT, 0, MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();

    if(choice==1){
        if(processId==0){
            filename = "Email-Enron.txt"; 
            readTxt(filename, totalNodes, totalEdges, edges);
        }

        // Broadcast the edges, totalNodes, and totalEdges to all processes
        MPI_Bcast(&totalNodes, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&totalEdges, 1, MPI_INT, 0, MPI_COMM_WORLD);
        for (int i = 0; i < totalEdges; i++) {
            edges.push_back(vector<int>(3));
            MPI_Bcast(edges[i].data(), 3, MPI_INT, 0, MPI_COMM_WORLD);
        }   

        srand(time(0));
        if(processId==0){
            cout << "Number of processes: " << numProcesses << endl;
            cout << "Total Nodes: " << totalNodes << ", Total Edges: " << totalEdges << "\n\n";
        }
        for(int i = 0; i<2;i++){
            startNode = rand() % totalNodes;
            endNode = rand() % totalNodes;
            if(processId==0){
                cout << "StartingNode = " << startNode << endl;
                cout << "EndingNode = " << endNode << endl << endl;
                cout << "Calling findKShortestPaths Serial version...\n";
                start = chrono::high_resolution_clock::now();  // Start timing here
                findKShortestPaths_Serial(edges, totalNodes, totalEdges, K, startNode, endNode);
                end = chrono::high_resolution_clock::now();  // End timing here
                chrono::duration<double> elapsed = end - start;
                cout << "Time elapsed for serial version: " << elapsed.count() << " seconds\n\n";
            }
            if(processId==0){
                cout << "Calling findKShortestPaths Parallel version...\n";
                start = chrono::high_resolution_clock::now();  // Start timing here
            }
            findKShortestPaths(edges, totalNodes, totalEdges, K, startNode, endNode, numProcesses, processId, MPI_PAIR_INT);
            if(processId==0){
                end = chrono::high_resolution_clock::now();  // End timing here
                chrono::duration<double> elapsed = end - start;
                cout << "Time elapsed for parallel version: " << elapsed.count() << " seconds\n\n";
            }
        }
    }
    else{
        if(processId==0){
            filename = "doctorwho.csv";
            totalNodes=0;
            totalEdges=0;
            readCSV(filename, totalNodes, totalEdges, edges, nodeMap);
        }

        // Broadcast the edges, totalNodes, and totalEdges to all processes
        MPI_Bcast(&totalNodes, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&totalEdges, 1, MPI_INT, 0, MPI_COMM_WORLD);
        for (int i = 0; i < totalEdges; i++) {
            edges.push_back(vector<int>(3));
            MPI_Bcast(edges[i].data(), 3, MPI_INT, 0, MPI_COMM_WORLD);
        }

        srand(time(0));
        if(processId==0){
            cout << "Number of processes: " << numProcesses << endl;
            cout << "Total Nodes: " << totalNodes << ", Total Edges: " << totalEdges << "\n\n";
        }

        for(int i = 0; i<2;i++){
            string startNode_str;
            string endNode_str;

            if(processId==0){
                // Get a random iterator position in the map
                int randomPosition = rand() % nodeMap.size();

                // Create an iterator pointing to the beginning of the map
                unordered_map<string, int>::iterator it = nodeMap.begin();

                // Advance the iterator to the random position
                advance(it, randomPosition);

                // The key of the map entry pointed to by the iterator is your random node
                startNode_str = it->first;

                // Repeat the process to get a random endNode
                randomPosition = rand() % nodeMap.size();
                it = nodeMap.begin();
                advance(it, randomPosition);
                endNode_str = it->first;

                startNode = nodeMap[startNode_str];
                endNode = nodeMap[endNode_str];
            }

            //broadcast the startNode and endNode
            MPI_Bcast(&startNode, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(&endNode, 1, MPI_INT, 0, MPI_COMM_WORLD);
            

            if(processId==0){
                cout << "StartingNode = " << startNode_str << endl;
                cout << "EndingNode = " << endNode_str << endl;
                cout << "Calling findKShortestPaths Serial version...\n";
                start = chrono::high_resolution_clock::now();  // Start timing here
                findKShortestPaths_Serial(edges, totalNodes, totalEdges, K, startNode, endNode);
                end = chrono::high_resolution_clock::now();  // End timing here
                chrono::duration<double> elapsed = end - start;
                cout << "Time elapsed for serial version: " << elapsed.count() << " seconds\n\n";
            }
            if(processId==0){
                cout << "Calling findKShortestPaths Parallel version...\n";
                start = chrono::high_resolution_clock::now();  // Start timing here
            }
            findKShortestPaths_Matrix(edges, totalNodes, totalEdges, K, startNode, endNode, numProcesses, processId, MPI_PAIR_INT);
            if(processId==0){
                end = chrono::high_resolution_clock::now();  // End timing here
                chrono::duration<double> elapsed = end - start;
                cout << "Time elapsed for parallel version: " << elapsed.count() << " seconds\n\n";
            }
        }
   
    }
        
    // Free the custom MPI datatype
    MPI_Type_free(&MPI_PAIR_INT);

    MPI_Finalize();
    return 0;
}