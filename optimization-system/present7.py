import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

# Set the backend to TkAgg
matplotlib.use('TkAgg')

# Step 1: Data Model
def create_data_model():
    """Creates the data model for the problem."""
    data = {
        'locations': {  # Simulated latitude/longitude coordinates for visualization
            "algiers": (36.737232, 3.086472),  # Algiers
            "oran": (35.691109, -0.641074),  # Oran
            "constantine": (36.365, 6.61472),  # Constantine
            "annaba": (36.9, 7.76667),  # Annaba
            "batna": (35.55, 6.16667),  # Batna
            "setif": (36.19112, 5.41373),  # Setif
            "bejaia": (36.75, 5.08333),  # Bejaia
            "tlemcen": (34.87833, -1.315),  # Tlemcen
            "sidi_bel_abbes": (35.18994, -0.63085),  # Sidi Bel Abbes
            "blida": (36.47004, 2.8277),  # Blida
            "boumerdes": (36.76639, 3.47717),  # Boumerdes
            "ghardaia": (32.49094, 3.67347),  # Ghardaia
            "laghouat": (33.8, 2.88333),  # Laghouat
            "ouargla": (31.95, 5.33333),  # Ouargla
            "bechar": (31.61667, -2.21667),  # Bechar
            "tamanrasset": (22.785789, 5.522926),  # Tamanrasset
        },
        "neighbors": {
            "algiers": ["blida", "boumerdes"],
            "oran": ["tlemcen", "sidi_bel_abbes"],
            "constantine": ["setif", "batna"],
            "annaba": ["constantine"],
            "batna": ["setif", "constantine"],
            "setif": ["bejaia", "batna", "constantine"],
            "bejaia": ["setif"],
            "tlemcen": ["oran", "sidi_bel_abbes"],
            "sidi_bel_abbes": ["oran", "tlemcen"],
            "blida": ["algiers", "boumerdes"],
            "boumerdes": ["algiers", "blida"],
            "ghardaia": ["laghouat", "ouargla"],
            "laghouat": ["ghardaia"],
            "ouargla": ["ghardaia", "bechar"],
            "bechar": ["ouargla", "tamanrasset"],
            "tamanrasset": ["bechar"]
        },
        "num_vehicles": 5,
        "starts": [0, 1, 2, 3, 4],  # Starting points for the vehicles
        "ends": [15, 14, 13, 12, 11],  # Different end points for the vehicles
        "vehicle_capacities": [100, 500, 2000, 1500, 1200],  # Example capacities
    }

    # Create a graph from the neighbors data
    G = nx.Graph()
    for city, neighbors in data["neighbors"].items():
        for neighbor in neighbors:
            G.add_edge(city, neighbor, weight=np.linalg.norm(np.array(data["locations"][city]) - np.array(data["locations"][neighbor])))

    # Calculate the distance matrix using the shortest path algorithm
    cities = list(data["locations"].keys())
    distance_matrix = np.zeros((len(cities), len(cities)))
    for i, city1 in enumerate(cities):
        for j, city2 in enumerate(cities):
            if i != j:
                try:
                    distance_matrix[i][j] = nx.shortest_path_length(G, source=city1, target=city2, weight='weight')
                except nx.NetworkXNoPath:
                    distance_matrix[i][j] = float('inf')  # No path exists
            else:
                distance_matrix[i][j] = 0

    data["distance_matrix"] = distance_matrix.tolist()
    data["starts"] = [cities.index(city) for city in ["algiers", "oran", "constantine", "annaba", "batna"]]
    data["ends"] = [cities.index(city) for city in ["tamanrasset", "bechar", "ouargla", "laghouat", "ghardaia"]]

    return data

# Step 2: Greedy Human Heuristic
def greedy_human_heuristic(data):
    """Simulates a human choice using a greedy nearest-city approach."""
    distance_matrix = np.array(data["distance_matrix"])
    num_vehicles = data["num_vehicles"]
    starts = data["starts"]
    ends = data["ends"]

    # Initialize visited nodes and routes
    visited = set()
    human_routes = {}

    for vehicle_id in range(num_vehicles):
        route = []
        current_city = starts[vehicle_id]
        route.append(current_city)
        visited.add(current_city)

        while len(visited) < len(distance_matrix):
            unvisited = [i for i in range(len(distance_matrix)) if i not in visited]
            if not unvisited:
                break
            # Pick the nearest unvisited city
            nearest_city = min(unvisited, key=lambda city: distance_matrix[current_city][city])
            route.append(nearest_city)
            visited.add(nearest_city)
            current_city = nearest_city

        # Ensure route ends at the predefined end point
        if current_city != ends[vehicle_id]:
            route.append(ends[vehicle_id])

        human_routes[vehicle_id] = route

    # Calculate total distance for the human heuristic
    total_human_distance = 0
    for route in human_routes.values():
        for i in range(len(route) - 1):
            total_human_distance += distance_matrix[route[i]][route[i + 1]]

    return human_routes, total_human_distance

# Step 3: OR-Tools Optimized Solution
def optimize_routes(data):
    """Solve the VRP using OR-Tools."""
    manager = pywrapcp.RoutingIndexManager(len(data["distance_matrix"]),
                                           data["num_vehicles"],
                                           data["starts"],
                                           data["ends"])
    routing = pywrapcp.RoutingModel(manager)

    # Define cost function
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data["distance_matrix"][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add Capacity constraint.
    def demand_callback(from_index):
        return 1  # Example demand, modify as needed

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        data['vehicle_capacities'],  # vehicle maximum capacities
        True,  # start cumul to zero
        'Capacity')

    # Solve the problem
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.time_limit.seconds = 30

    solution = routing.SolveWithParameters(search_parameters)
    if not solution:
        print("No solution found!")
        return None

    # Extract routes
    routes = {}
    for vehicle_id in range(data["num_vehicles"]):
        index = routing.Start(vehicle_id)
        route = []
        while not routing.IsEnd(index):
            route.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))
        route.append(manager.IndexToNode(index))  # Add the end node
        routes[vehicle_id] = route

    # Calculate total distance for the optimized solution
    total_optimized_distance = 0
    distance_matrix = data["distance_matrix"]
    for route in routes.values():
        for i in range(len(route) - 1):
            total_optimized_distance += distance_matrix[route[i]][route[i + 1]]

    return routes, total_optimized_distance

# Step 4: Compare Results
data = create_data_model()

# Greedy Human Solution
human_routes, total_human_distance = greedy_human_heuristic(data)

# Optimized Solution
optimized_routes, total_optimized_distance = optimize_routes(data)

# Calculate Optimization Percentage
optimization_percentage = ((total_human_distance - total_optimized_distance) / total_human_distance) * 100

# Step 5: Print Results
print("Greedy Human Heuristic Routes:")
for vehicle, route in human_routes.items():
    route_names = [list(data["locations"].keys())[i] for i in route]
    print(f"Vehicle {vehicle + 1}: {route_names}")
print(f"Total Distance (Human Heuristic): {total_human_distance} km")

print("\nOptimized Routes:")
for vehicle, route in optimized_routes.items():
    route_names = [list(data["locations"].keys())[i] for i in route]
    print(f"Vehicle {vehicle + 1}: {route_names}")
print(f"Total Distance (Optimized): {total_optimized_distance} km")

print(f"\nOptimization Percentage: {optimization_percentage:.2f}%")

# Step 6: Visualization
def visualize_routes(data, routes, title, filename):
    """Visualize routes."""
    pos = data['locations']
    G = nx.DiGraph()

    colors = ['r', 'g', 'b', 'y', 'c', 'm']  # Colors for different vehicles
    edge_labels = {}

    for vehicle_id, route in routes.items():
        for i in range(len(route) - 1):
            from_city = list(data["locations"].keys())[route[i]]
            to_city = list(data["locations"].keys())[route[i + 1]]
            G.add_edge(from_city, to_city, color=colors[vehicle_id % len(colors)], vehicle=vehicle_id)
            edge_labels[(from_city, to_city)] = f'Vehicle {vehicle_id + 1}'

    # Ensure all nodes are added to the graph
    for city in data["locations"].keys():
        if city not in G:
            G.add_node(city)

    edges = G.edges()
    colors = [G[u][v]['color'] for u, v in edges]

    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_color="skyblue", node_size=800, edge_color=colors, width=2, edge_cmap=plt.cm.Blues)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    # Adjust plot limits to ensure all nodes and edges are visible
    padding = 5  # Increase padding for more zoom out
    plt.xlim(min([pos[city][0] for city in pos]) - padding, max([pos[city][0] for city in pos]) + padding)
    plt.ylim(min([pos[city][1] for city in pos]) - padding, max([pos[city][1] for city in pos]) + padding)

    plt.title(title)
    plt.savefig(filename)
    plt.show()
    print(f"Plot saved as {filename}")

print("\nVisualizing Routes...")
visualize_routes(data, human_routes, "Human Heuristic Routes", "human_routes.png")
visualize_routes(data, optimized_routes, "Optimized Routes", "optimized_routes.png")