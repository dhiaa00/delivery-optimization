from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Step 1: Data Model
def create_data_model():
    """Creates the data model for 
    the problem."""
    data = {
        "locations": ["Algiers", "Oran", "Constantine", "Tamanrasset", "Béchar", "Tlemcen"],
        "distance_matrix": [
            [0, 4, 6, 24, 20, 7],  # Algiers
            [4, 0, 5, 22, 18, 2],  # Oran
            [6, 5, 0, 23, 19, 4],  # Constantine
                
            [24, 22, 23, 0, 9, 20],  # Tamanrasset
            [20, 18, 19, 9, 0, 17],  # Béchar
            [7, 2, 4, 20, 17, 0],  # Tlemcen
        ],
        "num_vehicles": 3,
        "starts": [0, 1, 2],
        "vehicle_capacities": [100, 500, 2000],  # Example capacities
    }
    return data

# Step 2: Greedy Human Heuristic
def greedy_human_heuristic(data):
    """Simulates a human choice using a greedy nearest-city approach."""
    distance_matrix = np.array(data["distance_matrix"])
    num_vehicles = data["num_vehicles"]
    starts = data["starts"]

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
                                           data["starts"])  # Assuming round trips
    routing = pywrapcp.RoutingModel(manager)

    # Define cost function
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data["distance_matrix"][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Solve the problem
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC

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
    route_names = [data["locations"][i] for i in route]
    print(f"Vehicle {vehicle + 1}: {route_names}")
print(f"Total Distance (Human Heuristic): {total_human_distance} km")

print("\nOptimized Routes:")
for vehicle, route in optimized_routes.items():
    route_names = [data["locations"][i] for i in route]
    print(f"Vehicle {vehicle + 1}: {route_names}")
print(f"Total Distance (Optimized): {total_optimized_distance} km")

print(f"\nOptimization Percentage: {optimization_percentage:.2f}%")

# Step 6: Visualization
def visualize_routes(data, routes, title):
    """Visualize routes."""
    city_positions = {
        "Algiers": (36.7, 3.2),
        "Oran": (35.7, -0.6),
        "Constantine": (36.3, 6.6),
        "Tamanrasset": (22.8, 5.5),
        "Béchar": (31.6, -2.2),
        "Tlemcen": (34.9, -1.3),
    }
    pos = {city: (lon, lat) for city, (lat, lon) in city_positions.items()}
    G = nx.DiGraph()

    for vehicle_id, route in routes.items():
        for i in range(len(route) - 1):
            from_city = data["locations"][route[i]]
            to_city = data["locations"][route[i + 1]]
            G.add_edge(from_city, to_city, vehicle=vehicle_id)

    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_color="skyblue", node_size=800, edge_color="gray")
    plt.title(title)
    plt.show()
    plt.savefig("vehicle_routes.png")
    print("Plot saved as vehicle_routes.png")


print("\nVisualizing Routes...")
visualize_routes(data, human_routes, "Human Heuristic Routes")
visualize_routes(data, optimized_routes, "Optimized Routes")
