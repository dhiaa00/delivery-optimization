import numpy as np  # Fix numpy import
import matplotlib.pyplot as plt
import networkx as nx
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

def create_data_model():
    """Creates the data model for the problem."""
    data = {
        'locations': {
            "algiers": (36.737232, 3.086472),
            "oran": (35.691109, -0.641074),
            "constantine": (36.365, 6.61472),
            "annaba": (36.9, 7.76667),
            "batna": (35.55, 6.16667),
            "setif": (36.19112, 5.41373),
            "bejaia": (36.75, 5.08333),
            "tlemcen": (34.87833, -1.315),
            "sidi_bel_abbes": (35.18994, -0.63085),
            "blida": (36.47004, 2.8277),
            "boumerdes": (36.76639, 3.47717),
            "ghardaia": (32.49094, 3.67347),
            "laghouat": (33.8, 2.88333),
            "ouargla": (31.95, 5.33333),
            "bechar": (31.61667, -2.21667),
            "tamanrasset": (22.785789, 5.522926),
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
        "vehicle_capacities": [100, 500, 2000, 1500, 1200],
        "vehicle_types": ["commercial_truck", "truck", "van", "truck", "commercial_truck"],
        "package_demands": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160],
        "vehicle_speeds": [60, 50, 70, 50, 60],
        "package_priorities": [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1],
        "time_windows": [(0, 5), (1, 6), (2, 7), (3, 8), (4, 9), (5, 10), (6, 11), (7, 12), 
                        (8, 13), (9, 14), (10, 15), (11, 16), (12, 17), (13, 18), (14, 19), (15, 20)]
    }

    # Create a graph from the neighbors data
    G = nx.Graph()
    for city, neighbors in data["neighbors"].items():
        for neighbor in neighbors:
            G.add_edge(city, neighbor, weight=np.linalg.norm(
                np.array(data["locations"][city]) - np.array(data["locations"][neighbor])))

    # Calculate the distance matrix using shortest path
    cities = list(data["locations"].keys())
    distance_matrix = np.zeros((len(cities), len(cities)))
    
    for i, city1 in enumerate(cities):
        for j, city2 in enumerate(cities):
            if i != j:
                try:
                    distance_matrix[i][j] = nx.shortest_path_length(G, source=city1, target=city2, weight='weight')
                except nx.NetworkXNoPath:
                    distance_matrix[i][j] = float('inf')
            else:
                distance_matrix[i][j] = 0

    data["distance_matrix"] = distance_matrix.tolist()
    data["starts"] = [cities.index(city) for city in ["algiers", "oran", "constantine", "annaba", "batna"]]
    data["ends"] = [cities.index(city) for city in ["tamanrasset", "bechar", "ouargla", "laghouat", "ghardaia"]]

    return data

def greedy_human_heuristic(data):
    """Simulates a human choice using a greedy nearest-city approach."""
    distance_matrix = np.array(data["distance_matrix"])
    num_vehicles = data["num_vehicles"]
    starts = data["starts"]
    ends = data["ends"]

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
            nearest_city = min(unvisited, key=lambda city: distance_matrix[current_city][city])
            route.append(nearest_city)
            visited.add(nearest_city)
            current_city = nearest_city

        if current_city != ends[vehicle_id]:
            route.append(ends[vehicle_id])

        human_routes[vehicle_id] = route

    total_human_distance = 0
    for route in human_routes.values():
        for i in range(len(route) - 1):
            total_human_distance += distance_matrix[route[i]][route[i + 1]]

    return human_routes, total_human_distance

def optimize_routes(data):
    """Solve the VRP using OR-Tools."""
    manager = pywrapcp.RoutingIndexManager(
        len(data["distance_matrix"]),
        data["num_vehicles"],
        data["starts"],
        data["ends"]
    )
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(data["distance_matrix"][from_node][to_node])

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return data["package_demands"][from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        data['vehicle_capacities'],  # vehicle maximum capacities
        True,  # start cumul to zero
        'Capacity'
    )

    def time_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(data["distance_matrix"][from_node][to_node] / 
                  data["vehicle_speeds"][manager.VehicleIndex(from_index)])

    time_callback_index = routing.RegisterTransitCallback(time_callback)
    routing.AddDimension(
        time_callback_index,
        0,  # no slack
        1000,  # maximum time per vehicle
        True,  # start cumul to zero
        'Time'
    )

    time_dimension = routing.GetDimensionOrDie('Time')
    for location_idx, time_window in enumerate(data['time_windows']):
        if location_idx in data['starts']:
            continue
        index = manager.NodeToIndex(location_idx)
        time_dimension.CumulVar(index).SetRange(time_window[0], time_window[1])

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    search_parameters.time_limit.seconds = 30

    solution = routing.SolveWithParameters(search_parameters)
    if not solution:
        return None

    routes = {}
    for vehicle_id in range(data["num_vehicles"]):
        index = routing.Start(vehicle_id)
        route = []
        while not routing.IsEnd(index):
            route.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))
        route.append(manager.IndexToNode(index))
        routes[vehicle_id] = route

    total_optimized_distance = 0
    for route in routes.values():
        for i in range(len(route) - 1):
            total_optimized_distance += data["distance_matrix"][route[i]][route[i + 1]]

    return routes, total_optimized_distance

def visualize_routes(data, routes, title, filename):
    """Visualize routes."""
    try:
        # Clear any existing plots
        plt.clf()
        plt.close('all')
        
        # Create new figure
        fig, ax = plt.subplots(figsize=(15, 10))
        
        # Prepare coordinates
        cities = list(data['locations'].keys())
        lons = []
        lats = []
        for city in cities:
            lat, lon = data['locations'][city]
            lats.append(lat)
            lons.append(lon)
        
        # Plot cities with larger markers
        ax.scatter(lons, lats, c='lightblue', s=500, zorder=2)
        
        # Add city labels with offset
        for city, lon, lat in zip(cities, lons, lats):
            ax.annotate(city, (lon, lat), 
                       xytext=(5, 5), 
                       textcoords='offset points',
                       fontsize=8)
        
        # Draw routes
        colors = ['red', 'green', 'blue', 'yellow', 'cyan']
        
        for vehicle_id, route in routes.items():
            color = colors[vehicle_id % len(colors)]
            
            # Draw route segments
            for i in range(len(route) - 1):
                from_city = cities[route[i]]
                to_city = cities[route[i + 1]]
                
                x = [data['locations'][from_city][1], data['locations'][to_city][1]]
                y = [data['locations'][from_city][0], data['locations'][to_city][0]]
                
                # Draw line
                ax.plot(x, y, c=color, linewidth=1.5, zorder=1)
                
                # Add vehicle label at midpoint
                mid_x = sum(x) / 2
                mid_y = sum(y) / 2
                ax.annotate(f'V{vehicle_id + 1}', 
                           (mid_x, mid_y),
                           bbox=dict(facecolor='white', alpha=0.7),
                           fontsize=6,
                           ha='center')
        
        # Set title and grid
        plt.title(title)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Adjust margins and limits
        plt.margins(0.1)
        
        # Save figure
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        plt.close(fig)
        
        print(f"Successfully saved {filename}")
        
    except Exception as e:
        print(f"Error in visualization: {str(e)}")
        plt.close('all')
def main():
    # Create data model
    data = create_data_model()

    # Get solutions
    print("Calculating routes...")
    human_routes, total_human_distance = greedy_human_heuristic(data)
    optimized_routes, total_optimized_distance = optimize_routes(data)

    if optimized_routes is None:
        print("Failed to find optimized solution")
        return

    # Calculate optimization percentage
    optimization_percentage = ((total_human_distance - total_optimized_distance) / total_human_distance) * 100

    # Print results
    print("\nGreedy Human Heuristic Routes:")
    for vehicle, route in human_routes.items():
        route_names = [list(data["locations"].keys())[i] for i in route]
        print(f"Vehicle {vehicle + 1}: {route_names}")
    print(f"Total Distance (Human Heuristic): {total_human_distance:.2f} km")

    print("\nOptimized Routes:")
    for vehicle, route in optimized_routes.items():
        route_names = [list(data["locations"].keys())[i] for i in route]
        print(f"Vehicle {vehicle + 1}: {route_names}")
    print(f"Total Distance (Optimized): {total_optimized_distance:.2f} km")
    print(f"\nOptimization Percentage: {optimization_percentage:.2f}%")

    # Visualize routes
    print("\nVisualizing Routes...")
    try:
        visualize_routes(data, human_routes, "Human Heuristic Routes", "human_routes.png")
        visualize_routes(data, optimized_routes, "Optimized Routes", "optimized_routes.png")
    except Exception as e:
        print(f"Error during visualization: {e}")

if __name__ == "__main__":
    main()