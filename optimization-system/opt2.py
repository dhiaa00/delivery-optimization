# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from geopy.distance import geodesic
from matplotlib.patches import Patch

# Data Preparation
def create_data_model():
    """Define the data model for routing problem."""
    data = {
        'distance_matrix': [
            [0, 400, 800, 1200, 1600, 2000],  # Distances between points in km
            [400, 0, 600, 1000, 1400, 1800],
            [800, 600, 0, 400, 800, 1200],
            [1200, 1000, 400, 0, 600, 1000],
            [1600, 1400, 800, 600, 0, 400],
            [2000, 1800, 1200, 1000, 400, 0]
        ],
        'locations': {  # Simulated latitude/longitude coordinates for visualization
            0: (36.737232, 3.086472),  # Algiers
            1: (35.691109, -0.641074),  # Oran
            2: (32.876174, -6.904827),  # Beni Mellal
            3: (28.033886, 1.659626),  # Ouargla
            4: (24.291832, 9.452044),  # In Salah
            5: (22.785789, 5.522926),  # Tamanrasset
        },
        'num_vehicles': 2,
        'starts': [0, 5],  # Start nodes for the vehicles
    }
    return data

# Calculate Total Distance Function
def calculate_total_distance(route, distance_matrix):
    total_distance = 0
    for i in range(len(route) - 1):
        total_distance += distance_matrix[route[i]][route[i + 1]]
    return total_distance

# Human Heuristic Route
def human_greedy_routes(data):
    distance_matrix = np.array(data['distance_matrix'])
    visited = set()
    routes = {}
    total_distance = 0
    for vehicle_id, start in enumerate(data['starts']):
        current = start
        route = [current]
        visited.add(current)
        while len(visited) < len(distance_matrix):
            nearest_city = None
            min_distance = float('inf')
            for city in range(len(distance_matrix)):
                if city not in visited and distance_matrix[current][city] < min_distance:
                    nearest_city = city
                    min_distance = distance_matrix[current][city]
            if nearest_city is None:
                break
            route.append(nearest_city)
            visited.add(nearest_city)
            current = nearest_city
        # Return to the hub
        route.append(start)
        distance = calculate_total_distance(route, distance_matrix)
        total_distance += distance
        routes[vehicle_id] = route
    return routes, total_distance

# Visualization Function
def visualize_routes(data, routes, title):
    locations = data['locations']
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['blue', 'green', 'orange', 'red', 'purple', 'brown']
    for vehicle, route in routes.items():
        coords = [locations[city] for city in route]
        coords = np.array(coords)
        ax.plot(coords[:, 1], coords[:, 0], marker='o', label=f"Vehicle {vehicle + 1}", color=colors[vehicle % len(colors)])
        for i, city in enumerate(route):
            ax.text(coords[i, 1], coords[i, 0], f"{city}", fontsize=9, ha='center')
    # Hub Annotations
    ax.scatter([loc[1] for loc in locations.values()], [loc[0] for loc in locations.values()], c='black', label='Hubs', s=100, marker='x')
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend()
    plt.show()
    plt.savefig("human_routes.png", dpi=300)
    plt.close()  # Close the figure to avoid overlapping


# Statistics Visualization
def plot_statistics(human_distance, optimized_distance):
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    # Distance Comparison
    ax[0].bar(['Human (Greedy)', 'Optimized'], [human_distance, optimized_distance], color=['blue', 'green'])
    ax[0].set_title('Total Distance Comparison')
    ax[0].set_ylabel('Distance (km)')
    # Optimization Percentage
    optimization_percentage = (human_distance - optimized_distance) / human_distance * 100
    ax[1].pie([optimization_percentage, 100 - optimization_percentage], labels=['Optimized (%)', 'Remaining (%)'], autopct='%1.1f%%', colors=['green', 'lightgray'])
    ax[1].set_title('Optimization Percentage')
    plt.show()
    plt.savefig("routes.png", dpi=300)
    plt.close()  # Close the figure to avoid overlapping


# Main Execution
data = create_data_model()
human_routes, human_distance = human_greedy_routes(data)
optimized_routes = {0: [0, 2, 3, 0], 1: [5, 4, 1, 5]}  # Simulated optimized routes
optimized_distance = 2800  # Simulated optimized total distance

print("Human Routes:", human_routes)
print("Optimized Routes:", optimized_routes)

# Visualize Routes
visualize_routes(data, human_routes, "Human Heuristic Routes")
visualize_routes(data, optimized_routes, "Optimized Routes")

# Plot Statistics
plot_statistics(human_distance, optimized_distance)
