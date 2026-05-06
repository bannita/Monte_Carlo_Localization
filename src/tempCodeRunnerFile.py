import cv2
import numpy as np
import math
import random as rd
from scipy.ndimage import binary_dilation

robot_radius = 8
num_particles = 200
move_noise = 0.3
turn_noise = 0.02

move_step = 5
turn_step = 0.1
sigma = 40.0
spread_threshold = 20
angle_threshold = 0.8
resample_keep_ratio = 0.75
position_jitter = 6
angle_jitter = 0.12

map_img = cv2.imread("map/map_processed.png", 0) #loading map in grayscale

if map_img is None:
    print("Error: map image was not loaded")
    exit()

grid_map = (map_img > 0).astype(int)

height = grid_map.shape[0]
width = grid_map.shape[1]
max_distance = math.sqrt(width * width + height * height)

kernel_size = robot_radius * 2 + 1
y_coords, x_coords = np.ogrid[-robot_radius:robot_radius+1, -robot_radius:robot_radius+1]
circle_kernel = (x_coords**2 + y_coords**2 <= robot_radius**2)

wall_mask = (grid_map == 1)
inflated_walls = binary_dilation(wall_mask, structure=circle_kernel)
valid_map = ~inflated_walls  # True = safe to place robot/particle

#generate random particles
def generate_random_particles(num_particles, grid_map, robot_radius):
    particles = []

    while len(particles) < num_particles:
        x = rd.randint(0, width -1)
        y = rd.randint(0, height - 1)
        theta = normalize_angle(rd.uniform(0, 2 * math.pi))

        if is_valid_position(x, y, grid_map, robot_radius):
            particle = {
                "x": x,
                "y": y,
                "theta": theta,
                "weight": 1.0
            }
            particles.append(particle)

    return particles

def normalize_angle(theta):
    while theta > math.pi:
        theta -= 2 * math.pi

    while theta < -math.pi:
        theta += 2 * math.pi

    return theta

def move_particles(particles, move_amount, turn_amount, grid_map, robot_radius):
    for particle in particles:
        noisy_move = move_amount + rd.uniform(-move_noise,move_noise)
        noisy_turn = turn_amount + rd.uniform(-turn_noise, turn_noise)

        new_theta = normalize_angle(particle["theta"] + noisy_turn)
        new_x = particle["x"] + int(noisy_move*math.cos(new_theta))
        new_y = particle["y"] + int(noisy_move*math.sin(new_theta))

        if is_valid_position(new_x,new_y, grid_map, robot_radius):
            particle["x"] = new_x
            particle["y"] = new_y
            particle["theta"] = new_theta


#giving weights to particles
def calculate_particle_weight(particle, robot_distances, sensor_offsets, grid_map, max_sensor_distance):
    particle_distances = get_all_sensor_distances(
        particle["x"],
        particle["y"],
        particle["theta"],
        sensor_offsets,
        grid_map,
        max_sensor_distance
    )

    squared_error_sum = 0.0

    for i in range(len(robot_distances)):
        error = robot_distances[i] - particle_distances[i]
        squared_error_sum += error * error

    mean_squared_error = squared_error_sum / len(robot_distances)

    weight = math.exp(-mean_squared_error / (2 * sigma * sigma))

    return weight

#resampling particles
def resample_particles(particles):
    weights = []

    for particle in particles:
        weights.append(particle["weight"])

    total_weight = sum(weights)

    if total_weight == 0:
        return generate_random_particles(num_particles, grid_map, robot_radius)

    normalized_weights = []

    for weight in weights:
        normalized_weights.append(weight / total_weight)

    keep_count = int(num_particles * resample_keep_ratio)
    random_count = num_particles - keep_count

    cumulative_weights = []
    accumulated_weight = 0.0

    for weight in normalized_weights:
        accumulated_weight += weight
        cumulative_weights.append(accumulated_weight)

    step = 1.0 / keep_count
    start = rd.uniform(0, step)

    chosen_particles = []
    index = 0

    for i in range(keep_count):
        pointer = start + i * step

        while index < len(cumulative_weights) - 1 and cumulative_weights[index] < pointer:
            index += 1

        chosen_particles.append(particles[index])

    new_particles = []

    for particle in chosen_particles:
        new_x = particle["x"] + rd.randint(-position_jitter, position_jitter)
        new_y = particle["y"] + rd.randint(-position_jitter, position_jitter)
        new_theta = normalize_angle(particle["theta"] + rd.uniform(-angle_jitter, angle_jitter))

        if is_valid_position(new_x, new_y, grid_map, robot_radius):
            new_particle = {
                "x": new_x,
                "y": new_y,
                "theta": new_theta,
                "weight": 1.0
            }
        else:
            new_particle = {
                "x": particle["x"],
                "y": particle["y"],
                "theta": particle["theta"],
                "weight": 1.0
            }

        new_particles.append(new_particle)

    random_particles = generate_random_particles(random_count, grid_map, robot_radius)

    for particle in random_particles:
        new_particles.append(particle)

    return new_particles


#function for checking whether the robot is in valid position
def is_valid_position(x, y, grid_map, robot_radius):
    if x - robot_radius < 0 or x + robot_radius >= width:
        return False
    if y - robot_radius < 0 or y + robot_radius >= height:
        return False
    return bool(valid_map[y, x])

#generate random position for robot
def generate_random_robot(grid_map, robot_radius):
    valid_coordinates = False

    while valid_coordinates != True:
        x_coordinate = rd.randint(0, width - 1)
        y_coordinate = rd.randint(0, height - 1)
        degree = normalize_angle(rd.uniform(0, 2 * math.pi))

        if is_valid_position(x_coordinate, y_coordinate, grid_map, robot_radius):
            valid_coordinates = True

    robot = {
        "x": x_coordinate,
        "y": y_coordinate,
        "theta": degree
    }

    return robot

robot = generate_random_robot(grid_map, robot_radius)
print(robot)

particles = generate_random_particles(num_particles, grid_map, robot_radius)

#sensor function(measures distance)
def get_sensor_distance(x, y, theta, grid_map, max_distance):
    steps = np.arange(0, max_distance)

    xs = (x + steps * math.cos(theta)).astype(int)
    ys = (y + steps * math.sin(theta)).astype(int)

    valid = (
        (xs >= 0) &
        (xs < grid_map.shape[1]) &
        (ys >= 0) &
        (ys < grid_map.shape[0])
    )

    invalid_indices = np.where(valid == False)[0]

    if len(invalid_indices) > 0:
        first_invalid = invalid_indices[0]
        xs = xs[:first_invalid]
        ys = ys[:first_invalid]
        steps = steps[:first_invalid]

    hits = np.where(grid_map[ys, xs] == 1)[0]

    if len(hits) > 0:
        return int(steps[hits[0]])

    if len(invalid_indices) > 0:
        return int(invalid_indices[0])

    return max_distance

sensor_offsets = []

num_sensor_rays = 15

for i in range(num_sensor_rays):
    angle = -math.pi + i * (2 * math.pi / num_sensor_rays)
    sensor_offsets.append(angle)

def get_all_sensor_distances(x, y, theta, sensor_offsets, grid_map, max_sensor_distance):
    distances = []

    for offset in sensor_offsets:
        sensor_theta = theta + offset
        distance = get_sensor_distance(x, y, sensor_theta, grid_map, max_sensor_distance)
        distances.append(distance)

    return distances

def estimate_pose(particles):
    total_weight = 0.0

    for particle in particles:
        total_weight += particle["weight"]

    if total_weight == 0:
        return None

    weighted_x_sum = 0.0
    weighted_y_sum = 0.0
    cos_sum = 0.0
    sin_sum = 0.0

    for particle in particles:
        normalized_weight = particle["weight"] / total_weight

        weighted_x_sum += particle["x"] * normalized_weight
        weighted_y_sum += particle["y"] * normalized_weight
        cos_sum += math.cos(particle["theta"]) * normalized_weight
        sin_sum += math.sin(particle["theta"]) * normalized_weight

    estimated_theta = math.atan2(sin_sum, cos_sum)

    estimated_pose = {
        "x": int(weighted_x_sum),
        "y": int(weighted_y_sum),
        "theta": estimated_theta
    }

    return estimated_pose

def calculate_position_spread(particles, estimated_pose):
    total_weight = 0.0

    for particle in particles:
        total_weight += particle["weight"]

    if total_weight == 0 or estimated_pose is None:
        return None

    weighted_distance_sum = 0.0

    for particle in particles:
        normalized_weight = particle["weight"] / total_weight

        dx = particle["x"] - estimated_pose["x"]
        dy = particle["y"] - estimated_pose["y"]
        distance = math.sqrt(dx * dx + dy * dy)

        weighted_distance_sum += distance * normalized_weight

    return weighted_distance_sum

def calculate_angle_concentration(particles):
    total_weight = 0.0

    for particle in particles:
        total_weight += particle["weight"]

    if total_weight == 0:
        return None

    cos_sum = 0.0
    sin_sum = 0.0

    for particle in particles:
        normalized_weight = particle["weight"] / total_weight

        cos_sum += math.cos(particle["theta"]) * normalized_weight
        sin_sum += math.sin(particle["theta"]) * normalized_weight

    concentration = math.sqrt(cos_sum * cos_sum + sin_sum * sin_sum)

    return concentration

#main loop
while True:
    display = cv2.cvtColor(map_img, cv2.COLOR_GRAY2BGR) #convert map into 3 channel color image

    for particle in particles:
        cv2.circle(display, (particle["x"], particle["y"]), 2, (255, 0, 255), -1)

    #draw filled red circle at the robot position
    cv2.circle(display, (robot["x"], robot["y"]), robot_radius, (0, 0, 255), -1)

    #draw robot's direction line
    line_length = 20
    end_x = int(robot["x"] + line_length * math.cos(robot["theta"]))
    end_y = int(robot["y"] + line_length * math.sin(robot["theta"]))

    cv2.line(display, (robot["x"], robot["y"]), (end_x, end_y), (0, 255, 0), 2)

    key = cv2.waitKey(50)

    new_x = robot["x"]
    new_y = robot["y"]
    move_amount = 0
    turn_amount = 0

    #controls
    action_taken = False
    if key == ord('w'): #move forward
        move_amount = move_step
        new_x += int(move_step*math.cos(robot["theta"]))
        new_y += int(move_step*math.sin(robot["theta"]))
        action_taken = True

    elif key == ord('s'): #move backward
        move_amount = -move_step
        new_x -= int(move_step*math.cos(robot["theta"]))
        new_y -= int(move_step*math.sin(robot["theta"]))
        action_taken = True
    
    elif key == ord('a'): #rotate left
        turn_amount = -turn_step
        robot["theta"] = normalize_angle(robot["theta"] + turn_amount)
        action_taken = True

    elif key == ord('d'): #rotate right
        turn_amount = turn_step
        robot["theta"] = normalize_angle(robot["theta"] + turn_amount)
        action_taken = True

    elif key == ord('r'):
        robot = generate_random_robot(grid_map, robot_radius)
        particles = generate_random_particles(num_particles, grid_map, robot_radius)
        continue

    elif key == 27: #esc key
        break

    if is_valid_position(new_x,new_y,grid_map, robot_radius):
        robot["x"] = new_x
        robot["y"] = new_y

    if action_taken:
        move_particles(particles, move_amount, turn_amount, grid_map, robot_radius)

    #sensor
    max_sensor_distance = max(width, height)

    robot_distances = get_all_sensor_distances(
        robot["x"],
        robot["y"],
        robot["theta"],
        sensor_offsets,
        grid_map,
        max_sensor_distance)

    if action_taken:
        print("Sensor distances:", robot_distances)

    for particle in particles:
        particle["weight"] = calculate_particle_weight(particle, robot_distances, sensor_offsets, grid_map, max_sensor_distance)
    
    estimated_pose = estimate_pose(particles)
    position_spread = calculate_position_spread(particles, estimated_pose)
    angle_concentration = calculate_angle_concentration(particles)

    if action_taken:
        if estimated_pose is not None:
            print("Robot theta:", robot["theta"])
            print("Estimated theta:", estimated_pose["theta"])

    if angle_concentration is not None:
        if action_taken:
            print("Angle concentration", angle_concentration)

    if action_taken:
        if estimated_pose is not None:
            print("Estimated pose:", estimated_pose)
            print("Position spread:", position_spread)
    
    position_error = None

    if estimated_pose is not None:
        dx = robot["x"] - estimated_pose["x"]
        dy = robot["y"] - estimated_pose["y"]
        position_error = math.sqrt(dx * dx + dy * dy)

    error_percent = (position_error / max_distance) * 100
    
    if position_error is not None:
        if action_taken:
            print("position error:", position_error)
            
        cv2.putText(
            display,
            "Error: " + str(round(error_percent, 2)) + "%",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

    converged = False

    if position_spread is not None and angle_concentration is not None:
        if position_spread < spread_threshold and angle_concentration > angle_threshold:
            converged = True

    if converged:
        status_text = "Status: Converged"
    else:
        status_text = "Status: Not converged"

    if action_taken:
        print(status_text)

    cv2.putText(
        display,
        status_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2
    )

    if action_taken:
        particles = resample_particles(particles)
    
    overlay = display.copy()
    for i in range(len(sensor_offsets)):
        sensor_theta = robot["theta"] + sensor_offsets[i]
        sensor_distance = robot_distances[i]

        sensor_start_x = int(robot["x"] + robot_radius * math.cos(sensor_theta))
        sensor_start_y = int(robot["y"] + robot_radius * math.sin(sensor_theta))
        sensor_end_x = int(robot["x"] + sensor_distance * math.cos(sensor_theta))
        sensor_end_y = int(robot["y"] + sensor_distance * math.sin(sensor_theta))

        cv2.line(overlay, (sensor_start_x, sensor_start_y), (sensor_end_x, sensor_end_y), (255, 0, 0), 2)
    
    alpha = 0.2  # transparency level
    display = cv2.addWeighted(overlay, alpha, display, 1 - alpha, 0)

    if estimated_pose is not None:
    #draw estimated position (yellow)
        cv2.circle(display, (estimated_pose["x"], estimated_pose["y"]), 6, (0, 255, 255), -1)

        # draw direction line
        line_length = 20
        end_x = int(estimated_pose["x"] + line_length * math.cos(estimated_pose["theta"]))
        end_y = int(estimated_pose["y"] + line_length * math.sin(estimated_pose["theta"]))

        cv2.line(
            display,
            (estimated_pose["x"], estimated_pose["y"]),
            (end_x, end_y),
            (0, 255, 255),
            2)

    real_text = "Real: (" + str(robot["x"]) + ", " + str(robot["y"]) + ")"

    cv2.putText(
        display,
        real_text,
        (10, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2
        )
        
    if estimated_pose is not None:

        estimate_text = "Estimate: (" + str(estimated_pose["x"]) + ", " + str(estimated_pose["y"]) + ")"

        cv2.putText(
            display,
            estimate_text,
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2)

    cv2.imshow("Robot on Map", display)

cv2.destroyAllWindows()