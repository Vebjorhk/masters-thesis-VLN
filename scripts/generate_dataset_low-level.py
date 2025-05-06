import os
import json
import time
import math
import random
import argparse
import shutil
from multiprocessing import Pool

import MatterSim
import numpy as np
import cv2 
import numpy as np
import networkx as nx

# having a lower turn rate might result in a much larger dataset, check that out later
TURN_LEFT = -math.radians(30)
TURN_RIGHT = math.radians(30)

def create_dataset_folder(dataset_folder="./code/dataset"):
    images_folder = os.path.join(dataset_folder, "images")
    
    os.makedirs(images_folder, exist_ok=True)
    return images_folder


def create_episode_folder(dataset_path, path_id):
    image_path = os.path.join(dataset_path, str(path_id))
    
    if not os.path.exists(image_path):
            os.makedirs(image_path, exist_ok=True)

    return image_path

def save_image(filename, rgb):
    img = np.array(rgb, copy=False)
    # Resize image to half size with INTER_LANCZOS4, this is also used during evaluation
    height, width = img.shape[:2]
    img = cv2.resize(img, (width // 2, height // 2), interpolation=cv2.INTER_LANCZOS4)
    cv2.imwrite(filename, img)
    
    
def get_path_mapping(data):
    path_dict = {}
    for p in data:
        path_dict[p["path_id"]] = p
    return path_dict

def setup_simulator():
    sim = MatterSim.Simulator()
    sim.setCameraResolution(640, 480)
    sim.setPreloadingEnabled(False)
    sim.setDepthEnabled(False)
    sim.setDiscretizedViewingAngles(False) # to 30 degrees
    sim.setCameraVFOV(math.radians(105)) # 105 degrees vertical field of view HUSK Å ENDRE DETTE SENERE
    sim.setBatchSize(1)
    sim.setCacheSize(2000)
    sim.initialize()

    return sim

def determine_possible_actions(locations):
    # means there are no nodes in the field of view
    if len(locations) < 2:
        return ["Turn Left", "Turn Right", "Stop Navigation"]
    
    return ["Turn Left", "Turn Right", "Move Forward", "Stop Navigation"]

    
def determine_optimal_turn_action(sim, next_node):
    """ Finds the optimal direction to turn """
    def find_closest_turn(sim, action, reset_action):
        closest = 20
        found = False

        for step in range(6):
            sim.makeAction([0], [action], [0])
            locations = sim.getState()[0].navigableLocations
            viewpoints = [loc.viewpointId for loc in locations]
            
            if next_node in viewpoints:
                node_index = viewpoints.index(next_node)
                if node_index == 1:
                    if abs(locations[node_index].rel_heading) < math.radians(30):
                        found = True
                        closest = 1
                        break
                else:
                    if abs(locations[node_index].rel_heading) < math.radians(30):
                        closest = min(closest, node_index)
                    #return 1, True, step + 1

        for _ in range(step + 1):
            sim.makeAction([0], [reset_action], [0])

        # closest = the index in the viewpoints where the next node is found (should be index 1, if not, another paths needs to be found)
        return closest, found, step + 1

    # Find closest turns for both directions
    closest_left, found_left, left_steps = find_closest_turn(sim, TURN_LEFT, TURN_RIGHT)
    closest_right, found_right, right_steps = find_closest_turn(sim, TURN_RIGHT, TURN_LEFT)

    if found_left and found_right:
        if left_steps == right_steps:
            return (TURN_LEFT if random.random() < 0.5 else TURN_RIGHT, 1)
        return (TURN_LEFT, 1) if left_steps < right_steps else (TURN_RIGHT, 1)
    
    elif found_left and not found_right:
        return (TURN_LEFT, 1)
    
    elif found_right and not found_left:
        return (TURN_RIGHT, 1)


    if closest_left < closest_right:
        return TURN_LEFT, closest_left
    if closest_right < closest_left:
        return TURN_RIGHT, closest_right

    return TURN_LEFT if random.random() < 0.5 else TURN_RIGHT, closest_left


def create_data_sample(path_data, sim, step_counter, image_history, distance_traveled, action, action_history, possible_actions, path_history):
    return {
                "path_id" : path_data["path_id"],
                "step_id" : step_counter,
                "scan" : path_data["scan"],
                "current_heading" : sim.getState()[0].heading,
                "current_node" : sim.getState()[0].navigableLocations[0].viewpointId,
                "current_image" : f"images/{path_data['path_id']}/step_{step_counter}.png",
                "past_images" : image_history.copy(),
                "distance_traveled" : distance_traveled,
                "instructions" : path_data["instructions"],
                "gold_label" : action,
                "previous_actions" : action_history.copy(),
                "possible_actions" : possible_actions.copy(),
                "path_history" : path_history.copy(),
            }

def load_nav_graphs(scans):
    ''' Load connectivity graph for each scan '''

    def distance(pose1, pose2):
        ''' Euclidean distance between two graph poses '''
        return ((pose1['pose'][3]-pose2['pose'][3])**2\
          + (pose1['pose'][7]-pose2['pose'][7])**2\
          + (pose1['pose'][11]-pose2['pose'][11])**2)**0.5

    graphs = {}
    for scan in scans:
        with open('connectivity/%s_connectivity.json' % scan) as f:
            G = nx.Graph()
            positions = {}
            data = json.load(f)
            for i,item in enumerate(data):
                if item['included']:
                    for j,conn in enumerate(item['unobstructed']):
                        if conn and data[j]['included']:
                            positions[item['image_id']] = np.array([item['pose'][3],
                                    item['pose'][7], item['pose'][11]]);
                            assert data[j]['unobstructed'][i], 'Graph should be undirected'
                            G.add_edge(item['image_id'],data[j]['image_id'],weight=distance(item,data[j]))
            nx.set_node_attributes(G, values=positions, name='position')
            graphs[scan] = G
    return graphs


def generate_dataset(dataset_path, r2r_folder, run_range=(0, 200), seed=32, auto_adjust=True):
    random.seed(seed)
    
    with open(r2r_folder, encoding="utf-8") as file:
        data = json.load(file)
    
    dataset = []
    
    for i in range(run_range[0], run_range[1]):
        sim = setup_simulator()
        tmp_dataset = []
        path_data = data[i]
        
        
        scan = path_data["scan"]
        G = load_nav_graphs([scan])[scan]
        
        # nodes
        node_path = path_data["path"]
        current_node = node_path[0]
        target_node = node_path[-1]
        
        # states
        step_counter = 0
        node_counter = 1
        target_index = 1 # used to find alternative routes
        turn_action = 3
        just_changed_node = True 
        
        # history
        image_history = []
        action_history = []
        path_history = [current_node]
        distance_traveled = 0
        
        save_path = create_episode_folder(dataset_path, path_data["path_id"])
        
        sim.newEpisode([path_data["scan"]], [path_data["path"][0]], [path_data["heading"]], [0])
        
        while current_node != target_node and step_counter <= 30:
            action = None
            locations = sim.getState()[0].navigableLocations
            viewpoints = [i.viewpointId for i in locations]
            next_node = node_path[-1] if node_counter >= len(node_path) else node_path[node_counter]

            possible_actions = determine_possible_actions(locations)

            image_path = f"{save_path}/step_{step_counter}.png"
            image_history_path = f"images/{path_data['path_id']}/step_{step_counter}.png"

            rgb = sim.getState()[0].rgb
            save_image(image_path, rgb) # this part saves the actual image for each run

            target_loc = None

            if next_node in viewpoints:
                tmp = viewpoints.index(next_node)
                if tmp == target_index:
                    target_loc = viewpoints[target_index]

            if next_node != target_loc or abs(locations[target_index].rel_heading) > math.radians(30):
                if just_changed_node == True:
                    action, target_index = determine_optimal_turn_action(sim, next_node)

                    # needs to find a new route as the agent cannot move straight forward (pick node at index 1)
                    if target_index > 1:
                        g_copy = G.copy()
                        # removes the edge between current_node and next_node to find a new shortest paths
                        if g_copy.has_edge(current_node, next_node):
                            g_copy.remove_edge(current_node, next_node)

                        try:
                            # calculates the new shortest path from current_node to next_node
                            tmp_path = nx.shortest_path(g_copy, source=viewpoints[0], target=next_node, weight="weights")

                        except:
                            print(f"Scan: {path_data['scan']}, path_id: {path_data['path_id']}; no path between {viewpoints[0]} and {next_node}, giving up on this path")
                            break

                        node_path = node_path[:node_counter] + tmp_path[1:-1] + node_path[node_counter:]
                        next_node = node_path[-1] if node_counter >= len(node_path) else node_path[node_counter]

                        # determines if it is still navigable and what action to select
                        action, target_index = determine_optimal_turn_action(sim, next_node)

                    # if the relative heading for the next node is larger than 30 degrees it makes more sense to turn one more time to align better
                    if next_node in viewpoints and viewpoints.index(next_node) == target_index and abs(locations[target_index].rel_heading) < math.radians(30):
                        action = "Move Forward"

                    just_changed_node = False   
                    turn_action = action

                if action != "Move Forward":
                    action = "Turn Left" if turn_action == TURN_LEFT else "Turn Right"

            # Moves forward
            else:
                action = "Move Forward"
            
            # adds the action to the dataset
            tmp_dataset.append(
                create_data_sample(
                    path_data, 
                    sim, 
                    step_counter, 
                    image_history,
                    distance_traveled, 
                    action, 
                    action_history, 
                    possible_actions, 
                    path_history
                )
            )
            image_history.append(image_history_path) 

            # actual movement in simulator
            if action == "Move Forward":
                if auto_adjust:
                    step_counter += 1
                    sim.makeAction([0], [locations[target_index].rel_heading], [0])
                    image_path = f"{save_path}/step_{step_counter}.png"
                    image_history_path = f"images/{path_data['path_id']}/step_{step_counter}.png"
                    rgb = sim.getState()[0].rgb
                    save_image(image_path, rgb)
                    image_history.append(image_history_path)
                    action_history.append("Automatically Turn Towards Node")

                tmp_distance = locations[target_index].rel_distance
                sim.makeAction([1], [0], [0])

                # in this case the navigation was able to find the correct path 
                #if target_index == 1:
                node_counter += 1
                distance_traveled += tmp_distance
                just_changed_node = True
                target_index = 1
                current_node = sim.getState()[0].navigableLocations[0].viewpointId
                path_history.append(current_node)

            else:
                sim.makeAction([0], [turn_action], [0])

            step_counter += 1
            action_history.append(action) 
        
        # End of while loop:

        # The agent always stops when it has selected Stop token
        image_path = f"{save_path}/step_{step_counter}.png"
        rgb = sim.getState()[0].rgb
        save_image(image_path, rgb)

        possible_actions = determine_possible_actions(sim.getState()[0].navigableLocations)
        tmp_dataset.append(
            create_data_sample(
                    path_data, 
                    sim, 
                    step_counter, 
                    image_history, 
                    distance_traveled,
                    "Stop Navigation", 
                    action_history, 
                    possible_actions, 
                    path_history
            )
        )
        
        if tmp_dataset[-1]["path_history"][-1] == path_data["path"][-1]:
            dataset.extend(tmp_dataset)
            
        else:
            # fjerner bildene hvis de ikke går, orker ikke bruke mer tid på dette, ser ut som noen paths bare suger
            #shutil.rmtree(save_path)
            print(f"path_id: {path_data['path_id']} was not included")
            
        sim.close()
        
    return dataset


if __name__ == "__main__":
    # TODO: remember to mark data examples where this programmed agent did not reach its goal.
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", help="the path to where the dataset should be stored", default="")
    parser.add_argument("--r2r_folder", help="path to where the room-to-room dataset is stored", default="tasks/R2R/data/R2R_train.json")
    parser.add_argument("--dataset_filename", help="The filename for the dataset", default="train_data.json")
    parser.add_argument("--dataset_type", help="what type of dataset is generated; train, val, test ...", default="train")
    parser.add_argument("--start_index", help="Start index of paths to process", type=int, default=0)
    parser.add_argument("--end_index", help="End index of paths to process", type=int, default=200)
    parser.add_argument("--random_chance", help="the probability for random event", type=float, default=0.25)
    parser.add_argument("--num_workers", help="How many parallel processes", type=int, default=5)
    parser.add_argument("--seed", help="seed for python random events", type=int, default=32)
    parser.add_argument("--auto_adjust", help="If the agent should auto adjust its heading towards the next node", type=bool, default=False)
    args = parser.parse_args()
    
    print(args.auto_adjust)
    dataset_path = create_dataset_folder(args.dataset_path)
    dataset_name = os.path.join(args.dataset_path, args.dataset_filename)

    num_workers = args.num_workers
    path_range = (args.end_index - args.start_index) // num_workers

    start_indices = []
    for i in range(num_workers):
        start = args.start_index + i * path_range
        end = start + path_range if i < num_workers - 1 else args.end_index
        start_indices.append((start, end))


    with Pool(num_workers) as pool:
        tasks = [
            (dataset_path, args.r2r_folder, run_range, args.seed+i, args.auto_adjust)
            for i, run_range in enumerate(start_indices)
        ]
        results = pool.starmap(generate_dataset, tasks)


    if os.path.exists(dataset_name):
        with open(dataset_name, "r", encoding="utf-8") as file:
            dataset = json.load(file)
            
    else:
        dataset = []

    for result in results:
        dataset.extend(result)

    print(f"Saved to: {dataset_name}")
    with open(dataset_name, "w") as file:
        json.dump(dataset, file, indent=4)

    metadata = {
        "seed" : args.seed,
        "type" : args.dataset_type,
        "num_workers" : args.num_workers,
        "data_source" : "Room-to-Room",
        "simulator" : "MatterPort3D",
        "about" : "This time the action space is reduced to only: Turn Left, Turn Right, Move Forward and Stop Navigation, auto_adjust = True"
    }

    with open(os.path.join(args.dataset_path, f"{args.dataset_type}_data_meta.json"), "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)
    
    