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


VFOV = math.radians(105)
WIDTH = 640
HEIGHT = 480

def build_simulator():
    sim = MatterSim.Simulator()
    sim.setCameraVFOV(VFOV) 
    sim.setCameraResolution(WIDTH, HEIGHT)
    sim.setPreloadingEnabled(False)
    sim.setDepthEnabled(False)
    sim.setRestrictedNavigation(False)
    sim.setDiscretizedViewingAngles(False) # 30 degrees viewing angles
    sim.setBatchSize(1)
    sim.setCacheSize(2000)
    sim.initialize()

    return sim

def generate_panorama(sim):

    aspect_ratio = WIDTH / HEIGHT  

    # Calculate horizontal FOV using the formula
    HFOV = 2 * np.arctan(np.tan(VFOV / 2) * aspect_ratio)

    overlap_fraction = 0.00  

    # Calculate step rotation
    step_rotation = HFOV * (1 - overlap_fraction)

    # Total rotation for a full circle (2π radians)
    total_rotation = 2 * np.pi
    
    # Calculate the number of images needed
    num_images = int(np.round(total_rotation / step_rotation))
    
    step_rotation = total_rotation / num_images
    images = []
    
    for i in range(num_images):
        rgb = np.array(sim.getState()[0].rgb)

        images.append(rgb)
        sim.makeAction([0], [step_rotation], [0])
    
    panorama = np.hstack(images)

    # Calculate the center of the whole panorama
    current_center = panorama.shape[1] // 2

    # Calculate the center of the first image (the first image should be centered in the panorama)
    first_image_center = images[0].shape[1] // 2

    # Calculate the shift required to align the first image's center with the overall panorama center
    shift_amount = current_center - first_image_center

    # Apply the shift (negative shift means move left)
    shifted_panorama = np.roll(panorama, shift_amount, axis=1)
    
    return shifted_panorama

def create_dataset_folder(dataset_folder="./code/dataset"):
    images_folder = os.path.join(dataset_folder, "images")
    
    os.makedirs(images_folder, exist_ok=True)
    return images_folder


def create_episode_folder(dataset_path, path_id):
    images_folder = os.path.join(dataset_path, str(path_id))
    
    os.makedirs(images_folder, exist_ok=True)
    return images_folder

def save_image(save_path, img):
    img = np.array(img, copy=False)
    #img = cv2.cvtColor(x, cv2.COLOR_RGB2BGR) denne linjen var problemet for rgb bildene
    cv2.imwrite(save_path, img)
    
    
def get_path_mapping(data):
    path_dict = {}
    for p in data:
        path_dict[p["path_id"]] = p
    return path_dict


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

def get_relative_direction(loc_heading_deg):
    if 45 <= loc_heading_deg <= 135:
        return "Right"
    elif -135 <= loc_heading_deg <= -45:
        return "Left"
    elif -45 < loc_heading_deg < 45:
        return "Front"
    else:
        return "Behind"


def generate_dataset(dataset_path, r2r_folder, run_range=(0, 200), seed=32):
    random.seed(seed)
    
    with open(r2r_folder, encoding="utf-8") as file:
        data = json.load(file)
    
    dataset = []

    for i in range(run_range[0], run_range[1]):
        sim = build_simulator()
        tmp_dataset = []

        path_data = data[i]
        scan = path_data["scan"]
        path = path_data["path"]
        initial_heading = path_data["heading"]
        start_node = path[0]
        
        path_history = []
        image_history = []
        candidate_history = []
        
        
        episode_folder = create_episode_folder(dataset_path, path_data["path_id"])
        
        sim.newEpisode([scan], [start_node], [initial_heading], [0])

        for i in range(len(path)):
            pano = generate_panorama(sim)
            pano_path = os.path.join(episode_folder, f"pano_step_{i}.png") 
            save_image(pano_path, pano)
            
            target_node = path[i+1] if i+1 < len(path) else None
            state = sim.getState()[0]
            locations = state.navigableLocations
            sorted_locations = sorted(locations[1:], key=lambda x: x.rel_heading) # sorted from left to right
            
            viewpoints = [loc.viewpointId for loc in locations] # used for makeAction lookup
            path_history.append(viewpoints[0])
            
            candidates = {}
            gold_label = None
            
            for j, loc in enumerate(sorted_locations):
                loc_heading = loc.rel_heading
                loc_heading_deg = math.degrees(loc_heading)

                sim.makeAction([0], [loc_heading], [0])

                rgb = sim.getState()[0].rgb
                image_path = os.path.join(episode_folder, f"step_{i}_candidate_{j}.png")
                save_image(image_path, rgb)

                relative_direction = get_relative_direction(loc_heading_deg)

                if loc.viewpointId == target_node:
                    gold_label = j
                    
                candidates[j] = {
                        "image_path" : image_path,
                        "relative_angle" : loc_heading_deg,
                        "relative_direction" : relative_direction,
                        "distance" : loc.rel_distance 
                    }
                
                sim.makeAction([0], [-loc_heading], [0])
            
            if target_node != None:
                move_index = viewpoints.index(target_node)
                sim.makeAction([move_index], [locations[move_index].rel_heading], [0])
                
            else:
                gold_label = "Stop"
                
            tmp_dataset.append(
                {
                    "path_id" : path_data["path_id"],
                    "step_id" : i,
                    "scan" : scan,
                    "current_node" : viewpoints[0],
                    "current_heading" : state.heading,
                    "current_image" : pano_path,
                    "image_history" : image_history.copy(),
                    "instructions" : path_data["instructions"],
                    "gold_label" : gold_label,
                    "path_history" : path_history.copy(),
                    "candidate_history" : candidate_history.copy(),
                    "candidates" : candidates,
                }
            )
            
            candidate_history.append(gold_label)
            image_history.append(pano_path)

        assert path_history == path
        dataset.extend(tmp_dataset)
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
    args = parser.parse_args()

    dataset_path = create_dataset_folder(args.dataset_path)
    dataset_name = os.path.join(args.dataset_path, args.dataset_filename)

    total_paths = args.end_index - args.start_index
    num_workers = min(args.num_workers, total_paths)
    path_range = (args.end_index - args.start_index) // num_workers
    remainder = total_paths % num_workers

    start_indices = []
    for i in range(num_workers):
        start = args.start_index + i * path_range
        end = start + path_range if i < num_workers - 1 else args.end_index
        start_indices.append((start, end))


    with Pool(num_workers) as pool:
        tasks = [
            (dataset_path, args.r2r_folder, run_range, args.seed+i)
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
        "about" : "This time the action space is reduced to only: Turn Left, Turn Right, Move Forward and Stop Navigation (no autmoatically adjust)"
    }

    with open(os.path.join(args.dataset_path, f"{args.dataset_type}_data_meta.json"), "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    unique_paths = set(item["path_id"] for item in dataset)
    print(f"Unique paths: {len(unique_paths)}")