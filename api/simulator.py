import cv2
import math
import numpy as np
import base64
import json 

import networkx as nx

import MatterSim


VFOV = math.radians(105) # 82 degrees vertical field of view HUSK Å ENDRE DETTE TIL 105 hvis det ikke er like bra
TURN_LEFT = -math.radians(30)
TURN_RIGHT = math.radians(30)
WIDTH = 640
HEIGHT = 480

class SimulatorV3():
    def __init__(self, path_data):
        self.sim = self.build_simulator()
        self.start_new_episode(path_data)

    def build_simulator(self):
        sim = MatterSim.Simulator()
        sim.setCameraVFOV(VFOV) 
        sim.setCameraResolution(WIDTH, HEIGHT)
        sim.setPreloadingEnabled(False)
        sim.setDepthEnabled(False)
        sim.setRestrictedNavigation(True) # want to see all nodes no matter if it is in filed of view
        sim.setDiscretizedViewingAngles(False) # 30 degrees viewing angles False
        sim.setBatchSize(1)
        sim.setCacheSize(2000)
        sim.initialize()

        return sim
    
    def start_new_episode(self, path_data):
        self.sim.newEpisode([path_data["scan"]], [path_data["path"][0]], [path_data["heading"]], [0])
        self.distance = 0
        self.step()

    def end_episode(self):
        self.sim.close() # explicitly closes the simulator to free up space

    def step(self):
        self.state = self.sim.getState()[0]
        self.locations = self.state.navigableLocations
        self.viewpoints = [i.viewpointId for i in self.locations]
        self.move_possible = True if len(self.viewpoints) > 1 else False

    def make_action(self, action: str):
        """ Moves the agent towards the chosen candidate node
        """
        if action == "Move": 
            if self.move_possible:
                self.distance += self.locations[1].rel_distance
            self.sim.makeAction([1], [0], [0])

        elif action == "Adjust":
            self.sim.makeAction([0], [self.locations[1].rel_heading], [0])

        elif action == "Left":
            self.sim.makeAction([0], [TURN_LEFT], [0])

        elif action == "Right":
            self.sim.makeAction([0], [TURN_RIGHT], [0])

        self.step()

    def get_current_viewpoint(self):
        return self.sim.getState()[0].navigableLocations[0].viewpointId
    
    def __encode_image_base64(self, image, width, height):
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LANCZOS4)
        _, img_encoded = cv2.imencode('.png', image) # hva faen skjer med dette????? Dette må faktisk undersøkes
        img_bytes = img_encoded.tobytes()
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')  # Convert image to base64
        return img_base64
    
    def get_image(self):
        rgb = np.array(self.sim.getState()[0].rgb)
        return self.__encode_image_base64(rgb, 320, 240)
    
    def get_content(self):
        return {
            "image" : self.get_image(),
            "move_possible" : self.move_possible,
            "distance" : self.distance
        }
    
    def getState(self):
        return self.sim.getState()[0]
    

# One instance for each path (not optimal but since we use batch size = 1 it doesnt make any difference, also solves a lot of bugs)
class SimulatorV5():
    def __init__(self, path_data):
        self.sim = self.build_simulator()
        self.start_new_episode(path_data)
        
    def build_simulator(self):
        sim = MatterSim.Simulator()
        sim.setCameraVFOV(VFOV) 
        sim.setCameraResolution(WIDTH, HEIGHT)
        sim.setPreloadingEnabled(False)
        sim.setDepthEnabled(False)
        sim.setRestrictedNavigation(False) # want to see all nodes no matter if it is in filed of view
        sim.setDiscretizedViewingAngles(False) # 30 degrees viewing angles False
        sim.setBatchSize(1)
        sim.setCacheSize(2000)
        sim.initialize()

        return sim
    
    def start_new_episode(self, path_data):
        self.sim.newEpisode([path_data["scan"]], [path_data["path"][0]], [path_data["heading"]], [0])
        self.step()

    def getState(self):
        return self.sim.getState()[0]
    
    def end_episode(self):
        self.sim.close() # explicitly closes the simulator to free up space

    def step(self):
        self.state = self.sim.getState()[0]
        self.locations = self.state.navigableLocations
        self.viewpoints = [i.viewpointId for i in self.locations]
        # sort from left to rigth
        self.sorted_locations = sorted(self.locations[1:], key=lambda x: x.rel_heading)

    def make_action(self, candidate: str):
        """ Moves the agent towards the chosen candidate node
        """
        selected_loc = self.sorted_locations[int(candidate)] # for rel heading
        node_index = self.viewpoints.index(selected_loc.viewpointId) # for index in the simulator as they are sorted from rel left to right
        # print(node_index)
        # moves towards node on node_index and change the heading towards that node
        self.sim.makeAction([node_index], [selected_loc.rel_heading], [0])
        self.step()
        
    def get_current_viewpoint(self):
        return self.sim.getState()[0].navigableLocations[0].viewpointId

    
    def get_candidates(self):
        """ fetches rgb, relative_distance and relative_angle for each candidate
        """
        candidates = {}

        for i, loc in enumerate(self.sorted_locations):
            rel_heading = loc.rel_heading
            rel_distance = loc.rel_distance
            self.sim.makeAction([0], [loc.rel_heading], [0])
            rgb = np.array(self.sim.getState()[0].rgb)

            candidates[f"can_{i}"] = {
                "image" : self.__encode_image_base64(rgb, 320, 240),
                "relative_angle" : math.degrees(rel_heading),
                "distance" : rel_distance
            }

            self.sim.makeAction([0], [-rel_heading], [0])
        
        return candidates
    
    def get_panorama(self):
        panorama = self.__generate_panorama()
        return self.__encode_image_base64(panorama, 960, 240)
    
    # inferes cumulative distance and step id on the reciving side
    def get_content(self):
        return {
            "panorama" : self.get_panorama(),
            "candidates" : self.get_candidates()
        }


    def __encode_image_base64(self, image, width, height):
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LANCZOS4)
        _, img_encoded = cv2.imencode('.png', image) # hva faen skjer med dette????? Dette må faktisk undersøkes
        img_bytes = img_encoded.tobytes()
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')  # Convert image to base64
        return img_base64

    def __generate_panorama(self):
        aspect_ratio = WIDTH / HEIGHT  

        # Calculate horizontal FOV using the formula
        HFOV = 2 * np.arctan(np.tan(VFOV/ 2) * aspect_ratio)

        # Define overlap fraction
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
            rgb = np.array(self.sim.getState()[0].rgb)

            images.append(rgb)
            self.sim.makeAction([0], [step_rotation], [0])
        
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


# One instance for each path (not optimal but since we use batch size = 1 it doesnt make any difference, also solves a lot of bugs)
class SimulatorV5Train():
    def __init__(self, path_data):
        self.sim = self.build_simulator()
        self.path = path_data["path"]
        self.target = 1
        self.goal = self.path[-1]
        self.has_reached_goal = False
        self.start_new_episode(path_data)
        
    def build_simulator(self):
        sim = MatterSim.Simulator()
        sim.setCameraVFOV(VFOV) 
        sim.setCameraResolution(WIDTH, HEIGHT)
        sim.setPreloadingEnabled(False)
        sim.setDepthEnabled(False)
        sim.setRestrictedNavigation(False) # want to see all nodes no matter if it is in filed of view
        sim.setDiscretizedViewingAngles(False) # 30 degrees viewing angles False
        sim.setBatchSize(1)
        sim.setCacheSize(2000)
        sim.initialize()

        return sim
    
    def start_new_episode(self, path_data):
        self.sim.newEpisode([path_data["scan"]], [path_data["path"][0]], [path_data["heading"]], [0])
        self.nav_graph = self.load_nav_graphs([path_data["scan"]])[path_data["scan"]]
        self.step()

    def end_episode(self):
        self.sim.close() # explicitly closes the simulator to free up space

    def step(self):
        self.state = self.sim.getState()[0]
        self.locations = self.state.navigableLocations
        self.viewpoints = [i.viewpointId for i in self.locations]
        # sort from left to rigth
        self.sorted_locations = sorted(self.locations[1:], key=lambda x: x.rel_heading)
        self.gold_candidate = self.find_gold_candidate()

    def find_gold_candidate(self):
        viewpoints = [i.viewpointId for i in self.sorted_locations]
        target_viewpoint = self.path[self.target]
        current_viewpoint = self.state.location.viewpointId
        # dette blir jo feil, den skal jo ikke da returne stopp før den har reacha denne location, så hvis current location er det da skal den stoppe
        if current_viewpoint == self.goal:
            return -1
        
        if target_viewpoint == self.goal:
            self.has_reached_goal = True
        
        # må finne ut av hva man gjør hvis den over shooter
        if target_viewpoint in viewpoints:
            if self.has_reached_goal == False:
                self.target += 1

            return viewpoints.index(target_viewpoint)
        
        
        # in that case it has gone to the wrong node and must go back again

        target_path = nx.shortest_path(self.nav_graph, source=current_viewpoint, target=target_viewpoint, weight="weights")
        return viewpoints.index(target_path[1])


    def make_action(self, candidate: str):
        """ Moves the agent towards the chosen candidate node
        """
        # if the agent select a candidate which is not in the list return False

        if candidate.isnumeric() and int(candidate) < len(self.sorted_locations):
            selected_loc = self.sorted_locations[int(candidate)] # for rel heading
            node_index = self.viewpoints.index(selected_loc.viewpointId) # for index in the simulator as they are sorted from rel left to right
            # print(node_index)
            # moves towards node on node_index and change the heading towards that node
            self.sim.makeAction([node_index], [selected_loc.rel_heading], [0])
            self.step()
            return True

        print("her gikk det feil")
        return False
        
    def get_current_viewpoint(self):
        return self.sim.getState()[0].navigableLocations[0].viewpointId

    
    def get_candidates(self):
        """ fetches rgb, relative_distance and relative_angle for each candidate
        """
        candidates = {}

        for i, loc in enumerate(self.sorted_locations):
            rel_heading = loc.rel_heading
            rel_distance = loc.rel_distance
            self.sim.makeAction([0], [loc.rel_heading], [0])
            rgb = np.array(self.sim.getState()[0].rgb)

            candidates[f"can_{i}"] = {
                "image" : self.__encode_image_base64(rgb, 320, 240),
                "relative_angle" : math.degrees(rel_heading),
                "distance" : rel_distance,
                "viewpoint_id" : loc.viewpointId
            }

            self.sim.makeAction([0], [-rel_heading], [0])
        
        return candidates
    
    def get_panorama(self):
        panorama = self.__generate_panorama()
        return self.__encode_image_base64(panorama, 960, 240)
    
    # inferes cumulative distance and step id on the reciving side
    def get_content(self):
        return {
            "panorama" : self.get_panorama(),
            "candidates" : self.get_candidates(),
            "gold_candidate" : self.gold_candidate
        }


    def __encode_image_base64(self, image, width, height):
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LANCZOS4)
        _, img_encoded = cv2.imencode('.png', image) # hva faen skjer med dette????? Dette må faktisk undersøkes
        img_bytes = img_encoded.tobytes()
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')  # Convert image to base64
        cv2.imwrite("test.png", image)
        return img_base64

    def __generate_panorama(self):
        aspect_ratio = WIDTH / HEIGHT  

        # Calculate horizontal FOV using the formula
        HFOV = 2 * np.arctan(np.tan(VFOV/ 2) * aspect_ratio)

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
            rgb = np.array(self.sim.getState()[0].rgb)

            images.append(rgb)
            self.sim.makeAction([0], [step_rotation], [0])
            #time.sleep(0.1) kanksje se om dette går fortere
        
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

    def load_nav_graphs(self, scans):
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

