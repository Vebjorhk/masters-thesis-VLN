import os
import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from config import config
from simulator import SimulatorV5, SimulatorV5Train, SimulatorV3
from utils import load_dataset, get_path_mapping

sim_sessions = {}  
model_path = {} 
model_action_path = {}  
submission_data = []
struct = {}

episode_routes = APIRouter()
action_routes = APIRouter()

r2r_data = load_dataset(config["dataset"])
path_dict = get_path_mapping(r2r_data)



@episode_routes.post("/start_episode/{path_id}")
async def start_episode(path_id: str):
    try:
        global struct
        path_data = path_dict[int(path_id)]
        if config["simulator"] == "v5Train":
            sim = SimulatorV5Train(path_data)

        elif config["simulator"] == "v3":
            sim = SimulatorV3(path_data)
        else:
            sim = SimulatorV5(path_data)

        sim_sessions[path_id] = sim

        model_path[path_id] = [sim.get_current_viewpoint()]
        
        model_action_path[path_id] = []

        content = sim.get_content()

        struct = {"instr_id" : f"{path_id}_{config['instruction_index']}", "trajectory" : [] }
        
        return JSONResponse(content=content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting the episode: {str(e)}")

@action_routes.post("/{session_id}")
async def take_action(session_id: str, action: str):
    try:
        global struct
        if session_id not in sim_sessions:
            raise HTTPException(status_code=404, detail="Session not found")

        sim = sim_sessions[session_id]
        state = sim.getState()
        current_heading = state.heading
        current_evaluation = state.elevation
        struct["trajectory"].append([state.location.viewpointId, current_heading, current_evaluation])


        # this here to check if the agent actually chose to stop or not
        model_action_path[session_id].append(action)

        if action != "Stop":
            sim.make_action(action)
            model_path[session_id].append(sim.get_current_viewpoint())


        else:
            print("User will call end_episode")

        content = sim.get_content()

        return JSONResponse(content=content)
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=f"Error processing the action: {str(e)}")


@episode_routes.delete("/end_episode/{session_id}")
async def end_episode(session_id: str):
    global submission_data
    try:
        if session_id not in sim_sessions:
            raise HTTPException(status_code=404, detail="Session not found")

        sim = sim_sessions[session_id]
        sim.end_episode()

        sim_sessions[session_id] = None  

        if os.path.exists(config["model_action_path"]):
            with open(config["model_action_path"], "r", encoding="utf-8") as file:
                actions = json.load(file)

        else:
            actions = {}
        
        if os.path.exists(config["save_path"]):
            with open(config["save_path"], "r", encoding="utf-8") as file:
                data = json.load(file)

        else:
            data = {}

        if os.path.exists(config["submission_path"]):
            with open(config["submission_path"], "r", encoding="utf-8") as file:
                submission_data = json.load(file)

        else:
            submission_data = []


        data[session_id] = model_path[session_id]
        actions[session_id] = model_action_path[session_id]

        with open(config["save_path"], "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4)

        with open(config["model_action_path"], "w", encoding="utf-8") as file:
            json.dump(actions, file, indent=4)

        with open(config["submission_path"], "w", encoding="utf-8") as file:
            submission_data.append(struct)
            json.dump(submission_data, file, indent=4)

        return {"message": "Episode ended successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error ending the episode: {str(e)}")
