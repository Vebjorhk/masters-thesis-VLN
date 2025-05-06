import uvicorn
from fastapi import FastAPI
from routes import episode_routes, action_routes

app = FastAPI()

app.include_router(episode_routes, prefix="/episode", tags=["Episode"])
app.include_router(action_routes, prefix="/take_action", tags=["Action"])

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8889)
