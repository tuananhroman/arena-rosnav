from pydantic import BaseModel


class TaskCfg(BaseModel):
    tm_robots: str = "random"
    tm_obstacles: str = "random"
    tm_modules: str = "staged"
