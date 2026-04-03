from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from .sre_environment import SREEnvironment
from typing import Optional

app = FastAPI(
    title="SRE Incident Response Environment",
    description="OpenEnv: AI agent learns to fix production incidents",
    version="1.0.0"
)

app.add_middleware(CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

env = SREEnvironment()


class ResetRequest(BaseModel):
    task_id: str = "easy"


class StepRequest(BaseModel):
    action_type: str
    target: str
    reasoning: str = ""


@app.post("/reset")
def reset(request: Optional[ResetRequest] = None):
    task_id = request.task_id if request else "easy"
    return env.reset(task_id=task_id)

@app.post("/step")
def step(request: StepRequest):
    return env.step(
        action_type=request.action_type,
        target=request.target,
        reasoning=request.reasoning
    )


@app.get("/state")
def state():
    return env.state()


@app.get("/")
def root():
    return {"status": "running",
            "environment": "SRE Incident Response",
            "version": "1.0.0"}
def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()