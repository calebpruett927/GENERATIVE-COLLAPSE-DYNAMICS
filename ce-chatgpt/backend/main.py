from ce_kernel import run_ce_audit
from fastapi import FastAPI  # type: ignore
from fastapi.middleware.cors import CORSMiddleware  # type: ignore
from gpt_client import call_gpt_api
from pydantic import BaseModel  # type: ignore

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):  # type: ignore[misc]
    message: str
    ce_mode: str  # 'silent', 'lightweight', 'full'


@app.post("/chat")  # type: ignore[misc]
async def chat_endpoint(req: ChatRequest) -> dict[str, object]:
    gpt_response = call_gpt_api(req.message)
    ce_audit = run_ce_audit(gpt_response, req.message, req.ce_mode)
    return {"answer": gpt_response, "ce_audit": ce_audit}


@app.get("/health")  # type: ignore[misc]
def health() -> dict[str, str]:
    return {"status": "ok"}
