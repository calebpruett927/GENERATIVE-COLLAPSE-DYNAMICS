from __future__ import annotations

from fastapi import APIRouter, HTTPException

from .casepack_service import CASEPACK_SERVICE
from .chat_service import CHAT_SERVICE
from .corpus_resolver import CORPUS_RESOLVER
from .session_store import SESSION_STORE
from .types import (
    BindContractRequest,
    BindContractResponse,
    ChatRequest,
    ChatResponse,
    ContractBinding,
    CorpusObjectResponse,
    CorpusSearchRequest,
    CorpusSearchResponse,
    CreateSessionRequest,
    OperatorSession,
    RunRecord,
    SessionEnvelope,
    SpineState,
    StartCasepackRunRequest,
    StartCasepackRunResponse,
)

router = APIRouter(prefix="/api/v1", tags=["operator"])


@router.post("/operator/session", response_model=OperatorSession)
def create_operator_session(payload: CreateSessionRequest) -> OperatorSession:
    return SESSION_STORE.create(payload)


@router.get("/operator/session/{session_id}", response_model=SessionEnvelope)
def get_operator_session(session_id: str) -> SessionEnvelope:
    session = SESSION_STORE.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    # For now, return a minimal spine with contract only
    spine = SpineState(
        contract=ContractBinding(
            contractId=session.activeContractId or "UMA.INTSTACK.v1",
            source="repo",
            frozen=True,
        ),
        closures=[],
    )
    return SessionEnvelope(session=session, spineState=spine)


@router.post("/corpus/search", response_model=CorpusSearchResponse)
def search_corpus(payload: CorpusSearchRequest) -> CorpusSearchResponse:
    return CORPUS_RESOLVER.search(payload)


@router.get("/corpus/object/{kind}/{object_id}", response_model=CorpusObjectResponse)
def get_corpus_object(kind: str, object_id: str) -> CorpusObjectResponse:
    try:
        return CORPUS_RESOLVER.get_object(kind, object_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/operator/chat", response_model=ChatResponse)
def operator_chat(payload: ChatRequest) -> ChatResponse:
    try:
        return CHAT_SERVICE.chat(payload)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/operator/bind-contract", response_model=BindContractResponse)
def bind_contract(payload: BindContractRequest) -> BindContractResponse:
    session = SESSION_STORE.get(payload.sessionId)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    updated = SESSION_STORE.update_contract(payload.sessionId, payload.contractId)
    spine = SpineState(
        contract=ContractBinding(
            contractId=updated.activeContractId or payload.contractId,
            source="repo",
            frozen=True,
        ),
        closures=[],
    )
    return BindContractResponse(
        sessionId=updated.sessionId,
        activeContractId=payload.contractId,
        spineState=spine,
    )


@router.post("/operator/casepacks/{casepack_id}/run", response_model=StartCasepackRunResponse)
def start_casepack_run(casepack_id: str, payload: StartCasepackRunRequest) -> StartCasepackRunResponse:
    try:
        return CASEPACK_SERVICE.start_run(casepack_id, payload)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/runs/{run_id}", response_model=RunRecord)
def get_run(run_id: str) -> RunRecord:
    try:
        return CASEPACK_SERVICE.get_run(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    # TODO: Parse payload and return operator answer
    return {}
