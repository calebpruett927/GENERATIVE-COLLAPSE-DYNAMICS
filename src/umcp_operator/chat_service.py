# Operator chat service for GCD operator backend
import uuid

from .ce_service import compute_ce_audit
from .corpus_resolver import CORPUS_RESOLVER
from .session_store import SESSION_STORE
from .types import (
    CanonFiveWordView,
    CanonState,
    ChatRequest,
    ChatResponse,
    ContractBinding,
    CorpusRef,
    LedgerState,
    OperatorAnswer,
    SpineState,
    StanceState,
)


class ChatService:
    def chat(self, payload: ChatRequest) -> ChatResponse:
        # Resolve session
        session = SESSION_STORE.get(payload.sessionId)
        if session is None:
            raise KeyError(f"Session not found: {payload.sessionId}")

        # Resolve corpus objects in working set
        used_objects = []
        if payload.workingSet:
            for ref in payload.workingSet:
                try:
                    # ref is a CorpusRef, not a dict
                    obj = CORPUS_RESOLVER.get_object(ref.kind, ref.id).object
                    used_objects.append(CorpusRef(kind=obj.kind, id=obj.id, title=obj.title))
                except Exception:
                    continue

        # Compose a simple answer
        contract_id = session.activeContractId or "UMA.INTSTACK.v1"
        contract_title = f"Contract {contract_id}"
        working_titles = ", ".join([o.title or o.id for o in used_objects]) if used_objects else "none"
        plain_text = (
            f"Session {session.sessionId} is bound to contract '{contract_title}'. Working set: {working_titles}."
        )

        # Compose a minimal structured spine
        spine = SpineState(
            contract=ContractBinding(contractId=contract_id, source="repo", frozen=True),
            canon=CanonState(
                summary=f"Session for contract {contract_id}",
                fiveWordView=CanonFiveWordView(
                    drift="none", fidelity="high", roughness="low", return_="yes", integrity="good"
                ),
            ),
            closures=[],
            ledger=LedgerState(
                runId=None, receiptId=None, residual=0.0, deltaKappa=0.0, regime="STABLE", passesTolerance=True
            ),
            stance=StanceState(verdict="CONFORMANT", regime="STABLE", critical=False, rationale=["Session valid"]),
        )

        # Compute CE audit
        ce_audit = compute_ce_audit(plain_text)
        # Ensure ce_audit is a CEAudit instance, not a dict
        from .types import CEAudit

        if isinstance(ce_audit, dict):
            ce_audit = CEAudit(**ce_audit)

        msg_id = f"msg_{uuid.uuid4().hex[:8]}"
        return ChatResponse(
            messageId=msg_id,
            answer=OperatorAnswer(
                messageId=msg_id,
                plainText=plain_text,
                structured=spine,
                usedObjects=used_objects,
                ceAudit=ce_audit,
            ),
            usedObjects=used_objects,
            ceAudit=ce_audit,
        )


CHAT_SERVICE = ChatService()
