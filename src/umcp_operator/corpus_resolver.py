# Corpus resolver for GCD operator backend

from .types import CorpusObject, CorpusObjectKind, CorpusObjectResponse, CorpusSearchRequest, CorpusSearchResponse


class CorpusResolver:
    def search(self, payload: CorpusSearchRequest) -> CorpusSearchResponse:
        try:
            query = payload.query.lower() if getattr(payload, "query", None) else ""
            results: list[CorpusObject] = []
            for kind in ["contracts", "casepacks", "papers"]:
                for obj in CORPUS.get(kind, []):
                    # Ensure title is str
                    title = str(obj.get("title", ""))
                    obj_id = str(obj.get("id", ""))
                    if query in title.lower() or query in obj_id.lower():
                        # Build CorpusObject
                        kind_typed: CorpusObjectKind = kind  # type: ignore
                        results.append(CorpusObject(kind=kind_typed, id=obj_id, title=title))
            return CorpusSearchResponse(results=results)
        except Exception:
            return CorpusSearchResponse(results=[])

    def get_object(self, kind: str, object_id: str) -> CorpusObjectResponse:
        try:
            for obj in CORPUS.get(kind, []):
                if obj["id"] == object_id:
                    kind_typed: CorpusObjectKind = kind  # type: ignore
                    title = str(obj.get("title", ""))
                    return CorpusObjectResponse(object=CorpusObject(kind=kind_typed, id=object_id, title=title))
            # Not found
            kind_typed: CorpusObjectKind = kind  # type: ignore
            return CorpusObjectResponse(object=CorpusObject(kind=kind_typed, id=object_id, title="[Not found]"))
        except Exception:
            kind_typed: CorpusObjectKind = kind  # type: ignore
            return CorpusObjectResponse(object=CorpusObject(kind=kind_typed, id=object_id, title="[Error]"))


CORPUS = {
    "contracts": [
        {
            "id": "UMA.INTSTACK.v1",
            "title": "Universal Measurement Contract (IntStack v1)",
            "type": "contract",
            "description": "Frozen contract for universal measurement, used in most casepacks.",
            "casepacks": ["hello_world", "atomic_physics_elements"],
        },
        {
            "id": "GCD.CORE.v1",
            "title": "Generative Collapse Dynamics Core Contract",
            "type": "contract",
            "description": "Core contract for GCD kernel validation.",
            "casepacks": ["atomic_physics_elements"],
        },
    ],
    "casepacks": [
        {
            "id": "hello_world",
            "title": "Hello World Casepack",
            "type": "casepack",
            "description": "Minimal casepack for orientation and testing.",
            "contract": "UMA.INTSTACK.v1",
            "papers": [],
        },
        {
            "id": "atomic_physics_elements",
            "title": "Atomic Physics Elements",
            "type": "casepack",
            "description": "Casepack for atomic physics kernel, covers all 118 elements.",
            "contract": "GCD.CORE.v1",
            "papers": ["standard_model_kernel"],
        },
    ],
    "papers": [
        {
            "id": "standard_model_kernel",
            "title": "Particle Physics in the GCD Kernel",
            "type": "paper",
            "description": "Published paper on Standard Model theorems in the GCD kernel.",
            "casepacks": ["atomic_physics_elements"],
        },
        {
            "id": "tau_r_star_dynamics",
            "title": "τ_R* Dynamics Paper",
            "type": "paper",
            "description": "Paper on τ_R* thermodynamic diagnostic and phase diagram.",
            "casepacks": [],
        },
    ],
}


# Only one CorpusResolver definition and CORPUS_RESOLVER instance should exist
CORPUS_RESOLVER = CorpusResolver()
