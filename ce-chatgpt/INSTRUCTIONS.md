# Cognitive Equalizer ChatGPT — Instruction Set

## Purpose
This GPT is a public-facing ChatGPT with always-on Cognitive Equalizer (CE) auditing. Every user message is processed by GPT-4 and receives a CE audit (8-channel scoring, regime, and five-word summary) in the response.

## How to Use
1. Enter your message in the chat box.
2. Select the desired CE audit mode:
   - **silent**: No visible audit, only the answer.
   - **lightweight**: Five-word summary appended to the answer.
   - **full**: Full audit (CONTRACT → CANON → CLOSURES → LEDGER → STANCE) with channel scores and regime.
3. Submit your message. The response will include both the GPT answer and the CE audit (if enabled).

## API Endpoints
- **POST /chat**: Send a chat message and receive a GPT answer plus CE audit.
  - Request body:
    - `message` (string): Your message to GPT.
    - `ce_mode` (string): "silent", "lightweight", or "full".
  - Response:
    - `answer` (string): GPT's reply.
    - `ce_audit` (object): CE audit result (structure depends on mode).
- **GET /health**: Health check endpoint. Returns `{ "status": "ok" }` if the backend is running.

## Requirements
- The backend must be running (FastAPI, Uvicorn, port 8000).
- `OPENAI_API_KEY` must be set in the environment for GPT calls to succeed.

## Cognitive Equalizer Audit
- 8 channels: relevance, accuracy, completeness, consistency, traceability, groundedness, constraint-respect, return-fidelity.
- Regimes: CONFORMANT / NONCONFORMANT / NON_EVALUABLE.
- Five-word summary: Drift, Fidelity, Roughness, Return, Integrity.
- Full audit: CONTRACT → CANON → CLOSURES → LEDGER → STANCE.

## Troubleshooting
- If you get a 5xx error, check that the backend is running and the API key is set.
- Use GET /health to verify backend status.
- For OpenAPI schema, see `openapi.json`.

## References
- [README.md](../README.md)
- [openapi.json](openapi.json)
- [actions.schema.json](actions.schema.json)
