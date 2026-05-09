# Cognitive Equalizer ChatGPT (CE-ChatGPT)

A public-facing conversational AI app that combines OpenAI's GPT-4 with the Cognitive Equalizer (CE) audit protocol. Every response is scored across 8 structural channels, with regime classification and audit summaries, providing transparent, contract-bound reasoning.

## Features
- Chat with GPT-4 and receive answers with a Cognitive Equalizer audit.
- CE audit scores 8 channels: relevance, accuracy, completeness, consistency, traceability, groundedness, constraint-respect, return-fidelity.
- Three audit modes: Silent, Lightweight (five-word summary), Full Spine (full audit trail).
- Regime classification: CONFORMANT / NONCONFORMANT / NON_EVALUABLE.
- Simple, modern web UI (React + Vite).

## Getting Started

### Prerequisites
- Python 3.9+
- Node.js 18+
- OpenAI API key (for GPT-4 access)

### Backend Setup
1. `cd ce-chatgpt/backend`
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set your OpenAI API key:
   ```bash
   export OPENAI_API_KEY=sk-...  # Replace with your key
   ```
4. Start the backend server:
   ```bash
   uvicorn main:app --reload
   ```
   The API will be available at `http://localhost:8000`.

### Frontend Setup
1. `cd ce-chatgpt/frontend`
2. Install dependencies:
   ```bash
   npm install
   ```
3. Start the frontend dev server:
   ```bash
   npm run dev
   ```
   The app will be available at `http://localhost:5173` (default Vite port).

### Usage
- Enter your message in the chat box.
- Select the desired CE audit mode (Silent, Lightweight, Full Spine).
- Click "Send" to receive a GPT-4 answer and a CE audit.
- The audit includes channel scores, regime, and (in full mode) a detailed spine report.

## File Structure
- `backend/` — FastAPI server, CE kernel, GPT client
- `frontend/` — React app, chat UI, audit display

## Customization
- To improve CE channel scoring, edit `backend/ce_kernel.py` (`auto_score_channels` function).
- To use a different GPT model, change the `model` parameter in `backend/gpt_client.py`.

## License
MIT

## Credits
- Cognitive Equalizer (Aequator Cognitivus) — GCD/UMCP project
- OpenAI GPT-4
