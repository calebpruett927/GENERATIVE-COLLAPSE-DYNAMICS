# Cognitive Equalizer ChatGPT — Deployment Instructions

## 1. Backend (FastAPI)
- Location: `ce-chatgpt/backend/`
- Install dependencies: `pip install -r requirements.txt`
- Set your OpenAI API key: `export OPENAI_API_KEY=sk-...`
- Start server: `uvicorn main:app --reload`
- API runs at: `http://localhost:8000`

## 2. Frontend (React + Vite)
- Location: `ce-chatgpt/frontend/`
- Install dependencies: `npm install`
- Start dev server: `npm run dev`
- App runs at: `http://localhost:5173`

## 3. Usage
- Open the frontend in your browser.
- Enter a message, select CE audit mode, and send.
- View GPT-4 answer and Cognitive Equalizer audit.

## 4. Configuration
- To change GPT model: edit `model` in `backend/gpt_client.py`.
- To improve CE scoring: edit `auto_score_channels` in `backend/ce_kernel.py`.

## 5. Troubleshooting
- If you see `[ERROR] OPENAI_API_KEY not set.`, ensure your API key is exported in the backend environment.
- If frontend cannot connect, check CORS settings and that both servers are running.

## 6. Production
- For production, use a process manager (e.g., gunicorn, pm2) and serve the frontend as static files.
- Secure your API key and endpoints.

---
For more details, see `README.md` in the project root.
