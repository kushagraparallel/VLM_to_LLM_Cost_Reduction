# Frontend

The comparison UI uses React, Vite, and TypeScript.

## Run locally

Use Node.js 22 LTS or another Vite-supported current Node release.

    cd frontend
    npm install
    cp .env.example .env
    npm run dev

Open http://127.0.0.1:5173. Start the FastAPI backend separately; its default address is http://127.0.0.1:8000.

Set VITE_API_BASE_URL if the API is hosted elsewhere. The backend's CORS_ORIGINS must include the frontend origin.

The UI lets you choose Gemini, Groq, or Bedrock independently for the direct VLM and structured text LLM paths. Provider credentials and model IDs must be configured in the backend environment. The UI sends the image, prompt, and provider choices to POST /api/v1/compare, then submits ratings/preference to POST /api/v1/feedback.
