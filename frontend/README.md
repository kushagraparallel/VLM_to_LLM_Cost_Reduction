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

The UI sends the image and prompt to POST /api/v1/compare, displays both outputs and available metrics, and submits ratings/preference to POST /api/v1/feedback.
