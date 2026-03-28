# PlantDoc Frontend

React + Vite + TypeScript interface for the PlantDoc FastAPI backend.

## Features

- Image disease prediction (`POST /predict/image`)
- Text disease prediction (`POST /predict/text`)
- Combined image + text prediction (`POST /predict/combined`)
- API metadata display from `GET /health` and `GET /classes`
- Responsive UI with confidence visualization

## Prerequisites

- Node.js 20+
- Running backend API (default: `http://127.0.0.1:8000`)

## Setup

1. Install dependencies:

```bash
npm install
```

2. Configure API URL:

```bash
cp .env.example .env
```

3. Start development server:

```bash
npm run dev
```

4. Build production bundle:

```bash
npm run build
```

## Environment Variables

- `VITE_API_BASE_URL`: Base URL for the backend API.
