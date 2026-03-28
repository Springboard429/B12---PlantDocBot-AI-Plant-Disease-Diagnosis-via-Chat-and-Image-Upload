import { useEffect, useMemo, useState } from 'react'
import type { FormEvent } from 'react'
import './App.css'

type Prediction = {
  disease: string
  confidence: number
  model?: string
}

type CombinedPredictionResponse = {
  image_prediction: Prediction | null
  text_prediction: Prediction | null
}

type HealthResponse = {
  status: string
  models_loaded: boolean
}

type ClassesResponse = {
  classes: string[]
}

type Mode = 'image' | 'text' | 'combined'

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL?.replace(/\/$/, '') ?? 'http://127.0.0.1:8000'

async function parseError(response: Response): Promise<string> {
  try {
    const payload = (await response.json()) as { detail?: string }
    if (payload.detail) {
      return payload.detail
    }
  } catch {
    return `Request failed with status ${response.status}.`
  }

  return `Request failed with status ${response.status}.`
}

function App() {
  const [mode, setMode] = useState<Mode>('combined')
  const [imageFile, setImageFile] = useState<File | null>(null)
  const [textInput, setTextInput] = useState('')

  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [classes, setClasses] = useState<string[]>([])
  const [loadingMeta, setLoadingMeta] = useState(false)
  const [loadingPrediction, setLoadingPrediction] = useState(false)

  const [imageResult, setImageResult] = useState<Prediction | null>(null)
  const [textResult, setTextResult] = useState<Prediction | null>(null)
  const [combinedResult, setCombinedResult] =
    useState<CombinedPredictionResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const controller = new AbortController()

    async function loadMeta(): Promise<void> {
      setLoadingMeta(true)
      setError(null)
      try {
        const [healthRes, classesRes] = await Promise.all([
          fetch(`${API_BASE_URL}/health`, { signal: controller.signal }),
          fetch(`${API_BASE_URL}/classes`, { signal: controller.signal }),
        ])

        if (!healthRes.ok) {
          throw new Error(await parseError(healthRes))
        }
        if (!classesRes.ok) {
          throw new Error(await parseError(classesRes))
        }

        const healthPayload = (await healthRes.json()) as HealthResponse
        const classesPayload = (await classesRes.json()) as ClassesResponse

        setHealth(healthPayload)
        setClasses(classesPayload.classes)
      } catch (loadError) {
        if (loadError instanceof DOMException && loadError.name === 'AbortError') {
          return
        }
        setError(
          loadError instanceof Error
            ? loadError.message
            : 'Failed to load API metadata.'
        )
      } finally {
        setLoadingMeta(false)
      }
    }

    void loadMeta()

    return () => {
      controller.abort()
    }
  }, [])

  const imagePreview = useMemo(() => {
    if (!imageFile) {
      return null
    }
    return URL.createObjectURL(imageFile)
  }, [imageFile])

  useEffect(() => {
    return () => {
      if (imagePreview) {
        URL.revokeObjectURL(imagePreview)
      }
    }
  }, [imagePreview])

  function resetResults(): void {
    setImageResult(null)
    setTextResult(null)
    setCombinedResult(null)
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault()
    setLoadingPrediction(true)
    setError(null)
    resetResults()

    try {
      if (mode === 'image') {
        if (!imageFile) {
          throw new Error('Please upload an image for image prediction.')
        }
        const body = new FormData()
        body.append('file', imageFile)

        const response = await fetch(`${API_BASE_URL}/predict/image`, {
          method: 'POST',
          body,
        })

        if (!response.ok) {
          throw new Error(await parseError(response))
        }
        setImageResult((await response.json()) as Prediction)
      }

      if (mode === 'text') {
        if (!textInput.trim()) {
          throw new Error('Please add a disease description for text prediction.')
        }
        const body = new FormData()
        body.append('text', textInput)

        const response = await fetch(`${API_BASE_URL}/predict/text`, {
          method: 'POST',
          body,
        })

        if (!response.ok) {
          throw new Error(await parseError(response))
        }
        setTextResult((await response.json()) as Prediction)
      }

      if (mode === 'combined') {
        if (!imageFile && !textInput.trim()) {
          throw new Error('Please provide an image, text, or both for combined prediction.')
        }

        const body = new FormData()
        if (imageFile) {
          body.append('file', imageFile)
        }
        if (textInput.trim()) {
          body.append('text', textInput)
        }

        const response = await fetch(`${API_BASE_URL}/predict/combined`, {
          method: 'POST',
          body,
        })

        if (!response.ok) {
          throw new Error(await parseError(response))
        }
        setCombinedResult((await response.json()) as CombinedPredictionResponse)
      }
    } catch (submitError) {
      setError(submitError instanceof Error ? submitError.message : 'Prediction failed.')
    } finally {
      setLoadingPrediction(false)
    }
  }

  function renderPredictionCard(label: string, prediction: Prediction | null) {
    if (!prediction) {
      return (
        <div className="prediction-card muted">
          <h3>{label}</h3>
          <p>Waiting for a prediction.</p>
        </div>
      )
    }

    return (
      <div className="prediction-card">
        <h3>{label}</h3>
        <p className="disease-name">{prediction.disease}</p>
        <p className="model-tag">Model: {prediction.model ?? 'inferred'}</p>
        <div
          className="confidence-meter"
          role="img"
          aria-label={`Confidence ${prediction.confidence.toFixed(2)} percent`}
        >
          <span
            style={{
              width: `${Math.max(0, Math.min(100, prediction.confidence))}%`,
            }}
          />
        </div>
        <p className="confidence-value">{prediction.confidence.toFixed(2)}%</p>
      </div>
    )
  }

  return (
    <div className="app-shell">
      <div className="ambient ambient-left" />
      <div className="ambient ambient-right" />

      <header className="hero-panel reveal delay-1">
        <p className="kicker">PlantDoc Intelligence Console</p>
        <h1>Diagnose Leaf Disease from Image, Text, or Both</h1>
        <p className="hero-copy">
          A React + Vite frontend for your FastAPI models. Run predictions,
          compare outputs, and inspect confidence in one place.
        </p>
        <div className="meta-strip">
          <span
            className={
              health?.models_loaded ? 'status-badge ready' : 'status-badge not-ready'
            }
          >
            {loadingMeta
              ? 'Checking models...'
              : health?.models_loaded
                ? 'Models ready'
                : 'Models unavailable'}
          </span>
          <span className="status-badge">API: {API_BASE_URL}</span>
        </div>
      </header>

      <main className="layout-grid">
        <section className="control-panel reveal delay-2">
          <div className="mode-toggle" role="tablist" aria-label="Prediction mode selector">
            {(['image', 'text', 'combined'] as const).map((option) => (
              <button
                key={option}
                type="button"
                role="tab"
                aria-selected={mode === option}
                className={mode === option ? 'mode active' : 'mode'}
                onClick={() => setMode(option)}
              >
                {option}
              </button>
            ))}
          </div>

          <form onSubmit={(event) => void handleSubmit(event)} className="predict-form">
            {(mode === 'image' || mode === 'combined') && (
              <label className="field">
                <span>Leaf image</span>
                <input
                  type="file"
                  accept="image/png,image/jpeg,image/jpg"
                  onChange={(event) => {
                    const selected = event.target.files?.[0] ?? null
                    setImageFile(selected)
                  }}
                />
              </label>
            )}

            {imagePreview && (
              <div className="preview-wrap">
                <img
                  src={imagePreview}
                  alt="Uploaded leaf preview"
                  className="preview-image"
                />
                <p>{imageFile?.name}</p>
              </div>
            )}

            {(mode === 'text' || mode === 'combined') && (
              <label className="field">
                <span>Symptoms or visual description</span>
                <textarea
                  placeholder="Example: Tomato leaves show dark circular spots with yellow halos."
                  value={textInput}
                  onChange={(event) => setTextInput(event.target.value)}
                  maxLength={512}
                />
                <small>{textInput.length}/512</small>
              </label>
            )}

            <div className="actions">
              <button type="submit" className="cta" disabled={loadingPrediction}>
                {loadingPrediction ? 'Running prediction...' : 'Predict disease'}
              </button>
              <button
                type="button"
                className="ghost"
                onClick={() => {
                  setImageFile(null)
                  setTextInput('')
                  resetResults()
                  setError(null)
                }}
                disabled={loadingPrediction}
              >
                Reset
              </button>
            </div>
          </form>
        </section>

        <section className="results-panel reveal delay-3">
          {error && <p className="error-banner">{error}</p>}

          {mode === 'image' && renderPredictionCard('Image prediction', imageResult)}
          {mode === 'text' && renderPredictionCard('Text prediction', textResult)}
          {mode === 'combined' && (
            <div className="combined-grid">
              {renderPredictionCard('Image branch', combinedResult?.image_prediction ?? null)}
              {renderPredictionCard('Text branch', combinedResult?.text_prediction ?? null)}
            </div>
          )}

          <article className="class-list">
            <div className="class-list-header">
              <h2>Known disease classes</h2>
              <span>{classes.length}</span>
            </div>
            <div className="chip-wrap">
              {classes.map((item) => (
                <span key={item} className="chip">
                  {item}
                </span>
              ))}
            </div>
          </article>
        </section>
      </main>
    </div>
  )
}

export default App
