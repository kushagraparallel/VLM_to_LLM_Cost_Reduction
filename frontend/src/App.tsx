import { useEffect, useMemo, useState, type FormEvent } from "react";

type ModelResult = {
  provider?: string;
  model?: string;
  text?: string | null;
  answer?: string | null;
  latency_seconds?: number | null;
  path_seconds?: number | null;
  input_tokens?: number | null;
  output_tokens?: number | null;
  estimated_cost_usd?: number | null;
  cost_estimate_usd?: number | null;
  cost_scope?: string;
  error?: string;
};

type ComparisonResult = {
  run_id: string;
  user_prompt: string;
  direct_vlm: ModelResult;
  structured_pipeline: ModelResult & {
    provider_result?: ModelResult;
    local_processing_seconds?: number;
    ocr_seconds?: number;
    detection_seconds?: number;
    representation?: {
      ocr_regions?: Array<{ text: string; confidence?: number | null }>;
      detections?: Array<{ label: string; confidence: number; bbox_xyxy: number[] }>;
    };
  };
  comparison: {
    direct_vlm_seconds?: number | null;
    structured_pipeline_seconds?: number | null;
    direct_vlm_cost_usd?: number | null;
    structured_pipeline_cost_usd?: number | null;
  };
};

type PreferredPath = "direct_vlm" | "structured_pipeline" | "tie";

const API_BASE = (import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000").replace(/\/$/, "");

function formatSeconds(value?: number | null): string {
  return value == null ? "Not reported" : value.toFixed(2) + " s";
}

function formatCost(value?: number | null): string {
  return value == null ? "Unavailable" : "$" + value.toFixed(6);
}

function formatTokens(value?: number | null): string {
  return value == null ? "Not reported" : value.toLocaleString();
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function ResultCard({
  title,
  subtitle,
  answer,
  provider,
  model,
  elapsed,
  result,
}: {
  title: string;
  subtitle: string;
  answer?: string | null;
  provider?: string;
  model?: string;
  elapsed?: number | null;
  result: ModelResult;
}) {
  return (
    <article className="result-card">
      <div className="result-heading">
        <div>
          <p className="eyebrow">{subtitle}</p>
          <h2>{title}</h2>
        </div>
        <span className="model-chip">{provider || "Provider"} · {model || "Model"}</span>
      </div>
      {result.error ? (
        <div className="error-panel">
          <strong>This path failed</strong>
          <p>{result.error}</p>
        </div>
      ) : (
        <div className="answer">{answer || "The provider returned an empty response."}</div>
      )}
      <div className="metrics-grid">
        <Metric label="Path latency" value={formatSeconds(elapsed)} />
        <Metric label="Estimated API cost" value={formatCost(result.estimated_cost_usd ?? result.cost_estimate_usd)} />
        <Metric label="Input tokens" value={formatTokens(result.input_tokens)} />
        <Metric label="Output tokens" value={formatTokens(result.output_tokens)} />
      </div>
      {result.cost_scope && <p className="metric-note">{result.cost_scope}</p>}
    </article>
  );
}

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [prompt, setPrompt] = useState("Summarize this image and identify its key information.");
  const [result, setResult] = useState<ComparisonResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [feedbackBusy, setFeedbackBusy] = useState(false);
  const [error, setError] = useState("");
  const [feedbackMessage, setFeedbackMessage] = useState("");
  const [preferred, setPreferred] = useState<PreferredPath>("tie");
  const [directRating, setDirectRating] = useState(0);
  const [structuredRating, setStructuredRating] = useState(0);
  const [comment, setComment] = useState("");

  const previewUrl = useMemo(() => (file ? URL.createObjectURL(file) : ""), [file]);
  useEffect(() => () => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
  }, [previewUrl]);

  async function runComparison(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!file || !prompt.trim()) {
      setError("Choose an image and enter a prompt to continue.");
      return;
    }

    setBusy(true);
    setError("");
    setFeedbackMessage("");
    setResult(null);
    try {
      const body = new FormData();
      body.append("file", file);
      body.append("prompt", prompt);
      const response = await fetch(API_BASE + "/api/v1/compare", {
        method: "POST",
        body,
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "The comparison request failed.");
      }
      setResult(data as ComparisonResult);
      setPreferred("tie");
      setDirectRating(0);
      setStructuredRating(0);
      setComment("");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to reach the API.");
    } finally {
      setBusy(false);
    }
  }

  async function submitFeedback() {
    if (!result) return;
    setFeedbackBusy(true);
    setFeedbackMessage("");
    setError("");
    try {
      const response = await fetch(API_BASE + "/api/v1/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          run_id: result.run_id,
          preferred_path: preferred,
          direct_vlm_rating: directRating || null,
          structured_pipeline_rating: structuredRating || null,
          comment: comment.trim() || null,
        }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Feedback could not be saved.");
      setFeedbackMessage("Thanks — your feedback was saved for this experiment.");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to save feedback.");
    } finally {
      setFeedbackBusy(false);
    }
  }

  return (
    <main className="page-shell">
      <header className="topbar">
        <a className="brand" href="/">
          <span className="brand-mark">V</span>
          <span>Vision Cost Lab</span>
        </a>
        <span className="status-pill"><span className="status-dot" /> Experiment workspace</span>
      </header>

      <section className="hero">
        <p className="eyebrow">IMAGE UNDERSTANDING · COST / QUALITY STUDY</p>
        <h1>Can structured vision deliver<br className="desktop-break" /> the same insight for less?</h1>
        <p className="hero-copy">
          Compare a direct vision-language model against a pipeline that extracts text and objects locally,
          then sends structured evidence to a text model.
        </p>
      </section>

      <section className="input-panel">
        <div className="section-title">
          <div>
            <p className="eyebrow">01 / INPUT</p>
            <h2>Set up your experiment</h2>
          </div>
          <span className="small-note">Same user prompt for both paths</span>
        </div>
        <form onSubmit={runComparison}>
          <div className="upload-layout">
            <label className={"upload-zone " + (file ? "has-file" : "")}>
              {previewUrl ? (
                <img className="image-preview" src={previewUrl} alt="Selected upload preview" />
              ) : (
                <span className="upload-icon">↑</span>
              )}
              <strong>{file ? file.name : "Choose an image"}</strong>
              <span>{file ? (file.size / 1024 / 1024).toFixed(2) + " MB · Click to replace" : "PNG, JPG, WEBP or another supported image"}</span>
              <input
                type="file"
                accept="image/*"
                onChange={(event) => {
                  setFile(event.target.files?.[0] || null);
                  setError("");
                }}
              />
            </label>
            <div className="prompt-field">
              <label htmlFor="prompt">Your prompt</label>
              <textarea
                id="prompt"
                value={prompt}
                maxLength={8000}
                onChange={(event) => setPrompt(event.target.value)}
                placeholder="What should the models tell you about this image?"
                rows={6}
              />
              <div className="field-footer">
                <span>This exact prompt is sent to both paths.</span>
                <span>{prompt.length}/8000</span>
              </div>
              <button className="primary-button" type="submit" disabled={busy || !file || !prompt.trim()}>
                {busy ? <><span className="spinner" /> Running comparison…</> : "Run comparison"}
              </button>
            </div>
          </div>
        </form>
        {error && <div className="alert error-alert" role="alert">{error}</div>}
      </section>

      {busy && (
        <section className="loading-panel">
          <span className="spinner dark-spinner" />
          <div><strong>Running both approaches</strong><p>Local OCR and detection may take longer on the first run while models load.</p></div>
        </section>
      )}

      {result && (
        <>
          <section className="results-section">
            <div className="section-title">
              <div>
                <p className="eyebrow">02 / RESULTS</p>
                <h2>Compare the outputs</h2>
              </div>
              <span className="run-id">Run {result.run_id.slice(0, 8)}</span>
            </div>
            <div className="results-grid">
              <ResultCard
                title="Direct VLM"
                subtitle="BASELINE · IMAGE → VISION MODEL"
                answer={result.direct_vlm.text}
                provider={result.direct_vlm.provider}
                model={result.direct_vlm.model}
                elapsed={result.comparison.direct_vlm_seconds}
                result={result.direct_vlm}
              />
              <ResultCard
                title="Structured pipeline"
                subtitle="OCR + YOLO26 / OPENVINO → TEXT MODEL"
                answer={result.structured_pipeline.answer}
                provider={result.structured_pipeline.provider}
                model={result.structured_pipeline.model}
                elapsed={result.comparison.structured_pipeline_seconds}
                result={result.structured_pipeline.provider_result || result.structured_pipeline}
              />
            </div>
            <p className="footnote">
              Costs are estimates only and appear when provider token usage and a verified model-pricing entry are available.
              Local OCR/detection compute cost is not included.
            </p>
          </section>

          {result.structured_pipeline.representation && (
            <details className="evidence-panel">
              <summary>
                <span><strong>Inspect structured image evidence</strong><small>OCR text and object detections sent to the text model</small></span>
                <span className="summary-count">
                  {(result.structured_pipeline.representation.ocr_regions?.length || 0) + " text regions · " + (result.structured_pipeline.representation.detections?.length || 0) + " objects"}
                </span>
              </summary>
              <div className="evidence-content">
                <h3>Recognized text</h3>
                {(result.structured_pipeline.representation.ocr_regions || []).length ? (
                  <ul>{result.structured_pipeline.representation.ocr_regions?.map((item, index) => (
                    <li key={index + "-" + item.text}><span>{item.text}</span><small>{item.confidence == null ? "Confidence unavailable" : "Confidence " + (item.confidence * 100).toFixed(0) + "%"}</small></li>
                  ))}</ul>
                ) : <p>No text regions were detected.</p>}
                <h3>Detected objects</h3>
                {(result.structured_pipeline.representation.detections || []).length ? (
                  <ul>{result.structured_pipeline.representation.detections?.map((item, index) => (
                    <li key={index + "-" + item.label}><span>{item.label}</span><small>{(item.confidence * 100).toFixed(0) + "% confidence"}</small></li>
                  ))}</ul>
                ) : <p>No objects were detected.</p>}
              </div>
            </details>
          )}

          <section className="feedback-panel">
            <div className="section-title">
              <div>
                <p className="eyebrow">03 / HUMAN EVALUATION</p>
                <h2>Which result worked better?</h2>
              </div>
            </div>
            <div className="preference-options">
              {([
                ["direct_vlm", "Direct VLM", "The original image produced a better answer."],
                ["structured_pipeline", "Structured pipeline", "OCR and object evidence were enough."],
                ["tie", "About the same", "Neither answer was clearly better."],
              ] as const).map(([value, label, description]) => (
                <button
                  type="button"
                  key={value}
                  className={"preference-option " + (preferred === value ? "selected" : "")}
                  onClick={() => setPreferred(value)}
                  aria-pressed={preferred === value}
                >
                  <span className="radio-indicator" />
                  <span><strong>{label}</strong><small>{description}</small></span>
                </button>
              ))}
            </div>
            <div className="rating-grid">
              <div className="rating-block">
                <label>Direct VLM rating</label>
                <div className="rating-buttons" role="group" aria-label="Direct VLM rating from 1 to 5">
                  {[1, 2, 3, 4, 5].map((rating) => (
                    <button key={rating} type="button" className={directRating === rating ? "active" : ""} onClick={() => setDirectRating(rating)}>{rating}</button>
                  ))}
                </div>
                <small>1 = poor · 5 = excellent</small>
              </div>
              <div className="rating-block">
                <label>Structured pipeline rating</label>
                <div className="rating-buttons" role="group" aria-label="Structured pipeline rating from 1 to 5">
                  {[1, 2, 3, 4, 5].map((rating) => (
                    <button key={rating} type="button" className={structuredRating === rating ? "active" : ""} onClick={() => setStructuredRating(rating)}>{rating}</button>
                  ))}
                </div>
                <small>1 = poor · 5 = excellent</small>
              </div>
            </div>
            <label className="comment-label" htmlFor="comment">Optional notes</label>
            <textarea id="comment" rows={3} maxLength={2000} value={comment} onChange={(event) => setComment(event.target.value)} placeholder="What was missing, incorrect, or especially useful?" />
            <div className="feedback-footer">
              <span>Feedback is linked to this experiment run.</span>
              <button className="primary-button" type="button" onClick={submitFeedback} disabled={feedbackBusy}>
                {feedbackBusy ? "Saving…" : "Save evaluation"}
              </button>
            </div>
            {feedbackMessage && <div className="alert success-alert" role="status">{feedbackMessage}</div>}
          </section>
        </>
      )}

      <footer className="footer">
        <span>Vision Cost Lab · Research prototype</span>
        <span>API usage and cost depend on configured providers and model pricing.</span>
      </footer>
    </main>
  );
}
