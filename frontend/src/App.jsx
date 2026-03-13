import { useEffect, useMemo, useRef, useState } from "react";

export default function App() {
  const apiBase = useMemo(
    () => (import.meta.env.VITE_SEARCH_API_BASE || "http://localhost:8000").replace(/\/$/, ""),
    []
  );
  const quickQueries = useMemo(
    () => [
      "a dog playing in the park",
      "sunset over the ocean",
      "city street at night with neon lights",
      "person running in a park",
      "sunset over the sea with silhouettes",
      "dog playing in the snow"
    ],
    []
  );
  const [query, setQuery] = useState("");
  const [debouncedQuery, setDebouncedQuery] = useState("");
  const [results, setResults] = useState([]);
  const [status, setStatus] = useState("idle");
  const [error, setError] = useState(null);
  const [serverReady, setServerReady] = useState(null);
  const [progress, setProgress] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [statusInfo, setStatusInfo] = useState(null);
  const [latencyMs, setLatencyMs] = useState(null);
  const [ollamaModels, setOllamaModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState("");
  const [ollamaError, setOllamaError] = useState(null);
  const storedModelRef = useRef(null);
  const [explanation, setExplanation] = useState("");
  const [explainStatus, setExplainStatus] = useState("idle");
  const [explainError, setExplainError] = useState(null);
  const [explainMeta, setExplainMeta] = useState(null);
  const explainAbortRef = useRef(null);
  const [topK, setTopK] = useState(5);
  const [rerank, setRerank] = useState(true);
  const [rerankAlpha, setRerankAlpha] = useState(0.25);
  const [modality, setModality] = useState("image");
  const [settingsInitialized, setSettingsInitialized] = useState(false);
  const [logs, setLogs] = useState([]);
  const [metricsSummary, setMetricsSummary] = useState(null);
  const [benchmarks, setBenchmarks] = useState({ run_1000: null, run_5000: null });
  const [selectedResult, setSelectedResult] = useState(null);
  const [showResearch, setShowResearch] = useState(true);

  const requestExplanation = () => {
    if (!debouncedQuery || results.length === 0 || explainStatus === "loading") return;
    if (explainAbortRef.current) {
      explainAbortRef.current.abort();
    }

    const controller = new AbortController();
    explainAbortRef.current = controller;
    setExplainStatus("loading");
    setExplainError(null);
    setExplainMeta(null);

    const payload = {
      query: debouncedQuery,
      model: selectedModel || undefined,
      results: results.slice(0, 5).map((item) => ({
        file_name: item.file_name,
        score: item.score,
        caption: item.caption || null,
        image_url: item.image_url
      }))
    };

    fetch(`${apiBase}/explain`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: controller.signal
    })
      .then((res) => {
        if (!res.ok) {
          throw new Error(`Explain failed (${res.status})`);
        }
        return res.json();
      })
      .then((data) => {
        if (data.error) {
          throw new Error(data.error);
        }
        const message =
          data.explanation || data.answer || data.summary || data.message || "";
        if (!message) {
          throw new Error("The Ollama response did not include an explanation.");
        }
        setExplanation(message);
        setExplainStatus("ready");
        setExplainMeta({
          model: data.model || data.ollama_model || null,
          duration_ms: data.duration_ms || data.latency_ms || null,
          prompt_tokens: data.prompt_tokens || null,
          completion_tokens: data.completion_tokens || null
        });
      })
      .catch((err) => {
        if (err.name === "AbortError") return;
        setExplainError(err.message);
        setExplainStatus("error");
      });
  };

  useEffect(() => {
    const timeout = setTimeout(() => {
      setDebouncedQuery(query.trim());
    }, 350);
    return () => clearTimeout(timeout);
  }, [query]);

  useEffect(() => {
    let active = true;
    fetch(`${apiBase}/status`)
      .then((res) => res.json())
      .then((data) => {
        if (active) {
          setServerReady(data.ready);
          setProgress(data.progress || null);
          setMetrics(data.metrics || null);
          setStatusInfo(data);
          if (!settingsInitialized) {
            if (typeof data.rerank === "boolean") {
              setRerank(data.rerank);
            }
            if (typeof data.rerank_alpha_default === "number") {
              setRerankAlpha(data.rerank_alpha_default);
            }
            setSettingsInitialized(true);
          }
        }
      })
      .catch(() => {
        if (active) {
          setServerReady(false);
        }
      });
    return () => {
      active = false;
    };
  }, [apiBase]);

  useEffect(() => {
    let active = true;
    fetch(`${apiBase}/benchmarks`)
      .then((res) => res.json())
      .then((data) => {
        if (!active) return;
        setBenchmarks({
          run_1000: data.run_1000 || null,
          run_5000: data.run_5000 || null
        });
      })
      .catch(() => {
        if (!active) return;
        setBenchmarks({ run_1000: null, run_5000: null });
      });
    return () => {
      active = false;
    };
  }, [apiBase]);

  useEffect(() => {
    const source = new EventSource(`${apiBase}/events`);
    source.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        setProgress(data);
        setServerReady(data.ready);
      } catch {
        // Ignore malformed events.
      }
    };
    source.onerror = () => {
      source.close();
    };
    return () => {
      source.close();
    };
  }, [apiBase]);

  useEffect(() => {
    const source = new EventSource(`${apiBase}/logs`);
    source.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (!data.line) return;
        setLogs((prev) => {
          const next = [...prev, data.line];
          if (next.length > 200) {
            return next.slice(next.length - 200);
          }
          return next;
        });
      } catch {
        // Ignore malformed log events.
      }
    };
    source.onerror = () => {
      source.close();
    };
    return () => {
      source.close();
    };
  }, [apiBase]);

  useEffect(() => {
    let active = true;
    const fetchSummary = () =>
      fetch(`${apiBase}/metrics/summary`)
        .then((res) => res.json())
        .then((data) => {
          if (!active) return;
          setMetricsSummary(data);
        })
        .catch(() => {
          if (!active) return;
          setMetricsSummary(null);
        });

    fetchSummary();
    const interval = setInterval(fetchSummary, 8000);
    return () => {
      active = false;
      clearInterval(interval);
    };
  }, [apiBase]);

  useEffect(() => {
    let active = true;
    const stored = window.localStorage.getItem("mir_ollama_model");
    storedModelRef.current = stored;

    fetch(`${apiBase}/ollama/models`)
      .then((res) => res.json())
      .then((data) => {
        if (!active) return;
        if (data.error) {
          setOllamaError(data.error);
          return;
        }
        const models = Array.isArray(data.models) ? data.models : [];
        setOllamaModels(models);
        const storedModel =
          storedModelRef.current && models.includes(storedModelRef.current)
            ? storedModelRef.current
            : null;
        const defaultModel =
          data.default && models.includes(data.default) ? data.default : models[0];
        setSelectedModel(storedModel || defaultModel || "");
      })
      .catch((err) => {
        if (!active) return;
        setOllamaError(err.message);
      });
    return () => {
      active = false;
    };
  }, [apiBase]);

  useEffect(() => {
    if (!selectedModel) return;
    window.localStorage.setItem("mir_ollama_model", selectedModel);
  }, [selectedModel]);

  useEffect(() => {
    if (!debouncedQuery) {
      setResults([]);
      setStatus("idle");
      setError(null);
      setExplanation("");
      setExplainStatus("idle");
      setExplainError(null);
      setExplainMeta(null);
      setLatencyMs(null);
      return;
    }

    let active = true;
    setStatus("loading");
    setError(null);
    setExplanation("");
    setExplainStatus("idle");
    setExplainError(null);
    setExplainMeta(null);
    setLatencyMs(null);
    setSelectedResult(null);

    const params = new URLSearchParams({
      query: debouncedQuery,
      top_k: String(topK),
      rerank: rerank ? "true" : "false",
      rerank_alpha: String(rerankAlpha),
      modality
    });
    fetch(`${apiBase}/search?${params.toString()}`)
      .then((res) => {
        if (!res.ok) {
          throw new Error(`Search failed (${res.status})`);
        }
        return res.json();
      })
      .then((data) => {
        if (!active) return;
        if (data.error) {
          setError(data.error);
          setResults([]);
          setStatus("error");
          return;
        }
        setResults(data.results || []);
        setLatencyMs(typeof data.latency_ms === "number" ? data.latency_ms : null);
        setStatus("ready");
      })
      .catch((err) => {
        if (!active) return;
        setError(err.message);
        setResults([]);
        setStatus("error");
      });

    return () => {
      active = false;
    };
  }, [apiBase, debouncedQuery, modality, rerank, rerankAlpha, topK]);

  useEffect(() => {
    if (!debouncedQuery) return;
    if (explainAbortRef.current) {
      explainAbortRef.current.abort();
    }
  }, [debouncedQuery]);

  useEffect(() => {
    return () => {
      if (explainAbortRef.current) {
        explainAbortRef.current.abort();
      }
    };
  }, []);

  const highlightText = (text) => {
    if (!text || !debouncedQuery) return text;
    const stopwords = new Set([
      "in",
      "on",
      "at",
      "with",
      "and",
      "of",
      "for",
      "the",
      "a",
      "an",
      "over",
      "to"
    ]);
    const tokens = debouncedQuery
      .toLowerCase()
      .split(/[^a-z0-9]+/i)
      .filter((token) => token.length > 2 && !stopwords.has(token));
    if (tokens.length === 0) return text;
    const escaped = tokens.map((token) => token.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
    const regex = new RegExp(`(${escaped.join("|")})`, "gi");
    const matcher = new RegExp(`(${escaped.join("|")})`, "i");
    const parts = String(text).split(regex);
    return parts.map((part, idx) =>
      matcher.test(part) ? (
        <mark key={`${part}-${idx}`}>{part}</mark>
      ) : (
        <span key={`${part}-${idx}`}>{part}</span>
      )
    );
  };

  return (
    <div className="app-shell">
      <div className="glow-layer" aria-hidden="true" />
      <header className="hero">
        <div>
          <div className="brand">MIR Lab</div>
          <h1>Semantic multimedia search engine</h1>
          <p>
            Local CLIP + FAISS + RAG baseline designed for direct comparison with
            larger multimodal and cloud-backed architectures.
          </p>
        </div>
        <div className="status-card">
          <div className="status-row">
            <span className="status-label">Server</span>
            <span
              className={`status-pill ${
                serverReady === true ? "ready" : serverReady === false ? "offline" : "pending"
              }`}
            >
              {serverReady === null && "checking"}
              {serverReady === true && "online"}
              {serverReady === false && "offline"}
            </span>
          </div>
          {progress && progress.stage !== "ready" && (
            <div className="progress-card">
              <div className="progress-head">
                <span>{progress.message || "Indexing"}</span>
                <span>{progress.percent || 0}%</span>
              </div>
              <div className="progress-track">
                <div
                  className="progress-fill"
                  style={{ width: `${progress.percent || 0}%` }}
                />
              </div>
            </div>
          )}
          <div className="status-note">API: {apiBase}</div>
          {statusInfo && (
            <div className="status-note">
              Index: {statusInfo.index_type} · CLIP: {statusInfo.clip_model}
            </div>
          )}
          {statusInfo && (
            <div className="status-note">
              Images: {statusInfo.index_size}/{statusInfo.max_images} · Device:{" "}
              {statusInfo.device}
            </div>
          )}
          {metrics && (
            <div className="status-note">
              Embeddings: {metrics.embeddings_source || "n/a"}{" "}
              {typeof metrics.encoding_time_sec === "number"
                ? `(${metrics.encoding_time_sec}s)`
                : ""}
            </div>
          )}
          {metrics && typeof metrics.index_build_time_sec === "number" && (
            <div className="status-note">
              Index build: {metrics.index_build_time_sec}s
            </div>
          )}
          {metrics && typeof metrics.avg_query_ms === "number" && (
            <div className="status-note">Avg query: {metrics.avg_query_ms} ms</div>
          )}
          {selectedModel && (
            <div className="status-note">Ollama model: {selectedModel}</div>
          )}
        </div>
      </header>

      <main>
        <section className="insights">
          <article className="insight-card">
            <h3>Core architecture</h3>
            <p>
              CLIP-based multimodal encoding, FAISS vector search, and grounded RAG
              explanations running locally for traceability.
            </p>
            <div className="badge-row">
              <span className="badge">CLIP ViT</span>
              <span className="badge">FAISS HNSW</span>
              <span className="badge">Ollama RAG</span>
            </div>
          </article>
          <article className="insight-card">
            <h3>Controlled methodology</h3>
            <p>
              Deterministic indexing, versioned query sets, and automatic metrics
              such as Recall@K, latency, and throughput.
            </p>
            <div className="badge-row">
              <span className="badge">Recall@K</span>
              <span className="badge">MRR</span>
              <span className="badge">nDCG</span>
            </div>
          </article>
          <article className="insight-card">
            <h3>Multi-cloud roadmap</h3>
            <p>
              Controlled replication across Redshift, Synapse, and BigQuery with
              managed search services for comparative evaluation.
            </p>
            <div className="badge-row">
              <span className="badge">AWS</span>
              <span className="badge">Azure</span>
              <span className="badge">GCP</span>
            </div>
          </article>
        </section>

        <section className="search-panel">
          <div className="panel-head">
            <label className="search-label" htmlFor="query">
              Query live
            </label>
            <span className="panel-pill">semantic retrieval</span>
          </div>
          <div className="search-row">
            <input
              id="query"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Ex: a dog playing in the park"
              className="search-input"
            />
          </div>
          <div className="quick-queries" aria-label="Quick queries">
            {quickQueries.map((item) => (
              <button
                key={item}
                type="button"
                className="quick-chip"
                onClick={() => setQuery(item)}
              >
                {item}
              </button>
            ))}
          </div>
          <div className="search-meta">
            {status === "loading" && "Searching..."}
            {status === "error" && `Error: ${error}`}
            {status === "ready" &&
              `${results.length} results${latencyMs ? ` · ${latencyMs} ms` : ""}`}
            {status === "idle" && "Type to start searching"}
          </div>
          <div className="toggle-row">
            <span>View mode</span>
            <button
              type="button"
              className={`toggle ${showResearch ? "active" : ""}`}
              onClick={() => setShowResearch((prev) => !prev)}
            >
              {showResearch ? "Research" : "Demo"}
            </button>
          </div>
        </section>

        {showResearch && (
          <section className="control-panel">
          <div className="control-head">
            <div>
              <h2>Search configuration</h2>
              <p>Active parameters for retrieval, reranking, and modality selection.</p>
            </div>
            <div className="control-tags">
              <span className="badge ghost">local</span>
              <span className="badge ghost">reproducible</span>
            </div>
          </div>
          <div className="control-grid">
            <div className="control-item">
              <label htmlFor="topK">Top-K results</label>
              <input
                id="topK"
                type="range"
                min="1"
                max="20"
                value={topK}
                onChange={(event) => setTopK(Number(event.target.value))}
              />
              <span className="control-value">{topK}</span>
            </div>
            <div className="control-item">
              <label>Caption reranking</label>
              <button
                type="button"
                className={`toggle ${rerank ? "active" : ""}`}
                onClick={() => setRerank((prev) => !prev)}
              >
                {rerank ? "Enabled" : "Disabled"}
              </button>
            </div>
            <div className="control-item">
              <label htmlFor="alpha">Alpha rerank</label>
              <input
                id="alpha"
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={rerankAlpha}
                onChange={(event) => setRerankAlpha(Number(event.target.value))}
              />
              <span className="control-value">{rerankAlpha.toFixed(2)}</span>
            </div>
            <div className="control-item">
              <label htmlFor="modality">Modality target</label>
              <select
                id="modality"
                className="search-input"
                value={modality}
                onChange={(event) => setModality(event.target.value)}
              >
                <option value="image">Image</option>
                <option value="video" disabled>
                  Video (soon)
                </option>
                <option value="audio" disabled>
                  Audio (soon)
                </option>
              </select>
            </div>
          </div>
          <div className="control-foot">
            {statusInfo?.modalities_planned && (
              <span>
                Planned modalities: {statusInfo.modalities_planned.join(", ")} · Index{" "}
                {statusInfo.index_type}
              </span>
            )}
          </div>
          </section>
        )}

        <section className="explain-panel rag-spotlight">
          <div className="explain-header">
            <div>
              <div className="rag-badge">RAG enabled</div>
              <h2>Generated explanations (Ollama)</h2>
              <p>Locally generated summary for the retrieved matches.</p>
            </div>
            <div className="explain-actions">
              <select
                className="search-input"
                value={selectedModel}
                onChange={(event) => setSelectedModel(event.target.value)}
                disabled={ollamaModels.length === 0}
                aria-label="Select Ollama model"
              >
                {ollamaModels.length === 0 && (
                  <option value="">No models</option>
                )}
                {ollamaModels.map((model) => (
                  <option key={model} value={model}>
                    {model}
                  </option>
                ))}
              </select>
              <button
                type="button"
                className="explain-button"
                onClick={requestExplanation}
                disabled={
                  status !== "ready" ||
                  results.length === 0 ||
                  explainStatus === "loading" ||
                  !selectedModel
                }
              >
                {explainStatus === "loading"
                  ? "Generating..."
                  : explanation
                    ? "Regenerate"
                    : "Generate explanation"}
              </button>
            </div>
          </div>
          <div className="explain-body explain-output">
            {ollamaError && `Ollama error: ${ollamaError}`}
            {explainStatus === "idle" && "Press the button to generate an explanation."}
            {explainStatus === "loading" && "Ollama is processing the results..."}
            {explainStatus === "error" && `Error: ${explainError}`}
            {explainStatus === "ready" &&
              explanation.split("\n").map((line, idx) => (
                <div key={`${line}-${idx}`}>{highlightText(line)}</div>
              ))}
          </div>
          {explainMeta && (
            <div className="explain-meta">
              {explainMeta.model && <span>Model: {explainMeta.model}</span>}
              {explainMeta.duration_ms && <span>Duration: {explainMeta.duration_ms}ms</span>}
              {explainMeta.prompt_tokens && (
                <span>Prompt: {explainMeta.prompt_tokens} tok</span>
              )}
              {explainMeta.completion_tokens && (
                <span>Completion: {explainMeta.completion_tokens} tok</span>
              )}
            </div>
          )}
        </section>

        <section className="results">
          <div className="results-head">
            <h2>Results</h2>
            <span className="results-count">
              {status === "ready" ? `${results.length} results` : ""}
            </span>
          </div>
          <div className="results-grid">
            {results.length > 0 ? (
              results.map((item) => {
                const imageUrl = `${apiBase}${item.image_url}`;
                const contextText =
                  item.context || item.explanation || item.rag_context || item.reason || null;
                return (
                  <article
                    key={`${item.file_name}-${item.rank}`}
                    className="result-card"
                    onClick={() => setSelectedResult(item)}
                    role="button"
                    tabIndex={0}
                    onKeyDown={(event) => {
                      if (event.key === "Enter") {
                        setSelectedResult(item);
                      }
                    }}
                  >
                    <div className="result-media">
                      <img
                        src={imageUrl}
                        alt={item.file_name}
                        loading="lazy"
                      />
                    </div>
                    <div className="result-body">
                      <div className="result-name">{item.file_name}</div>
                      <div className="result-score">Score: {item.score.toFixed(3)}</div>
                      {typeof item.caption_count === "number" && (
                        <div className="result-score">
                          Captions: {item.caption_count}
                        </div>
                      )}
                      {item.caption && (
                        <div className="result-caption">{highlightText(item.caption)}</div>
                      )}
                      {contextText && <div className="result-context">{contextText}</div>}
                    </div>
                  </article>
                );
              })
            ) : (
              <div className="empty-card">
                Results will appear here after the first search.
              </div>
            )}
          </div>
        </section>

        {showResearch && (
          <section className="roadmap">
          <div>
            <h2>Multimedia extensions</h2>
            <p>
              The backend is structured for expanded semantic indexing across video
              and audio through embedding aggregation and temporal chunking.
            </p>
          </div>
          <div className="roadmap-grid">
            <div className="roadmap-card">
              <h4>Video</h4>
              <p>Temporal sampling and per-segment embeddings for scene search.</p>
            </div>
            <div className="roadmap-card">
              <h4>Audio</h4>
              <p>Acoustic embeddings plus transcripts for precise multimodal search.</p>
            </div>
            <div className="roadmap-card">
              <h4>Hybrid Search</h4>
              <p>Metadata filtering and semantic reranking to reduce false positives.</p>
            </div>
          </div>
          </section>
        )}

        {showResearch && (
          <section className="console">
          <div className="console-head">
            <div>
              <h2>System Console</h2>
              <p>Live logs and runtime metric summaries.</p>
            </div>
            <div className="console-badges">
              <span className="badge ghost">/logs</span>
              <span className="badge ghost">/metrics/summary</span>
            </div>
          </div>
          <div className="console-grid">
            <div className="console-panel">
              <h4>Metrics summary</h4>
              <div className="console-metrics">
                <div>
                  <span className="metric-label">Events</span>
                  <span className="metric-value">
                    {metricsSummary?.events ?? "n/a"}
                  </span>
                </div>
                <div>
                  <span className="metric-label">Avg search</span>
                  <span className="metric-value">
                    {metricsSummary?.avg_search_ms ?? "n/a"} ms
                  </span>
                </div>
                <div>
                  <span className="metric-label">Avg LLM</span>
                  <span className="metric-value">
                    {metricsSummary?.avg_llm_ms ?? "n/a"} ms
                  </span>
                </div>
              </div>
            </div>
            <div className="console-panel console-logs">
              <h4>Live logs</h4>
              <div className="console-logstream">
                {logs.length === 0 && (
                  <div className="console-empty">No log events yet.</div>
                )}
                {logs.map((line, index) => (
                  <div key={`${line}-${index}`} className="console-line">
                    {line}
                  </div>
                ))}
              </div>
            </div>
          </div>
          </section>
        )}

        {showResearch && (
          <section className="benchmarks">
            <div className="benchmarks-head">
              <div>
                <h2>Local benchmarks</h2>
                <p>Direct comparison between the 1k and 5k configurations.</p>
              </div>
              <div className="benchmarks-tags">
                <span className="badge ghost">baseline</span>
                <span className="badge ghost">comparative</span>
              </div>
            </div>
            {!benchmarks.run_1000 && !benchmarks.run_5000 && (
              <div className="bench-note">
                Data unavailable. Start the updated backend to expose the
                <code>/benchmarks</code> endpoint.
              </div>
            )}
            <div className="benchmarks-grid">
              <div className="bench-card">
                <h4>1k images</h4>
                <div className="bench-row">
                  <span>Recall@1</span>
                  <strong>{benchmarks.run_1000?.evaluation?.recall?.["recall@1"] ?? "n/a"}</strong>
                </div>
                <div className="bench-row">
                  <span>Recall@5</span>
                  <strong>{benchmarks.run_1000?.evaluation?.recall?.["recall@5"] ?? "n/a"}</strong>
                </div>
                <div className="bench-row">
                  <span>Recall@10</span>
                  <strong>{benchmarks.run_1000?.evaluation?.recall?.["recall@10"] ?? "n/a"}</strong>
                </div>
                <div className="bench-row">
                  <span>Avg query</span>
                  <strong>{benchmarks.run_1000?.evaluation?.avg_query_ms ?? "n/a"} ms</strong>
                </div>
              </div>
              <div className="bench-card">
                <h4>5k images</h4>
                <div className="bench-row">
                  <span>Recall@1</span>
                  <strong>{benchmarks.run_5000?.evaluation?.recall?.["recall@1"] ?? "n/a"}</strong>
                </div>
                <div className="bench-row">
                  <span>Recall@5</span>
                  <strong>{benchmarks.run_5000?.evaluation?.recall?.["recall@5"] ?? "n/a"}</strong>
                </div>
                <div className="bench-row">
                  <span>Recall@10</span>
                  <strong>{benchmarks.run_5000?.evaluation?.recall?.["recall@10"] ?? "n/a"}</strong>
                </div>
                <div className="bench-row">
                  <span>Avg query</span>
                  <strong>{benchmarks.run_5000?.evaluation?.avg_query_ms ?? "n/a"} ms</strong>
                </div>
              </div>
            </div>
          </section>
        )}
      </main>

      {selectedResult && (
        <div className="drawer" role="dialog" aria-modal="true">
          <div className="drawer-content">
            <button
              type="button"
              className="drawer-close"
              onClick={() => setSelectedResult(null)}
            >
              Close
            </button>
            <div className="drawer-media">
              <img
                src={`${apiBase}${selectedResult.image_url}`}
                alt={selectedResult.file_name}
              />
            </div>
            <div className="drawer-body">
              <h3>{selectedResult.file_name}</h3>
              <div className="drawer-meta">
                <span>Rank: {selectedResult.rank}</span>
                <span>Score: {selectedResult.score.toFixed(3)}</span>
                {typeof selectedResult.caption_count === "number" && (
                  <span>Captions: {selectedResult.caption_count}</span>
                )}
              </div>
            {selectedResult.caption && (
              <p className="drawer-caption">{highlightText(selectedResult.caption)}</p>
            )}
            <div className="drawer-rag">
              <div className="drawer-rag-head">
                <span>RAG</span>
                <button
                  type="button"
                  className="drawer-link"
                  onClick={requestExplanation}
                  disabled={
                    status !== "ready" ||
                    results.length === 0 ||
                    explainStatus === "loading" ||
                    !selectedModel
                  }
                >
                  {explainStatus === "loading"
                    ? "Generating..."
                    : explanation
                      ? "Regenerate"
                      : "Generate explanation"}
                </button>
              </div>
              <div className="drawer-rag-body">
                {explainStatus === "ready" && explanation
                  ? explanation
                      .split("\n")
                      .slice(0, 4)
                      .map((line, idx) => (
                        <div key={`${line}-${idx}`}>{highlightText(line)}</div>
                      ))
                  : "Generate an explanation to preview the RAG summary."}
              </div>
            </div>
            <button
              type="button"
              className="drawer-link"
              onClick={() => window.open(`${apiBase}${selectedResult.image_url}`, "_blank")}
            >
                Open image
              </button>
            </div>
          </div>
        </div>
      )}

      <footer>
        MIR multimodal · CLIP + FAISS + RAG
      </footer>
    </div>
  );
}
