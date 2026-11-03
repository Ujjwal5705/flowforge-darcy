import { useCallback, useEffect, useState } from 'react';
import {
  Activity,
  ArrowDown,
  ArrowRight,
  ArrowUpRight,
  Check,
  CheckCircle2,
  ChevronRight,
  CircleHelp,
  Clock3,
  Download,
  FlaskConical,
  Grid2X2,
  Layers3,
  LoaderCircle,
  Pencil,
  Play,
  RotateCcw,
  Shuffle,
  SlidersHorizontal,
  Waves,
  X,
} from 'lucide-react';
import { api, download } from './api';
import { fieldStats, fmt, makePreset, presets, validField } from './fields';
import type { Preset } from './fields';
import { Heatmap } from './Heatmap';
import { LossChart } from './LossChart';
import type { Example, ExampleSummary, Field, History, ModelInfo, Prediction } from './types';

const clone = (field: Field) => field.map((row) => [...row]);

export default function App() {
  const [info, setInfo] = useState<ModelInfo | null>(null);
  const [examples, setExamples] = useState<ExampleSummary[]>([]);
  const [history, setHistory] = useState<History[]>([]);
  const [field, setField] = useState<Field | null>(null);
  const [baseField, setBaseField] = useState<Field | null>(null);
  const [baseId, setBaseId] = useState<string | null>(null);
  const [baseName, setBaseName] = useState('');
  const [referenceId, setReferenceId] = useState<string | null>(null);
  const [sourceName, setSourceName] = useState('');
  const [activePreset, setActivePreset] = useState<Preset | null>(null);
  const [seed, setSeed] = useState(7);
  const [brush, setBrush] = useState(1);
  const [radius, setRadius] = useState(1);
  const [result, setResult] = useState<Prediction | null>(null);
  const [roundTrip, setRoundTrip] = useState(0);
  const [busy, setBusy] = useState(false);
  const [loadingExample, setLoadingExample] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [reload, setReload] = useState(0);
  const [compare, setCompare] = useState(true);

  useEffect(() => {
    let active = true;
    setLoading(true);
    setError('');
    Promise.all([
      api<ModelInfo>('/api/model-info'),
      api<ExampleSummary[]>('/api/examples'),
      api<History[]>('/api/training-history'),
    ])
      .then(async ([model, list, rows]) => {
        if (!list.length)
          throw new Error('No held-out examples are available. Please try again later.');
        const sample = await api<Example>(`/api/examples/${list[0].id}`);
        if (!active) return;
        setInfo(model);
        setExamples(list);
        setHistory(rows);
        setField(clone(sample.coefficient));
        setBaseField(clone(sample.coefficient));
        setReferenceId(sample.id);
        setBaseId(sample.id);
        setSourceName(sample.name);
        setBaseName(sample.name);
      })
      .catch((reason) => {
        if (active)
          setError(reason instanceof Error ? reason.message : 'Could not load the model.');
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [reload]);

  async function loadExample(id: string) {
    setLoadingExample(true);
    setError('');
    try {
      const sample = await api<Example>(`/api/examples/${id}`);
      setField(clone(sample.coefficient));
      setBaseField(clone(sample.coefficient));
      setReferenceId(id);
      setBaseId(id);
      setSourceName(sample.name);
      setBaseName(sample.name);
      setActivePreset(null);
      setResult(null);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Could not load this example.');
    } finally {
      setLoadingExample(false);
    }
  }

  function selectPreset(id: Preset, newSeed = seed) {
    const next = makePreset(id, newSeed);
    const name = presets.find((preset) => preset.id === id)!.name;
    setField(next);
    setBaseField(clone(next));
    setSourceName(name);
    setBaseName(name);
    setActivePreset(id);
    setReferenceId(null);
    setBaseId(null);
    setResult(null);
    setError('');
  }

  const paint = useCallback(
    (i: number, j: number) => {
      setField(
        (previous) =>
          previous?.map((row, x) =>
            row.map((value, y) =>
              (x - i) ** 2 + (y - j) ** 2 <= (radius - 0.4) ** 2 ? brush : value,
            ),
          ) ?? null,
      );
      setReferenceId(null);
      setResult(null);
      setSourceName('Custom field');
      setError('');
    },
    [brush, radius],
  );

  function reset() {
    if (!baseField) return;
    setField(clone(baseField));
    setReferenceId(baseId);
    setSourceName(baseName);
    setResult(null);
    setError('');
  }

  async function run() {
    if (!field || !validField(field)) {
      setError('Use a 32×32 field with finite coefficients between 0.1 and 1.0.');
      return;
    }
    setBusy(true);
    setError('');
    const start = performance.now();
    try {
      const response = await api<Prediction>('/api/predict', {
        coefficient: field,
        example_id: referenceId,
      });
      setRoundTrip(performance.now() - start);
      setResult(response);
      setCompare(true);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Prediction failed. Please try again.');
    } finally {
      setBusy(false);
    }
  }

  function exportResult(format: 'json' | 'csv') {
    if (!result || !field || !info) return;
    const name = `flowforge-${result.model_version}`;
    if (format === 'json') {
      download(
        `${name}.json`,
        JSON.stringify(
          {
            created_at: new Date().toISOString(),
            input: field,
            ...result,
            coordinates: info.coordinates,
            equation: info.equation,
            units: info.solution_units,
            source: info.provenance,
            normalization: info.normalization,
          },
          null,
          2,
        ),
        'application/json',
      );
    } else {
      const rows = [
        'x_index,y_index,x_coordinate,y_coordinate,coefficient,prediction,reference,absolute_error',
      ];
      for (let i = 0; i < 32; i++)
        for (let j = 0; j < 32; j++)
          rows.push(
            [
              i,
              j,
              info.coordinates.x_coordinates[i],
              info.coordinates.y_coordinates[j],
              field[i][j],
              result.prediction[i][j],
              result.reference?.[i][j] ?? '',
              result.absolute_error?.[i][j] ?? '',
            ].join(','),
          );
      download(`${name}.csv`, rows.join('\n'), 'text/csv');
    }
  }

  const disabled = loading || loadingExample || busy || !info;
  const stats = field ? fieldStats(field) : null;
  const pressureRange: [number, number] | undefined = result
    ? [
        Math.min(
          result.summary.minimum,
          result.reference ? fieldStats(result.reference).min : Infinity,
        ),
        Math.max(
          result.summary.maximum,
          result.reference ? fieldStats(result.reference).max : -Infinity,
        ),
      ]
    : undefined;

  return (
    <>
      <a className="skip-link" href="#workspace">
        Skip to simulation
      </a>
      <header className="site-header">
        <div className="header-inner">
          <a className="brand" href="#overview" aria-label="FLOWFORGE overview">
            <span className="brand-symbol">
              <Waves size={25} strokeWidth={1.8} />
            </span>
            <span>
              FLOW<span className="brand-light">FORGE</span>
              <small>NEURAL PDE SOLVER</small>
            </span>
          </a>
          <nav aria-label="Main navigation">
            <a href="#overview">Overview</a>
            <a href="#workspace">Workspace</a>
            <a href="#methodology">Methodology</a>
          </nav>
          <span className="research-label">
            <FlaskConical size={14} /> Research preview
          </span>
        </div>
      </header>

      <main>
        <section id="overview" className="overview section-wrap">
          <div className="hero-copy">
            <div className="eyebrow">
              <span /> SCIENTIFIC MACHINE LEARNING
            </div>
            <h1>
              Explore the medium.
              <br />
              <span>Predict the flow.</span>
            </h1>
            <p className="hero-description">
              Turn a permeability field into an approximate Darcy flow solution with a trained
              Fourier neural operator.
            </p>
            <div className="hero-actions">
              <a className="primary-button" href="#workspace">
                Run a simulation <ArrowRight size={17} />
              </a>
              <a className="text-link" href="#methodology">
                Explore the method <ArrowUpRight size={15} />
              </a>
            </div>
            <div className="hero-note">
              <CheckCircle2 size={14} /> Real PDEBench data <span>·</span> Trained in this session{' '}
              <span>·</span> Open methodology
            </div>
          </div>
          <div className="equation-panel">
            <div className="equation-heading">
              <span className="eyebrow">THE DARCY PROBLEM</span>
              <span className="tag">2D · steady flow</span>
            </div>
            <div
              className="equation"
              aria-label="Negative divergence of a times gradient u equals one"
            >
              −∇ · (a ∇u) = <span>1</span>
            </div>
            <div className="operator-flow">
              <span>
                <i className="mini-grid" />
                <strong>a(x, y)</strong>
                <small>Permeability</small>
              </span>
              <ArrowRight size={20} />
              <span className="operator-node">
                <Layers3 size={22} />
                <strong>FNO</strong>
                <small>Learned operator</small>
              </span>
              <ArrowRight size={20} />
              <span>
                <Activity size={26} />
                <strong>û(x, y)</strong>
                <small>Predicted solution</small>
              </span>
            </div>
            <div className="equation-footer">
              <span>Unit square</span>
              <span>Fixed forcing</span>
              <span>Zero-exterior reference</span>
            </div>
          </div>
        </section>

        <div className="provenance-strip section-wrap">
          <div>
            <span className="tiny-label">DATASET</span>
            <strong>
              PDEBench <ArrowUpRight size={12} />
            </strong>
          </div>
          <div>
            <span className="tiny-label">MODEL</span>
            <strong>Fourier Neural Operator</strong>
          </div>
          <div>
            <span className="tiny-label">SPATIAL GRID</span>
            <strong>
              32 × 32 <span className="muted">/ sampled from 128²</span>
            </strong>
          </div>
          <div>
            <span className="tiny-label">INFERENCE</span>
            <strong className={info ? 'status-ready' : 'muted'}>
              <i className={info ? 'status-dot' : 'status-dot offline'} />
              {info ? 'Model ready · CPU' : loading ? 'Connecting to model…' : 'Model unavailable'}
            </strong>
          </div>
        </div>

        <section id="workspace" className="workspace section-wrap">
          <div className="section-heading">
            <div>
              <div className="eyebrow">01 / INTERACTIVE WORKSPACE</div>
              <h2>A field of possibilities.</h2>
              <p>Select a held-out sample, or make a medium of your own.</p>
            </div>
            <span className="section-index">
              <Grid2X2 size={15} /> DARCY FLOW LAB
            </span>
          </div>
          {error && (
            <div className="error-banner" role="alert">
              <CircleHelp size={18} />
              <span>{error}</span>
              {!info ? (
                <button onClick={() => setReload(reload + 1)}>Retry connection</button>
              ) : (
                <button aria-label="Dismiss error" onClick={() => setError('')}>
                  <X size={16} />
                </button>
              )}
            </div>
          )}

          <div className="workspace-shell">
            <aside className="controls">
              <div className="control-heading">
                <SlidersHorizontal size={16} />
                <h3>Construct your input</h3>
              </div>
              <label className="control-label" htmlFor="sample-select">
                BENCHMARK SAMPLE{' '}
                <span
                  className="help-title"
                  title="These samples belong to the held-out test split."
                >
                  <CircleHelp size={13} />
                </span>
              </label>
              <select
                id="sample-select"
                value={referenceId ?? ''}
                disabled={disabled}
                onChange={(event) => {
                  if (event.target.value) void loadExample(event.target.value);
                }}
              >
                <option value="" disabled>
                  Choose a held-out sample
                </option>
                {examples.map((sample) => (
                  <option key={sample.id} value={sample.id}>
                    {sample.name}
                  </option>
                ))}
              </select>
              <p className="control-hint">Includes a reference solution for comparison.</p>
              <div className="control-divider">
                <span>or start with a preset</span>
              </div>
              <div className="preset-list">
                {presets.map((preset, index) => (
                  <button
                    key={preset.id}
                    disabled={disabled}
                    className={`preset-button ${activePreset === preset.id ? 'selected' : ''}`}
                    onClick={() => selectPreset(preset.id)}
                    title={preset.description}
                  >
                    <span className={`preset-thumb thumb-${index}`} />
                    <span>{preset.name}</span>
                    {activePreset === preset.id ? <Check size={14} /> : <ChevronRight size={14} />}
                  </button>
                ))}
              </div>
              {activePreset === 'heterogeneous' && (
                <button
                  className="new-pattern"
                  disabled={disabled}
                  onClick={() => {
                    setSeed(seed + 1);
                    selectPreset('heterogeneous', seed + 1);
                  }}
                >
                  <Shuffle size={13} /> New pattern <span>seed {seed}</span>
                </button>
              )}
              <div className="brush-controls">
                <div className="brush-heading">
                  <Pencil size={14} />
                  <strong>Paint the field</strong>
                  <span>binary brush</span>
                </div>
                <div className="brush-options" role="group" aria-label="Brush permeability">
                  <button
                    disabled={disabled}
                    aria-pressed={brush === 0.1}
                    className={brush === 0.1 ? 'active' : ''}
                    onClick={() => setBrush(0.1)}
                  >
                    <i className="low-swatch" />
                    Low · 0.1
                  </button>
                  <button
                    disabled={disabled}
                    aria-pressed={brush === 1}
                    className={brush === 1 ? 'active' : ''}
                    onClick={() => setBrush(1)}
                  >
                    <i className="high-swatch" />
                    High · 1.0
                  </button>
                </div>
                <label className="range-label" htmlFor="brush-radius">
                  Brush radius{' '}
                  <span>
                    {radius} {radius === 1 ? 'cell' : 'cells'}
                  </span>
                </label>
                <input
                  id="brush-radius"
                  type="range"
                  min="1"
                  max="4"
                  value={radius}
                  disabled={disabled}
                  onChange={(event) => setRadius(Number(event.target.value))}
                />
              </div>
              <button className="reset-button" onClick={reset} disabled={disabled || !baseField}>
                <RotateCcw size={14} />
                Reset field
              </button>
              <div className="scope-note">
                <CircleHelp size={15} />
                <p>
                  Presets and edits are exploratory. A valid input range does not guarantee an
                  accurate prediction.
                </p>
              </div>
            </aside>

            <div className="simulation-area">
              <div className="simulation-toolbar">
                <div>
                  <span className={`field-badge ${referenceId ? 'reference-badge' : ''}`}>
                    {referenceId ? 'HELD-OUT SAMPLE' : 'CUSTOM INPUT'}
                  </span>
                  <strong>{sourceName || 'Preparing the workspace'}</strong>
                </div>
                <span className="grid-tag">
                  32 × 32 <span>·</span> a ∈ [0.1, 1.0]
                </span>
              </div>
              <div className="field-pair">
                <article className="field-panel">
                  <div className="field-heading">
                    <div>
                      <span className="step-number">A</span>
                      <h3>Permeability field</h3>
                    </div>
                    <span className="panel-label">INPUT · a(x, y)</span>
                  </div>
                  {field ? (
                    <Heatmap
                      field={field}
                      label="Permeability field editor"
                      kind="coefficient"
                      range={[0.1, 1]}
                      onPaint={paint}
                      disabled={disabled}
                      testId="input-canvas"
                    />
                  ) : (
                    <div className="map-skeleton" aria-label="Loading coefficient field" />
                  )}
                  <div className="field-caption">
                    <Pencil size={12} />
                    <span>Draw on the grid · arrow keys + Space also work</span>
                  </div>
                </article>
                <article className="field-panel output-panel">
                  <div className="field-heading">
                    <div>
                      <span className="step-number">B</span>
                      <h3>Predicted solution</h3>
                    </div>
                    <span className="panel-label">SURROGATE · û(x, y)</span>
                  </div>
                  {busy ? (
                    <div className="empty-map busy-state" role="status">
                      <LoaderCircle size={28} className="spin" />
                      <h4>Evaluating the operator</h4>
                      <p>
                        Applying the saved training transforms
                        <br />
                        and running the FNO.
                      </p>
                    </div>
                  ) : result ? (
                    <Heatmap
                      field={result.prediction}
                      label="FNO predicted solution"
                      range={pressureRange}
                      testId="prediction-canvas"
                    />
                  ) : (
                    <div className="empty-map">
                      <span className="empty-symbol">
                        <Layers3 size={30} strokeWidth={1.3} />
                      </span>
                      <h4>Your solution starts here.</h4>
                      <p>
                        Prepare a permeability field,
                        <br />
                        then run the trained operator.
                      </p>
                      <span className="empty-equation">
                        a(x, y) <ArrowRight size={14} /> û(x, y)
                      </span>
                    </div>
                  )}
                  <div className="field-caption">
                    <Activity size={12} />
                    <span>
                      {result
                        ? 'Model prediction · unconstrained dataset solution units'
                        : 'A learned approximation, with measured held-out evidence'}
                    </span>
                  </div>
                </article>
              </div>
              <div className="run-toolbar">
                <div className="input-summary">
                  <span>
                    INPUT RANGE{' '}
                    <strong>{stats ? `${fmt(stats.min)} – ${fmt(stats.max)}` : '—'}</strong>
                  </span>
                  <span>
                    MEAN <strong>{stats ? fmt(stats.mean) : '—'}</strong>
                  </span>
                  {referenceId && (
                    <span className="reference-available">
                      <CheckCircle2 size={13} />
                      Reference available
                    </span>
                  )}
                </div>
                <button
                  className="primary-button run-button"
                  disabled={disabled || !field}
                  onClick={() => void run()}
                >
                  {busy ? (
                    <LoaderCircle size={16} className="spin" />
                  ) : (
                    <Play size={15} fill="currentColor" />
                  )}
                  {busy ? 'Running…' : loadingExample ? 'Loading sample…' : 'Run prediction'}
                  <span>→</span>
                </button>
              </div>
            </div>
          </div>

          {result && (
            <div className="results" data-testid="results">
              <div className="results-heading">
                <div>
                  <span className="success-icon">
                    <CheckCircle2 size={17} />
                  </span>
                  <h3>Prediction complete</h3>
                  <span className="caption">{result.model_version}</span>
                </div>
                <div className="export-actions">
                  <span>Export result</span>
                  <button onClick={() => exportResult('json')}>
                    <Download size={13} />
                    JSON
                  </button>
                  <button onClick={() => exportResult('csv')}>
                    <Download size={13} />
                    CSV
                  </button>
                </div>
              </div>
              <div className="result-stats">
                <div>
                  <span className="tiny-label">MODEL INFERENCE</span>
                  <strong>
                    {fmt(result.inference_ms)}
                    <small> ms</small>
                  </strong>
                  <span>{fmt(roundTrip)} ms including round trip</span>
                </div>
                <div>
                  <span className="tiny-label">SOLUTION MIN / MAX</span>
                  <strong>
                    {fmt(result.summary.minimum)}
                    <small> / </small>
                    {fmt(result.summary.maximum)}
                  </strong>
                  <span>Dataset solution units</span>
                </div>
                <div>
                  <span className="tiny-label">MEAN SOLUTION</span>
                  <strong>{fmt(result.summary.mean)}</strong>
                  <span>Across all 1,024 grid values</span>
                </div>
                <div>
                  <span className="tiny-label">SAMPLE RELATIVE L₂</span>
                  <strong>
                    {result.comparison
                      ? `${fmt(result.comparison.relative_l2 * 100)}%`
                      : 'No reference'}
                  </strong>
                  <span>
                    {result.comparison
                      ? 'Compared with this held-out sample'
                      : 'Custom fields have no ground truth'}
                  </span>
                </div>
              </div>
              {result.warnings.map((warning, i) => (
                <div className="result-notice" key={i}>
                  <CircleHelp size={14} />
                  <span>{warning}</span>
                </div>
              ))}
              {result.reference && (
                <>
                  <button className="compare-toggle" onClick={() => setCompare(!compare)}>
                    <Layers3 size={15} />
                    {compare ? 'Hide' : 'Show'} reference comparison{' '}
                    <ArrowDown size={14} className={compare ? 'rotated' : ''} />
                  </button>
                  {compare && (
                    <div className="comparison-pair">
                      <article>
                        <div className="field-heading">
                          <h3>Dataset reference</h3>
                          <span className="tag">HELD-OUT · u</span>
                        </div>
                        <Heatmap
                          field={result.reference}
                          label="Held-out dataset reference"
                          range={pressureRange}
                          testId="reference-canvas"
                        />
                        <p className="caption">
                          Same scale as the prediction. A sampled fine-grid solution.
                        </p>
                      </article>
                      <article>
                        <div className="field-heading">
                          <h3>Absolute error</h3>
                          <span className="tag">|û − u|</span>
                        </div>
                        <Heatmap
                          field={result.absolute_error!}
                          label="Absolute prediction error"
                          kind="error"
                          testId="error-canvas"
                        />
                        <p className="caption">
                          Sample MAE {fmt(result.comparison!.mae)} · MSE{' '}
                          {fmt(result.comparison!.mse)}
                        </p>
                      </article>
                    </div>
                  )}
                </>
              )}
            </div>
          )}
        </section>

        <section id="methodology" className="methodology section-wrap">
          <div className="section-heading">
            <div>
              <div className="eyebrow">02 / THE METHOD & THE EVIDENCE</div>
              <h2>Learned from data. Open to scrutiny.</h2>
              <p>A small, reproducible experiment with clearly bounded claims.</p>
            </div>
            <span className="section-index">
              <FlaskConical size={15} /> RESEARCH NOTES
            </span>
          </div>
          <div className="method-top">
            <article className="method-explanation">
              <span className="tiny-label">WHY A NEURAL OPERATOR?</span>
              <h3>A mapping between fields.</h3>
              <p>
                Darcy flow describes transport through a porous medium. Permeability <i>a</i>{' '}
                controls how easily fluid moves; the solution <i>u</i> describes the resulting field
                under fixed forcing.
              </p>
              <p>
                The FNO learns this mapping from examples. Fourier layers mix spatial information
                globally, while local channel transformations add nonlinear structure. One forward
                pass produces the approximate solution.
              </p>
              <div className="architecture-line">
                <span>
                  Lift<small>1 channel + grid</small>
                </span>
                <ChevronRight size={15} />
                <span>
                  Fourier blocks
                  <small>
                    {info?.architecture.n_layers ?? '—'} layers ·{' '}
                    {info?.architecture.hidden_channels ?? '—'} channels
                  </small>
                </span>
                <ChevronRight size={15} />
                <span>
                  Project<small>1 solution channel</small>
                </span>
              </div>
            </article>
            <article className="experiment-card">
              <div className="card-title">
                <h3>The experiment</h3>
                <span className="tag">FROM RANDOM INITIALIZATION</span>
              </div>
              <dl>
                <div>
                  <dt>Data source</dt>
                  <dd>PDEBench · Darcy β = 1</dd>
                </div>
                <div>
                  <dt>Subset / resolution</dt>
                  <dd>
                    {info
                      ? `First ${info.provenance.sample_count.toLocaleString()} pairs / 32²`
                      : '—'}
                  </dd>
                </div>
                <div>
                  <dt>Train / validation / test</dt>
                  <dd>
                    {info ? Object.values(info.training.split_sizes).join(' / ') : 'Loading…'}
                  </dd>
                </div>
                <div>
                  <dt>Epochs / selected checkpoint</dt>
                  <dd>
                    {info ? `${info.training.epochs} / epoch ${info.training.best_epoch}` : '—'}
                  </dd>
                </div>
                <div>
                  <dt>Seed / batch size</dt>
                  <dd>{info ? `${info.training.seed} / ${info.training.batch_size}` : '—'}</dd>
                </div>
                <div>
                  <dt>Parameters · real scalars</dt>
                  <dd>{info?.training.parameters.real_scalars.toLocaleString() ?? '—'}</dd>
                </div>
                <div>
                  <dt>Training time / device</dt>
                  <dd>
                    {info
                      ? `${fmt(info.training.training_seconds / 60)} min / ${info.training.hardware.machine} CPU`
                      : '—'}
                  </dd>
                </div>
                <div>
                  <dt>Package</dt>
                  <dd>neuraloperator {info?.training.versions.neuraloperator ?? '—'}</dd>
                </div>
              </dl>
            </article>
          </div>
          <div className="evidence-grid">
            <article className="chart-card">
              <div className="card-title">
                <h3>Learning curves</h3>
                <span className="tag">MEASURED</span>
              </div>
              {info && history.length ? (
                <LossChart
                  history={history}
                  yStd={info.normalization.y_std}
                  bestEpoch={info.training.best_epoch}
                />
              ) : (
                <div className="chart-empty">
                  {loading
                    ? 'Loading recorded training history…'
                    : 'Connect to the model to view measured curves.'}
                </div>
              )}
              <p className="caption">
                Validation selects the checkpoint. The test split is reserved for final evaluation.
              </p>
            </article>
            <article className="evaluation-card">
              <div className="card-title">
                <h3>Held-out evaluation</h3>
                <span className="tag">{info?.training.split_sizes.test ?? '—'} SAMPLES</span>
              </div>
              <table>
                <caption className="sr-only">
                  FNO and training-mean-field baseline on the same test split
                </caption>
                <thead>
                  <tr>
                    <th scope="col">Metric</th>
                    <th scope="col">FNO</th>
                    <th scope="col">Mean field</th>
                  </tr>
                </thead>
                <tbody>
                  {(['relative_l2', 'mse', 'mae'] as const).map((key) => (
                    <tr key={key}>
                      <th scope="row">
                        {key === 'relative_l2' ? 'Relative L₂' : key.toUpperCase()}
                      </th>
                      <td>{info ? fmt(info.evaluation[key], 4) : '—'}</td>
                      <td>{info ? fmt(info.baseline[key], 4) : '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="caption">
                Baseline: the pixelwise mean training solution. Relative L₂ is the mean of
                individual sample norm ratios.
              </p>
              <div className="offline-latency">
                <Clock3 size={15} />
                <div>
                  <strong>
                    {info ? `${fmt(info.latency.median_ms)} ms` : '—'}
                    <span> median inference</span>
                  </strong>
                  <small>
                    {info
                      ? `${info.latency.timed_calls} warmed calls · ${info.latency.machine} CPU · ${info.latency.threads} threads`
                      : 'Offline benchmark context appears with model metadata.'}
                  </small>
                </div>
              </div>
            </article>
          </div>
          <div className="method-details">
            <article>
              <h3>Exact inputs. Consistent transforms.</h3>
              <p>
                Coefficients are the dataset’s raw 0.1 / 1.0 phases. Fields retain every fourth cell
                center from the 128² source. Scalar normalization is fitted only on training data,
                saved, and shared by training and inference.
              </p>
              <p>
                Arrays keep <code>field[x][y]</code> order. Plots display x horizontally and y
                upward. Hover a grid to inspect a value; exports include sampled physical
                coordinates.
              </p>
            </article>
            <article>
              <h3>Fixed physics. Explicit limits.</h3>
              <p>
                The unit-square problem uses constant forcing 1. The generator fills exterior ghost
                cells with zero; its interior edge samples need not be zero. The model does not
                enforce the boundary or positivity.
              </p>
              <p>
                Hand-drawn fields may fall outside the training distribution. Results use dataset
                units with no SI calibration. Coarse fields lose fine structure; this is a learned
                surrogate, not an exact numerical solver.
              </p>
            </article>
          </div>
          <div className="sources">
            <span className="tiny-label">PRIMARY SOURCES</span>
            <a href="https://doi.org/10.18419/darus-2986" target="_blank" rel="noreferrer">
              PDEBench dataset <ArrowUpRight size={13} />
            </a>
            <a href="https://github.com/pdebench/PDEBench" target="_blank" rel="noreferrer">
              PDEBench source <ArrowUpRight size={13} />
            </a>
            <a href="https://neuraloperator.github.io/stable/" target="_blank" rel="noreferrer">
              NeuralOperator <ArrowUpRight size={13} />
            </a>
            <a href="https://arxiv.org/abs/2010.08895" target="_blank" rel="noreferrer">
              FNO paper <ArrowUpRight size={13} />
            </a>
          </div>
        </section>
      </main>
      <footer className="site-footer section-wrap">
        <a className="footer-brand" href="#overview">
          <Waves size={20} />
          FLOWFORGE
        </a>
        <span>Ujjwal Sharma · QUEST Lab application · 2026</span>
        <span>Research surrogate · PDEBench CC BY 4.0</span>
      </footer>
    </>
  );
}
