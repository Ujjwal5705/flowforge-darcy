import { useState } from 'react';
import type { History } from './types';
import { fmt } from './fields';

export function LossChart({
  history,
  yStd,
  bestEpoch,
}: {
  history: History[];
  yStd: number;
  bestEpoch: number;
}) {
  const [mode, setMode] = useState<'mse' | 'l2'>('mse');
  if (!history.length) return <p>Training history is unavailable.</p>;
  const series =
    mode === 'mse'
      ? [
          history.map((row) => row.train_normalized_mse),
          history.map((row) => row.validation_mse / yStd ** 2),
        ]
      : [history.map((row) => row.validation_relative_l2)];
  const logs = series.flat().map((value) => Math.log10(Math.max(value, 1e-12)));
  const low = Math.floor(Math.min(...logs)),
    high = Math.ceil(Math.max(...logs));
  const x = (index: number) => 55 + (index / Math.max(history.length - 1, 1)) * 565;
  const y = (value: number) =>
    195 - ((Math.log10(Math.max(value, 1e-12)) - low) / (high - low || 1)) * 170;
  return (
    <div className="loss-chart">
      <div className="chart-toolbar">
        <span className="caption">
          {mode === 'mse' ? 'Standardized MSE' : 'Validation relative L₂'} · log scale
        </span>
        <div className="segmented">
          <button className={mode === 'mse' ? 'active' : ''} onClick={() => setMode('mse')}>
            MSE
          </button>
          <button className={mode === 'l2' ? 'active' : ''} onClick={() => setMode('l2')}>
            Relative L₂
          </button>
        </div>
      </div>
      <svg
        viewBox="0 0 650 235"
        role="img"
        aria-label={`${mode === 'mse' ? 'Training and validation MSE' : 'Validation relative L2'} over ${history.length} real training epochs`}
      >
        <title>Measured training history. Best checkpoint selected at epoch {bestEpoch}.</title>
        {Array.from({ length: high - low + 1 }, (_, i) => low + i).map((exponent) => (
          <g key={exponent}>
            <line x1="55" x2="620" y1={y(10 ** exponent)} y2={y(10 ** exponent)} stroke="#e5e9f0" />
            <text x="44" y={y(10 ** exponent) + 4} textAnchor="end">
              {fmt(10 ** exponent)}
            </text>
          </g>
        ))}
        <line
          x1={x(bestEpoch - 1)}
          x2={x(bestEpoch - 1)}
          y1="25"
          y2="195"
          stroke="#9ca6bb"
          strokeDasharray="4 4"
        />
        {series.map((values, index) => (
          <polyline
            key={index}
            points={values.map((v, i) => `${x(i)},${y(v)}`).join(' ')}
            fill="none"
            stroke={index === 0 ? '#365ee8' : '#159d8f'}
            strokeWidth="2.4"
            strokeLinejoin="round"
          />
        ))}
        {[0, Math.floor((history.length - 1) / 2), history.length - 1].map((i, k) => (
          <text key={k} x={x(i)} y="218" textAnchor="middle">
            {history[i].epoch}
          </text>
        ))}
        <text x="337" y="234" textAnchor="middle">
          Epoch
        </text>
      </svg>
      <div className="chart-legend">
        <span>
          <i className="legend-blue" />
          {mode === 'mse' ? 'Training' : 'Validation'}
        </span>
        {mode === 'mse' && (
          <span>
            <i className="legend-teal" />
            Validation
          </span>
        )}
        <span className="best-label">Best checkpoint · epoch {bestEpoch}</span>
      </div>
    </div>
  );
}
