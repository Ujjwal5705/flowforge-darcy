import { useEffect, useRef, useState } from 'react';
import type { KeyboardEvent, PointerEvent } from 'react';
import type { Field } from './types';
import { fieldStats, fmt } from './fields';

const pressure = ['#1c224c', '#315e9b', '#238eaa', '#45b7a4', '#a6d59b', '#f2df81', '#fde8a2'];
const coefficient = ['#edf0ff', '#b4c4fa', '#6b8ce8', '#3459d7', '#173887'];
const error = ['#f1f1fb', '#c9c2e6', '#9a79bc', '#bd527f', '#ef7759', '#ffd48c'];

function rgb(hex: string): number[] {
  return [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16));
}
function color(value: number, low: number, high: number, palette: string[]) {
  const t = Math.max(0, Math.min(1, (value - low) / (high - low || 1))) * (palette.length - 1);
  const index = Math.min(palette.length - 2, Math.floor(t));
  const a = rgb(palette[index]),
    b = rgb(palette[index + 1]);
  return a.map((v, i) => Math.round(v + (b[i] - v) * (t - index)));
}

type Props = {
  field: Field;
  label: string;
  kind?: 'coefficient' | 'pressure' | 'error';
  range?: [number, number];
  onPaint?: (x: number, y: number) => void;
  disabled?: boolean;
  testId?: string;
};

export function Heatmap({
  field,
  label,
  kind = 'pressure',
  range,
  onPaint,
  disabled,
  testId,
}: Props) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const drawing = useRef(false);
  const [cursor, setCursor] = useState<[number, number]>([16, 16]);
  const [hovered, setHovered] = useState(false);
  const stats = fieldStats(field);
  const low = range?.[0] ?? (kind === 'error' ? 0 : stats.min);
  const high = range?.[1] ?? stats.max;
  const palette = kind === 'coefficient' ? coefficient : kind === 'error' ? error : pressure;
  const n = field.length;

  useEffect(() => {
    const context = canvas.current?.getContext('2d');
    if (!context) return;
    const source = document.createElement('canvas');
    source.width = n;
    source.height = n;
    const sourceContext = source.getContext('2d')!;
    const pixels = sourceContext.createImageData(n, n);
    // JSON/tensors are x-major. Canvas x=i; canvas y=n-1-j makes increasing y upward.
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++) {
        const offset = ((n - 1 - j) * n + i) * 4;
        pixels.data.set([...color(field[i][j], low, high, palette), 255], offset);
      }
    sourceContext.putImageData(pixels, 0, 0);
    context.imageSmoothingEnabled = false;
    context.drawImage(source, 0, 0, 512, 512);
    if (onPaint && hovered && !disabled) {
      context.strokeStyle = '#ffffff';
      context.lineWidth = 2;
      context.strokeRect(
        (cursor[0] * 512) / n + 1,
        ((n - 1 - cursor[1]) * 512) / n + 1,
        512 / n - 2,
        512 / n - 2,
      );
    }
  }, [field, low, high, kind, cursor, hovered, disabled, onPaint, n]);

  function point(event: PointerEvent<HTMLCanvasElement>): [number, number] {
    const rect = event.currentTarget.getBoundingClientRect();
    return [
      Math.max(0, Math.min(n - 1, Math.floor(((event.clientX - rect.left) / rect.width) * n))),
      Math.max(
        0,
        Math.min(n - 1, n - 1 - Math.floor(((event.clientY - rect.top) / rect.height) * n)),
      ),
    ];
  }
  function keyboard(event: KeyboardEvent<HTMLCanvasElement>) {
    if (!onPaint || disabled) return;
    const [i, j] = cursor;
    const keys: Record<string, [number, number]> = {
      ArrowLeft: [i - 1, j],
      ArrowRight: [i + 1, j],
      ArrowUp: [i, j + 1],
      ArrowDown: [i, j - 1],
    };
    if (keys[event.key]) {
      event.preventDefault();
      setHovered(true);
      setCursor(
        keys[event.key].map((value) => Math.max(0, Math.min(n - 1, value))) as [number, number],
      );
    } else if (event.key === ' ' || event.key === 'Enter') {
      event.preventDefault();
      onPaint(i, j);
    }
  }

  return (
    <div className="heatmap">
      <div className="map-with-axis">
        <div className="axis-y">
          <span>{n - 1}</span>
          <span className="axis-name">y index</span>
          <span>0</span>
        </div>
        <canvas
          ref={canvas}
          width={512}
          height={512}
          role="img"
          data-testid={testId}
          className={onPaint ? 'editable-canvas' : ''}
          tabIndex={onPaint && !disabled ? 0 : undefined}
          aria-label={`${label}. ${n} by ${n}, x-major. Range ${fmt(low)} to ${fmt(high)}.${onPaint ? ` Arrow keys move; Space paints. Selected x ${cursor[0]}, y ${cursor[1]}.` : ''}`}
          onKeyDown={keyboard}
          onFocus={() => setHovered(true)}
          onBlur={() => setHovered(false)}
          onPointerEnter={() => setHovered(true)}
          onPointerLeave={() => setHovered(false)}
          onPointerDown={(event) => {
            if (!onPaint || disabled) return;
            event.preventDefault();
            event.currentTarget.focus();
            drawing.current = true;
            event.currentTarget.setPointerCapture(event.pointerId);
            const p = point(event);
            setCursor(p);
            onPaint(...p);
          }}
          onPointerMove={(event) => {
            const p = point(event);
            setCursor(p);
            if (drawing.current && onPaint && !disabled) onPaint(...p);
          }}
          onPointerUp={() => {
            drawing.current = false;
          }}
          onPointerCancel={() => {
            drawing.current = false;
          }}
        />
      </div>
      <div className="axis-x">
        <span>0</span>
        <span>x index</span>
        <span>{n - 1}</span>
      </div>
      <div
        className="color-scale"
        style={{ background: `linear-gradient(to right, ${palette.join(',')})` }}
      />
      <div className="scale-labels">
        <span>{fmt(low)}</span>
        <span>
          {kind === 'coefficient'
            ? 'coefficient a'
            : kind === 'error'
              ? '|û − u|'
              : 'solution · dataset units'}
        </span>
        <span>{fmt(high)}</span>
      </div>
      <div className="probe">
        <span>
          x {cursor[0]}, y {cursor[1]}
        </span>
        <span>{fmt(field[cursor[0]]?.[cursor[1]] ?? 0, 5)}</span>
      </div>
    </div>
  );
}
