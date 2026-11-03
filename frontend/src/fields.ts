import type { Field } from './types';

export type Preset =
  | 'homogeneous'
  | 'high-inclusion'
  | 'low-inclusion'
  | 'channel'
  | 'heterogeneous';
export const presets: { id: Preset; name: string; description: string }[] = [
  { id: 'homogeneous', name: 'Homogeneous', description: 'Uniform high permeability' },
  {
    id: 'high-inclusion',
    name: 'High inclusion',
    description: 'Conductive region in a low matrix',
  },
  { id: 'low-inclusion', name: 'Low inclusion', description: 'Resistant region in a high matrix' },
  { id: 'channel', name: 'Channel', description: 'A connected high-permeability path' },
  { id: 'heterogeneous', name: 'Heterogeneous', description: 'Seeded smooth binary pattern' },
];

export function makePreset(preset: Preset, seed = 7, n = 32): Field {
  // These are user-facing exploratory inputs, never labelled benchmark data.
  return Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => {
      const x = (i + 0.5) / n,
        y = (j + 0.5) / n;
      const inclusion = (x - 0.5) ** 2 + (y - 0.5) ** 2 < 0.21 ** 2;
      switch (preset) {
        case 'homogeneous':
          return 1;
        case 'high-inclusion':
          return inclusion ? 1 : 0.1;
        case 'low-inclusion':
          return inclusion ? 0.1 : 1;
        case 'channel':
          return Math.abs(y - (0.5 + 0.19 * Math.sin(x * Math.PI * 2))) < 0.095 ? 1 : 0.1;
        case 'heterogeneous': {
          const phase = seed * 0.731;
          return Math.sin(7 * x + phase) * Math.cos(5 * y - phase) +
            0.5 * Math.sin(11 * y + x * 3) >
            0
            ? 1
            : 0.1;
        }
      }
    }),
  );
}

export function fieldStats(field: Field) {
  const values = field.flat();
  return {
    min: Math.min(...values),
    max: Math.max(...values),
    mean: values.reduce((a, b) => a + b, 0) / values.length,
  };
}

export function validField(field: Field): boolean {
  return (
    field.length === 32 &&
    field.every(
      (row) =>
        row.length === 32 &&
        row.every(
          (value) =>
            typeof value === 'number' && Number.isFinite(value) && value >= 0.1 && value <= 1,
        ),
    )
  );
}

export const fmt = (value: number, digits = 3) =>
  Math.abs(value) < 0.001 && value !== 0
    ? value.toExponential(2)
    : value.toLocaleString('en-US', { maximumSignificantDigits: digits });
