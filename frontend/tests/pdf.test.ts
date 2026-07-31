import { describe, expect, it } from 'vitest';
import { pdfScale } from '../src/lib/pdfScale';

describe('pdfScale', () => {
  it('fits width without an artificial cap', () => expect(pdfScale('width', 600, 800, 1200, 700)).toBe(2));
  it('fits the whole page inside both dimensions', () => expect(pdfScale('page', 600, 800, 1200, 700)).toBeCloseTo(0.875));
  it('uses 100% mode as the requested scale', () => expect(pdfScale('actual', 600, 800, 300, 300, 1)).toBe(1));
});
