import { describe, expect, it } from 'vitest';
import { bboxToViewportStyle } from '../src/lib/pdfOverlay';

describe('bboxToViewportStyle', () => {
  it('maps top-left PDF coordinates through a rotation-aware viewport transform', () => {
    const viewport = {
      width: 200,
      height: 300,
      transform: [0, 1, 1, 0, 0, 0] as [number, number, number, number, number, number],
    };
    expect(bboxToViewportStyle([10, 20, 40, 80], 100, viewport)).toEqual({
      left: 20,
      top: 10,
      width: 60,
      height: 30,
    });
  });

  it('maps an unrotated PDF.js viewport transform', () => {
    const viewport = {
      width: 200,
      height: 300,
      transform: [2, 0, 0, -2, 0, 200] as [number, number, number, number, number, number],
    };
    expect(bboxToViewportStyle([10, 20, 40, 80], 100, viewport)).toEqual({
      left: 20,
      top: 40,
      width: 60,
      height: 120,
    });
  });

  it('rejects invalid boxes', () => {
    const viewport = {
      width: 1,
      height: 1,
      transform: [1, 0, 0, 1, 0, 0] as [number, number, number, number, number, number],
    };
    expect(bboxToViewportStyle([4, 8, 2, 9], 100, viewport)).toBeNull();
  });
});
