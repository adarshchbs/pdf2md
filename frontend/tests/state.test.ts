import { describe, expect, it } from 'vitest';
import { bboxOf, bboxToCss, formatBbox } from '../src/lib/api';
import { readPaneState, togglePane, writePaneState } from '../src/lib/paneState';

describe('comparison invariants', () => {
  it('keeps PDF bbox lower-left transform inside page', () => {
    expect(bboxToCss([10, 20, 110, 220], 612, 792)).toEqual({
      left: (10 / 612) * 100,
      bottom: (20 / 792) * 100,
      width: (100 / 612) * 100,
      height: (200 / 792) * 100,
    });
    expect(bboxToCss([2, 4, 1, 9], 612, 792)).toBeNull();
  });
  it('uses one rounded outer bbox instead of every fragment bbox', () => {
    const element = {
      element: {
        fragments: [
          { page_number: 8, bbox: { x0: 206.917, y0: 437.253, x1: 390.499, y1: 461.964 } },
          { page_number: 8, bbox: { x0: 211.105, y0: 461.253, x1: 386.308, y1: 509.964 } },
          { page_number: 9, bbox: { x0: 1, y0: 2, x1: 3, y1: 4 } },
        ],
      },
    };
    const bbox = bboxOf(element);
    expect(bbox).toEqual([206.917, 437.253, 390.499, 509.964]);
    expect(formatBbox(bbox)).toBe('206.9, 437.3, 390.5, 510.0');
  });

  it('persists pane collapse state', () => {
    const s = new Map<string, string>();
    const storage = {
      getItem: (k: string) => s.get(k) ?? null,
      setItem: (k: string, v: string) => s.set(k, v),
    } as unknown as Storage;
    const next = togglePane(readPaneState(storage), 'reference');
    writePaneState(storage, next);
    expect(readPaneState(storage).reference).toBe(true);
    expect(togglePane(next, 'reference').reference).toBe(false);
  });
});
