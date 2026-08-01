import { describe, expect, it } from 'vitest';
import { backendSplit } from '../src/lib/api';
import { type FeedbackPacket, serializeFeedback } from '../src/lib/feedback';
import { togglePane } from '../src/lib/paneState';

describe('API contracts and packet safety', () => {
  it('maps product test label to backend holdout', () => expect(backendSplit('test')).toBe('holdout'));

  it('keeps unmatched sides compact and queryable', () => {
    const packet: FeedbackPacket = {
      documentId: 'doc-1',
      pdfPath: '/original/a.md\nnext',
      split: 'test',
      page: null,
      bbox: null,
      candidate: { id: 'c', rendered: 'ok' },
      reference: null,
      alignment: 'unmatched candidate',
      comment: 'line\ncomment',
    };
    const out = serializeFeedback(packet);
    expect(out).toContain('bbox=unavailable');
    expect(out).toContain('Reference: unmatched | unavailable');
    expect(out).toContain('PDF: /original/a.md next');
    expect(out.split('\n')).toHaveLength(9);
  });

  it('supports maximize as persisted pane mode shape', () => {
    expect(togglePane({ source: true }, 'source').source).toBe(false);
  });
});
