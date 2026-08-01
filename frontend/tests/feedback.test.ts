import { describe, expect, it } from 'vitest';
import { type FeedbackPacket, serializeFeedback } from '../src/lib/feedback';

const packet: FeedbackPacket = {
  documentId: 'doc-1',
  pdfPath: '/a.pdf',
  split: 'validation',
  page: 4,
  bbox: [1.234, 2.345, 3.456, 4.567],
  candidate: { id: 'p-1', rendered: 'Candidate text' },
  reference: { id: 's-1', rendered: 'Reference text' },
  alignment: 'mismatch · text differs',
  comment: 'Check grouping',
};

describe('feedback packet', () => {
  it('serializes a compact queryable packet with rounded geometry', () => {
    const out = serializeFeedback(packet);
    expect(out.split('\n')).toHaveLength(10);
    expect(out).toContain('PDF: /a.pdf');
    expect(out).toContain('bbox [1.2, 2.3, 3.5, 4.6]');
    expect(out).toContain('candidate=p-1; reference=s-1');
    expect(out).toContain('element_id=p-1');
    expect(out).toContain('Check grouping');
    expect(out).not.toContain('Exact JSON');
  });

  it('collapses whitespace and truncates long rendered previews', () => {
    const out = serializeFeedback({
      ...packet,
      candidate: { id: 'p-1', rendered: `first\nsecond ${'x'.repeat(300)}` },
    });
    expect(out).toContain('Candidate preview: "first second');
    expect(out).toContain('…"');
    expect(out.split('\n')).toHaveLength(10);
  });
});
