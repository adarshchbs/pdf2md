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
    expect(out.split('\n')).toHaveLength(9);
    expect(out).toContain('Document: doc-1 | split=validation');
    expect(out).toContain('PDF: /a.pdf');
    expect(out).toContain('bbox=[1.2, 2.3, 3.5, 4.6]');
    expect(out).toContain('Candidate: p-1 | "Candidate text"');
    expect(out).toContain('Reference: s-1 | "Reference text"');
    expect(out).toContain('/elements/candidate id=p-1');
    expect(out).toContain('/elements/reference id=s-1');
    expect(out).toContain('Issue: Check grouping');
    expect(out).not.toContain('Exact JSON');
    expect(out).not.toContain('source_item_ids');
  });

  it('collapses whitespace and bounds previews and comments', () => {
    const out = serializeFeedback({
      ...packet,
      candidate: { id: 'p-1', rendered: `first\nsecond ${'x'.repeat(300)}` },
      comment: `check ${'y'.repeat(300)}`,
    });
    expect(out).toContain('Candidate: p-1 | "first second');
    expect(out).toContain('Issue: check');
    expect(out.match(/…/g)).toHaveLength(2);
    expect(out.split('\n')).toHaveLength(9);
  });

  it('emits a usable reference locator without a candidate', () => {
    const out = serializeFeedback({ ...packet, candidate: null });
    expect(out).toContain('/elements/reference id=s-1');
    expect(out).not.toContain('/elements/candidate id=unmatched');
  });
});
