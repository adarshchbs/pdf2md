import { describe, expect, it } from 'vitest';
import { renderCanonical } from '../src/lib/canonical';

describe('renderCanonical', () => {
  it('renders GFM tables instead of showing pipe syntax', () => {
    const html = renderCanonical('| A | B |\n| --- | --- |\n| 1 | 2 |', 'markdown');
    expect(html).toContain('<table>');
    expect(html).toContain('<td>1</td>');
    expect(html).not.toContain('| --- |');
  });

  it('preserves HTML tables', () => {
    const html = renderCanonical('<table><tr><td>value</td></tr></table>', 'html');
    expect(html).toContain('<table>');
    expect(html).toContain('<td>value</td>');
  });

  it('sanitizes active content from Markdown and HTML', () => {
    expect(renderCanonical('<script>alert(1)</script>safe', 'html')).toBe('safe');
    expect(renderCanonical('[bad](javascript:alert(1))', 'markdown')).not.toContain('javascript:');
  });
});
