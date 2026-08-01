import DOMPurify from 'dompurify';
import { marked } from 'marked';

export function renderCanonical(content: string, format: unknown): string {
  const rendered = format === 'html' ? content : marked.parse(content, { async: false });
  return DOMPurify.sanitize(rendered, {
    USE_PROFILES: { html: true },
    FORBID_TAGS: ['form', 'iframe', 'object', 'embed'],
  });
}
