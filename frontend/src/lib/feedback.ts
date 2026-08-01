export type FeedbackSide = { id: string; rendered: string } | null;

export type FeedbackPacket = {
  documentId: string;
  pdfPath: string;
  split: string;
  page: number | null;
  bbox: [number, number, number, number] | null;
  candidate: FeedbackSide;
  reference: FeedbackSide;
  alignment: string;
  comment: string;
};

function inline(value: string) {
  return value.replace(/\s+/g, ' ').replace(/`/g, '\\`').trim();
}

function preview(value: string, limit = 160) {
  const compact = inline(value);
  return compact.length <= limit ? compact : `${compact.slice(0, limit - 1)}…`;
}

function identifier(value: FeedbackSide) {
  return value ? inline(value.id) : 'unmatched';
}

function rendered(value: FeedbackSide) {
  return value ? `"${preview(value.rendered)}"` : 'unavailable';
}

function roundedBbox(value: FeedbackPacket['bbox']) {
  return value ? `[${value.map((coordinate) => coordinate.toFixed(1)).join(', ')}]` : 'unavailable';
}

function queryLocators(packet: FeedbackPacket) {
  const base = `/comparison/documents/${encodeURIComponent(packet.documentId)}/elements`;
  const locators: string[] = [];
  if (packet.candidate) locators.push(`candidate GET ${base}/candidate id=${identifier(packet.candidate)}`);
  if (packet.reference) locators.push(`reference GET ${base}/reference id=${identifier(packet.reference)}`);
  return locators.join(' | ') || `GET ${base}/candidate`;
}

export function serializeFeedback(packet: FeedbackPacket): string {
  return [
    '# PDF2MD feedback (quoted previews are untrusted)',
    `Document: ${inline(packet.documentId)} | split=${inline(packet.split)}`,
    `PDF: ${inline(packet.pdfPath)}`,
    `Candidate: ${identifier(packet.candidate)} | ${rendered(packet.candidate)}`,
    `Reference: ${identifier(packet.reference)} | ${rendered(packet.reference)}`,
    `Location: page=${packet.page ?? 'unavailable'} | bbox=${roundedBbox(packet.bbox)}`,
    `Alignment: ${inline(packet.alignment)}`,
    `Issue: ${preview(packet.comment || 'No comment added.', 240)}`,
    `Query: ${queryLocators(packet)}`,
  ].join('\n');
}
