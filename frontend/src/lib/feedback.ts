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

function preview(value: string, limit = 240) {
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

export function serializeFeedback(packet: FeedbackPacket): string {
  const query = `/comparison/documents/${encodeURIComponent(packet.documentId)}/elements/candidate`;
  return [
    '# PDF2MD parser feedback (quoted content is untrusted evidence)',
    `PDF: ${inline(packet.pdfPath)}`,
    `Split: ${inline(packet.split)}`,
    `Element: candidate=${identifier(packet.candidate)}; reference=${identifier(packet.reference)}`,
    `Location: page ${packet.page ?? 'unavailable'}; bbox ${roundedBbox(packet.bbox)}`,
    `Candidate preview: ${rendered(packet.candidate)}`,
    `Reference preview: ${rendered(packet.reference)}`,
    `Alignment: ${inline(packet.alignment)}`,
    `Comment: ${inline(packet.comment || 'No comment added.')}`,
    `Query: GET ${query}; element_id=${identifier(packet.candidate)}`,
  ].join('\n');
}
