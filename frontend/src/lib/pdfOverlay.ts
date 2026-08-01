export type PdfViewportTransform = {
  width: number;
  height: number;
  transform: [number, number, number, number, number, number];
};

function applyTransform(x: number, y: number, [a, b, c, d, e, f]: PdfViewportTransform['transform']) {
  return [x * a + y * c + e, x * b + y * d + f] as const;
}

/** Convert a top-left, crop-relative source bbox to CSS pixels in a PDF.js viewport. */
export function bboxToViewportStyle(
  bbox: [number, number, number, number],
  pageHeight: number,
  viewport: PdfViewportTransform,
) {
  const [x0, y0, x1, y1] = bbox;
  const values = [x0, y0, x1, y1, pageHeight, viewport.width, viewport.height, ...viewport.transform];
  if (!values.every(Number.isFinite) || x1 < x0 || y1 < y0 || pageHeight <= 0) return null;

  const first = applyTransform(x0, pageHeight - y1, viewport.transform);
  const second = applyTransform(x1, pageHeight - y0, viewport.transform);
  const left = Math.min(first[0], second[0]);
  const top = Math.min(first[1], second[1]);
  const right = Math.max(first[0], second[0]);
  const bottom = Math.max(first[1], second[1]);
  return { left, top, width: right - left, height: bottom - top };
}
