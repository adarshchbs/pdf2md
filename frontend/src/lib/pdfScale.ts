export type PdfFit = 'width' | 'page' | 'actual';

export function pdfScale(mode: PdfFit, pageWidth: number, pageHeight: number, availableWidth: number, availableHeight: number, zoom = 1) {
  if (mode === 'actual') return zoom;
  const widthScale = availableWidth / pageWidth;
  const pageScale = Math.min(widthScale, availableHeight / pageHeight);
  return Math.max(0.1, mode === 'width' ? widthScale : pageScale);
}
