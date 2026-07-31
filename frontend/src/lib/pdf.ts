import workerUrl from 'pdfjs-dist/build/pdf.worker.min.mjs?url';
export { pdfScale, type PdfFit } from './pdfScale';
let pdfjsPromise: Promise<typeof import('pdfjs-dist')> | undefined;
async function pdfjs() {
  pdfjsPromise ??= import('pdfjs-dist').then(module => { module.GlobalWorkerOptions.workerSrc = workerUrl; return module; });
  return pdfjsPromise;
}
export async function loadPdf(source: File | string) {
  const module = await pdfjs();
  return typeof source === 'string' ? module.getDocument({ url: source }).promise : module.getDocument({ data: await source.arrayBuffer() }).promise;
}
