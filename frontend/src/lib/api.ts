export type Availability = {
  status: string;
  available: boolean;
  reason?: string | null;
  stage?: string | null;
};

type ApiFragment = {
  page_number?: unknown;
  page_width?: unknown;
  page_height?: unknown;
  rotation?: unknown;
  bbox?: unknown;
};

type ApiElementData = {
  fragments?: ApiFragment[];
  bbox?: unknown;
  geometry?: { bbox?: unknown };
  page_number?: unknown;
  page?: unknown;
  source?: { page?: unknown };
  [key: string]: unknown;
};

export type ApiElement = { element: ApiElementData; canonical_rendering?: string | null };
export type CatalogDoc = {
  id: string;
  split: string;
  local_path?: string;
  metadata?: Record<string, unknown>;
};
export type CatalogEntry = {
  document: CatalogDoc;
  original_source: Availability;
  candidate: Availability;
  reference: Availability;
  gold: null | Availability;
};
export type DocumentDetail = CatalogEntry;
export type EvaluationReport = {
  metrics?: { element_f1: number; mean_geometry_iou: number };
  alignments?: Array<{ candidate_index: number; reference_index: number; score: number }>;
};

export function backendSplit(label: string) {
  return label === 'test' ? 'holdout' : label;
}

export async function apiFetch<T>(path: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(path, { signal });
  if (!response.ok) throw new Error(`${response.status}: ${await response.text()}`);
  return response.json() as Promise<T>;
}

export const api = {
  documents: (split: string, signal?: AbortSignal) =>
    apiFetch<{ documents: CatalogEntry[] }>(
      `/comparison/documents?split=${encodeURIComponent(backendSplit(split))}`,
      signal,
    ),
  document: (id: string, signal?: AbortSignal) =>
    apiFetch<DocumentDetail>(`/comparison/documents/${encodeURIComponent(id)}`, signal),
  elements: (id: string, source: 'candidate' | 'reference', signal?: AbortSignal) =>
    apiFetch<{ elements: ApiElement[]; source: string; stage: string }>(
      `/comparison/documents/${encodeURIComponent(id)}/elements/${source}`,
      signal,
    ),
  evaluation: (id: string, signal?: AbortSignal) =>
    apiFetch<EvaluationReport>(`/comparison/documents/${encodeURIComponent(id)}/evaluation`, signal),
};

export function field(element: ApiElement | undefined, key: string, fallback: unknown = '') {
  return element?.element?.[key] ?? fallback;
}

function readBbox(value: unknown): [number, number, number, number] | null {
  if (Array.isArray(value) && value.length === 4) {
    const values = value.map(Number);
    return values.every(Number.isFinite) ? (values as [number, number, number, number]) : null;
  }
  if (value && typeof value === 'object') {
    const values = ['x0', 'y0', 'x1', 'y1'].map((key) => Number((value as Record<string, unknown>)[key]));
    return values.every(Number.isFinite) ? (values as [number, number, number, number]) : null;
  }
  return null;
}

/** Outer bbox for every fragment of the selected element on its first page. */
export function bboxOf(element: ApiElement | undefined): [number, number, number, number] | null {
  const fragments = element?.element?.fragments;
  if (Array.isArray(fragments) && fragments.length) {
    const firstPage = Number(fragments[0]?.page_number);
    const boxes = fragments
      .filter((fragment) => Number(fragment?.page_number) === firstPage)
      .map((fragment) => readBbox(fragment?.bbox))
      .filter((box): box is [number, number, number, number] => box !== null);
    if (boxes.length) {
      return [
        Math.min(...boxes.map((box) => box[0])),
        Math.min(...boxes.map((box) => box[1])),
        Math.max(...boxes.map((box) => box[2])),
        Math.max(...boxes.map((box) => box[3])),
      ];
    }
  }
  return readBbox(element?.element?.bbox ?? element?.element?.geometry?.bbox);
}

export function formatBbox(bbox: [number, number, number, number] | null) {
  return bbox ? bbox.map((value) => value.toFixed(1)).join(', ') : 'unavailable';
}

export function pageOf(element: ApiElement | undefined) {
  return Number(
    element?.element?.fragments?.[0]?.page_number ??
      element?.element?.page_number ??
      element?.element?.page ??
      element?.element?.source?.page ??
      1,
  );
}

export function bboxToCss(bbox: [number, number, number, number], pageWidth: number, pageHeight: number) {
  if (pageWidth <= 0 || pageHeight <= 0 || bbox[2] < bbox[0] || bbox[3] < bbox[1]) return null;
  return {
    left: (bbox[0] / pageWidth) * 100,
    bottom: (bbox[1] / pageHeight) * 100,
    width: ((bbox[2] - bbox[0]) / pageWidth) * 100,
    height: ((bbox[3] - bbox[1]) / pageHeight) * 100,
  };
}

export function geometryOf(element: ApiElement | undefined) {
  const fragment = element?.element?.fragments?.[0];
  return {
    width: Number(fragment?.page_width || 0),
    height: Number(fragment?.page_height || 0),
    rotation: Number(fragment?.rotation || 0),
  };
}
