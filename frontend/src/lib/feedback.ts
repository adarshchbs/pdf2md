export type FeedbackPacket = { pdfPath:string; split:string; page:number|null; bbox:[number,number,number,number]|null; candidate:{id:string;json:unknown;rendered:string}|null; reference:{id:string;json:unknown;rendered:string}|null; alignment:string; comment:string };
function safe(value:string){return value.replace(/\r\n?/g,'\n').trim();}
function fence(value:string){return value.replace(/```/g,'``\\`');}
export function serializeFeedback(packet: FeedbackPacket): string {
  const json=(value:unknown)=>fence(JSON.stringify(value,null,2));
  const side=(label:string,value:FeedbackPacket['candidate'])=>value?`## ${label}\n- Element ID: ${safe(value.id)}\n- Exact JSON:\n\`\`\`json\n${json(value.json)}\n\`\`\`\n- Rendered output:\n\n${fence(safe(value.rendered))}`:`## ${label}\n- Element: null (unmatched / unavailable)`;
  const bbox=packet.bbox ? `[${packet.bbox.map(Number).join(', ')}]` : 'null (unavailable)';
  return `# Parser feedback\n\n## Source\n- PDF: ${fence(safe(packet.pdfPath))}\n- Split: ${fence(safe(packet.split))}\n- Page: ${packet.page ?? 'null'}\n- BBox: ${bbox}\n\n${side('Candidate',packet.candidate)}\n\n${side('Reference',packet.reference)}\n\n## Alignment\n${fence(safe(packet.alignment))}\n\n## Engineer comment\n${fence(safe(packet.comment || '_No comment added._'))}\n`;
}
