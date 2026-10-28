export type Row = Record<string, string>;

export function parseCSV(text: string): { cols: string[]; rows: Row[] } {
  const out: string[][] = [];
  let f = '', r: string[] = [], q = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (q) {
      if (c === '"') { if (text[i + 1] === '"') { f += '"'; i++; } else q = false; }
      else f += c;
    } else if (c === '"') q = true;
    else if (c === ',') { r.push(f); f = ''; }
    else if (c === '\n' || c === '\r') {
      if (c === '\r' && text[i + 1] === '\n') i++;
      r.push(f); f = ''; out.push(r); r = [];
    } else f += c;
  }
  if (f !== '' || r.length) { r.push(f); out.push(r); }
  const body = out.filter(x => x.some(v => v.trim() !== ''));
  const cols = (body.shift() ?? []).map(s => s.trim());
  const rows = body.map(x => Object.fromEntries(cols.map((c, i) => [c, x[i] ?? ''])) as Row);
  return { cols, rows };
}
