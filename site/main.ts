import './style.css';
import { parseCSV, Row } from './csv';

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const PAL = ['#0F766E', '#D97706', '#7C3AED', '#DC2626', '#2563EB', '#65A30D', '#DB2777', '#64748B'];
const esc = (s: string) => s.replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c] as string));

let cols: string[] = [], rows: Row[] = [], types: string[] = [], nums: string[] = [];
let typeCol = '', confCol = '', nameCol = '';
const F = { q: '', type: '', conf: '' };
const sort = { col: '', dir: 1 };
const ax = { x: '', y: '' };
let selected = -1, hover = -1;

interface Pt { x: number; y: number; tx: number; ty: number; a: number; ta: number; sx: number; sy: number }
let pts: Pt[] = [];
const colorOf = (t: string) => PAL[Math.max(0, types.indexOf(t)) % PAL.length];
const pass = (r: Row) =>
  (!F.type || r[typeCol] === F.type) && (!F.conf || r[confCol] === F.conf) &&
  (!F.q || (r[nameCol] ?? '').toLowerCase().includes(F.q.toLowerCase()));

function load(text: string) {
  const p = parseCSV(text);
  if (!p.rows.length) { $('dropmsg').textContent = 'That file has no rows.'; return; }
  cols = p.cols; rows = p.rows;
  const find = (...res: RegExp[]) => { for (const re of res) { const c = cols.find(x => re.test(x.toLowerCase())); if (c) return c; } return ''; };
  typeCol = find(/^(city_)?type$/, /type/, /class/, /label/) || cols[cols.length - 1];
  confCol = find(/conf/);
  nameCol = find(/^(city|name|city_name|urban_centre|slug)$/, /city|name|slug/) || cols[0];
  types = [...new Set(rows.map(r => r[typeCol]))].filter(Boolean).sort();
  nums = cols.filter(c => ![typeCol, confCol, nameCol].includes(c) &&
    rows.filter(r => r[c] !== '' && isFinite(parseFloat(r[c]))).length >= rows.length * 0.8);
  F.q = F.type = F.conf = ''; selected = -1; sort.col = '';
  ax.x = nums[0] ?? ''; ax.y = nums[1] ?? nums[0] ?? '';
  $('app').hidden = false;
  setup(); renderAll();
  $('app').scrollIntoView({ behavior: 'smooth' });
}

function countUp(el: HTMLElement, to: number) {
  const t0 = performance.now();
  const step = (t: number) => {
    const k = Math.min((t - t0) / 900, 1);
    el.textContent = Math.round(to * (1 - Math.pow(1 - k, 3))).toLocaleString();
    if (k < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

function setup() {
  const opts = (a: string[]) => a.map(c => `<option>${esc(c)}</option>`).join('');
  $<HTMLSelectElement>('ax').innerHTML = opts(nums); $<HTMLSelectElement>('ay').innerHTML = opts(nums);
  $<HTMLSelectElement>('ax').value = ax.x; $<HTMLSelectElement>('ay').value = ax.y;
  const confs = confCol ? [...new Set(rows.map(r => r[confCol]))].filter(Boolean).sort() : [];
  const cf = $<HTMLSelectElement>('conf');
  cf.innerHTML = `<option value="">All confidence</option>` + confs.map(c => `<option>${esc(c)}</option>`).join('');
  cf.hidden = !confCol; $<HTMLInputElement>('q').value = '';
  $('legend').innerHTML = types.map(t => `<span><i style="background:${colorOf(t)}"></i>${esc(t)}</span>`).join('');
  pts = rows.map(() => ({ x: .5, y: .5, tx: .5, ty: .5, a: 0, ta: 1, sx: 0, sy: 0 }));
}

function renderStats() {
  const n = rows.filter(pass).length;
  const items = [{ v: n, l: F.type || F.conf || F.q ? 'Cities (filtered)' : 'Cities' },
    ...types.map(t => ({ v: rows.filter(r => pass(r) && r[typeCol] === t).length, l: t }))];
  $('stats').innerHTML = items.map((x, i) =>
    `<div class="card" style="animation-delay:${i * 80}ms"><div class="val">0</div><div class="lab">${esc(x.l)}</div></div>`).join('');
  $('stats').querySelectorAll<HTMLElement>('.val').forEach((el, i) => countUp(el, items[i].v));
}

function renderBars() {
  const base = rows.filter(r => !F.conf || r[confCol] === F.conf);
  const max = Math.max(1, ...types.map(t => base.filter(r => r[typeCol] === t).length));
  $('bars').innerHTML = types.map(t => {
    const c = base.filter(r => r[typeCol] === t).length;
    return `<div class="bar ${F.type === t ? 'on' : ''}" data-t="${esc(t)}"><span>${esc(t)}</span>
      <div class="track"><div class="fill" style="background:${colorOf(t)}" data-w="${c / max * 100}"></div></div><b>${c}</b></div>`;
  }).join('');
  requestAnimationFrame(() => requestAnimationFrame(() =>
    $('bars').querySelectorAll<HTMLElement>('.fill').forEach(f => f.style.width = f.dataset.w + '%')));
  $('bars').querySelectorAll<HTMLElement>('.bar').forEach(b => b.onclick = () => {
    F.type = F.type === b.dataset.t ? '' : (b.dataset.t as string); renderAll();
  });
}

function retarget() {
  const range = (c: string) => {
    const v = rows.map(r => parseFloat(r[c])).filter(isFinite);
    const lo = Math.min(...v), hi = Math.max(...v); return [lo, hi === lo ? lo + 1 : hi];
  };
  if (!ax.x) return;
  const [x0, x1] = range(ax.x), [y0, y1] = range(ax.y);
  rows.forEach((r, i) => {
    const px = parseFloat(r[ax.x]), py = parseFloat(r[ax.y]), ok = isFinite(px) && isFinite(py);
    pts[i].tx = ok ? (px - x0) / (x1 - x0) : .5; pts[i].ty = ok ? (py - y0) / (y1 - y0) : .5;
    pts[i].ta = ok && pass(r) ? 1 : .07;
  });
}

function renderTable() {
  const show = [nameCol, typeCol, confCol, ...nums.slice(0, 4)].filter((c, i, a) => c && a.indexOf(c) === i);
  let idx = rows.map((_, i) => i).filter(i => pass(rows[i]));
  if (sort.col) idx.sort((a, b) => {
    const x = rows[a][sort.col], y = rows[b][sort.col], nx = parseFloat(x), ny = parseFloat(y);
    return (isFinite(nx) && isFinite(ny) ? nx - ny : x.localeCompare(y)) * sort.dir;
  });
  $('count').textContent = `${idx.length} of ${rows.length} cities` + (idx.length > 300 ? ' (showing 300)' : '');
  $('tbl').innerHTML = `<thead><tr>${show.map(c => `<th data-c="${esc(c)}">${esc(c)}${sort.col === c ? (sort.dir > 0 ? ' ▲' : ' ▼') : ''}</th>`).join('')}</tr></thead><tbody>` +
    idx.slice(0, 300).map((i, k) => `<tr data-i="${i}" class="${i === selected ? 'sel' : ''}" style="animation-delay:${Math.min(k, 20) * 25}ms">` +
      show.map(c => c === typeCol ? `<td><span class="pill" style="background:${colorOf(rows[i][c])}">${esc(rows[i][c])}</span></td>` : `<td>${esc(rows[i][c])}</td>`).join('') + '</tr>').join('') + '</tbody>';
  $('tbl').querySelectorAll<HTMLElement>('th').forEach(th => th.onclick = () => {
    const c = th.dataset.c as string; sort.dir = sort.col === c ? -sort.dir : 1; sort.col = c; renderTable();
  });
  $('tbl').querySelectorAll<HTMLElement>('tbody tr').forEach(tr => tr.onclick = () => select(+(tr.dataset.i as string)));
}

function select(i: number) {
  selected = i; const d = $('detail');
  if (i < 0) { d.hidden = true; return; }
  const r = rows[i];
  d.hidden = false;
  d.innerHTML = `<button class="ghost" id="x">Close ✕</button><h3>${esc(r[nameCol])}</h3>` +
    `<span class="pill" style="background:${colorOf(r[typeCol])}">${esc(r[typeCol])}</span><dl>` +
    cols.map(c => `<dt>${esc(c)}</dt><dd>${esc(r[c])}</dd>`).join('') + '</dl>';
  $('x').onclick = () => select(-1);
  renderTable();
}

function renderAll() { renderStats(); renderBars(); retarget(); renderTable(); }

// ---- animated scatter ----
const cv = $<HTMLCanvasElement>('cv');
function frame() {
  const dpr = window.devicePixelRatio || 1, W = cv.clientWidth, H = cv.clientHeight, P = 46;
  if (cv.width !== Math.round(W * dpr)) { cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); }
  const g = cv.getContext('2d') as CanvasRenderingContext2D;
  g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, W, H);
  const css = getComputedStyle(document.documentElement);
  g.strokeStyle = css.getPropertyValue('--line'); g.fillStyle = css.getPropertyValue('--text'); g.font = '12px Roboto Mono, monospace';
  g.beginPath(); g.moveTo(P, 12); g.lineTo(P, H - P); g.lineTo(W - 12, H - P); g.stroke();
  g.globalAlpha = .7; g.fillText(ax.x, W / 2 - 20, H - 14);
  g.save(); g.translate(14, H / 2 + 20); g.rotate(-Math.PI / 2); g.fillText(ax.y, 0, 0); g.restore();
  rows.forEach((r, i) => {
    const p = pts[i]; if (!p) return;
    p.x += (p.tx - p.x) * .12; p.y += (p.ty - p.y) * .12; p.a += (p.ta - p.a) * .15;
    p.sx = P + 8 + p.x * (W - P - 28); p.sy = H - P - 8 - p.y * (H - P - 28);
    g.globalAlpha = p.a; g.fillStyle = colorOf(r[typeCol]);
    g.beginPath(); g.arc(p.sx, p.sy, i === selected ? 9 : i === hover ? 8 : 5.5, 0, 7); g.fill();
    if (i === selected) { g.strokeStyle = css.getPropertyValue('--text'); g.lineWidth = 2; g.stroke(); g.lineWidth = 1; }
  });
  g.globalAlpha = 1; requestAnimationFrame(frame);
}
cv.addEventListener('mousemove', e => {
  const b = cv.getBoundingClientRect(), mx = e.clientX - b.left, my = e.clientY - b.top;
  let best = -1, bd = 144;
  rows.forEach((r, i) => { if (!pass(r)) return; const d = (pts[i].sx - mx) ** 2 + (pts[i].sy - my) ** 2; if (d < bd) { bd = d; best = i; } });
  hover = best; const tip = $('tip'); tip.hidden = best < 0; cv.style.cursor = best < 0 ? 'default' : 'pointer';
  if (best >= 0) { tip.textContent = `${rows[best][nameCol]} · ${rows[best][typeCol]}`; tip.style.left = pts[best].sx + 'px'; tip.style.top = pts[best].sy + 'px'; }
});
cv.addEventListener('mouseleave', () => { hover = -1; $('tip').hidden = true; });
cv.addEventListener('click', () => { if (hover >= 0) select(hover); });

// ---- wiring ----
$<HTMLSelectElement>('ax').onchange = e => { ax.x = (e.target as HTMLSelectElement).value; retarget(); };
$<HTMLSelectElement>('ay').onchange = e => { ax.y = (e.target as HTMLSelectElement).value; retarget(); };
$<HTMLInputElement>('q').oninput = e => { F.q = (e.target as HTMLInputElement).value; renderAll(); };
$<HTMLSelectElement>('conf').onchange = e => { F.conf = (e.target as HTMLSelectElement).value; renderAll(); };
$('clear').onclick = () => { F.q = F.type = F.conf = ''; $<HTMLInputElement>('q').value = ''; $<HTMLSelectElement>('conf').value = ''; renderAll(); };

const read = (f: File) => f.text().then(load);
$<HTMLInputElement>('file').onchange = e => { const f = (e.target as HTMLInputElement).files?.[0]; if (f) read(f); };
const drop = $('drop');
['dragover', 'dragenter'].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add('over'); }));
['dragleave', 'drop'].forEach(ev => drop.addEventListener(ev, () => drop.classList.remove('over')));
drop.addEventListener('drop', e => { e.preventDefault(); const f = (e as DragEvent).dataTransfer?.files[0]; if (f) read(f); });

$('demo').onclick = () => {
  const T = ['monocentric', 'polycentric', 'unclear'], C = ['high', 'medium', 'low'];
  const lines = ['city,type,confidence,population,builtup_km2,gs_subcentres,mcmillen_subcentres,poi_per_km2'];
  for (let i = 1; i <= 80; i++) {
    const t = T[Math.floor(Math.random() * 3)], s = t === 'polycentric' ? 2 + Math.random() * 4 : Math.random() * 1.5;
    const pop = Math.round(1e5 + Math.random() ** 3 * 1.2e7);
    lines.push([`City ${i}`, t, C[Math.floor(Math.random() * 3)], pop, (pop / 9000 + Math.random() * 40).toFixed(1), Math.round(s), Math.round(s * Math.random()), (5 + Math.random() * 60).toFixed(1)].join(','));
  }
  load(lines.join('\n'));
};

fetch('city_types.csv').then(r => r.ok ? r.text() : '').then(t => { if (t && !t.trimStart().startsWith('<')) load(t); }).catch(() => {});
requestAnimationFrame(frame);
