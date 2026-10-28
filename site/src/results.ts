import { City, TYPES, FinalType, countBy } from "./data";

const calm = matchMedia("(prefers-reduced-motion: reduce)").matches;

const byId = (id: string): HTMLElement => {
  const e = document.getElementById(id);
  if (!e) throw new Error(`Missing element #${id}`);
  return e;
};

function el<K extends keyof HTMLElementTagNameMap>(tag: K, cls = "", text = ""): HTMLElementTagNameMap[K] {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text) e.textContent = text;
  return e;
}

const pct = (n: number, total: number): string => `${((100 * n) / total).toFixed(1)}%`;
const num = (n: number): string => n.toLocaleString("en");

const CONF: [string, string][] = [["high", "High"], ["medium", "Medium"], ["low", "Low"], ["", "None"]];
const SIZES = ["0.1-0.5 million", "0.5-1 million", "1-5 million", "5 million+"];

function table(head: string[], body: string[][]): HTMLTableElement {
  const t = el("table");
  const hr = t.createTHead().insertRow();
  head.forEach(h => hr.appendChild(el("th", "", h)));
  const tb = t.createTBody();
  for (const row of body) {
    const tr = tb.insertRow();
    row.forEach((v, i) => tr.appendChild(el("td", i ? "n" : "", v)));
  }
  return t;
}

function stats(cities: City[], counts: Map<string, number>): void {
  const box = byId("res-stats");
  const total = cities.length;
  const first = el("div", "card");
  first.append(el("div", "val", num(total)), el("div", "lab", "Cities classified"));
  box.appendChild(first);
  for (const t of TYPES) {
    const n = counts.get(t) ?? 0;
    const c = el("div", t === "polycentric" ? "card warn" : "card");
    c.append(el("div", "val", num(n)), el("div", "lab", t), el("div", "sn", `${pct(n, total)} of cities`));
    box.appendChild(c);
  }
}

function bars(total: number, counts: Map<string, number>): void {
  const box = byId("res-bars");
  const fills: [HTMLElement, string][] = [];
  for (const t of TYPES) {
    const n = counts.get(t) ?? 0;
    const row = el("div", "bar");
    const fill = el("div", `fill ${t}`);
    const track = el("div", "track");
    track.appendChild(fill);
    row.append(el("span", "", t), track, el("span", "", `${num(n)} · ${pct(n, total)}`));
    box.appendChild(row);
    fills.push([fill, `${(100 * n) / total}%`]);
  }
  const grow = (): void => {
    requestAnimationFrame(() => fills.forEach(([f, w]) => { f.style.width = w; }));
  };
  if (calm || !("IntersectionObserver" in window)) { grow(); return; }
  const io = new IntersectionObserver(es => {
    if (es.some(e => e.isIntersecting)) { grow(); io.disconnect(); }
  }, { threshold: 0.3 });
  io.observe(box);
}

function confidence(cities: City[]): void {
  const body = TYPES.map(t => {
    const sub = cities.filter(c => c.type === t);
    const by = countBy(sub, c => c.confidence);
    return [t, ...CONF.map(([k]) => num(by.get(k) ?? 0))];
  });
  byId("res-conf").appendChild(table(["final_type", ...CONF.map(([, l]) => l)], body));

  const poly = cities.filter(c => c.type === "polycentric");
  const low = poly.filter(c => c.confidence === "low").length;
  const p = byId("res-poly");
  p.textContent = `${low} of ${poly.length} polycentric calls are low confidence. Treat the polycentric list as candidates to check, not as established results.`;
}

function sizes(cities: City[]): void {
  const found = [...new Set(cities.map(c => c.sizeClass))];
  const order = [...SIZES.filter(s => found.includes(s)), ...found.filter(s => !SIZES.includes(s))];
  const body = order.map(s => {
    const by = countBy(cities.filter(c => c.sizeClass === s), c => c.type);
    return [s, ...TYPES.map((t: FinalType) => num(by.get(t) ?? 0))];
  });
  byId("res-size").appendChild(table(["size_class", ...TYPES], body));
}

export function renderResults(cities: City[]): void {
  const counts = countBy(cities, c => c.type);
  byId("res-status").textContent = "Counts are computed live from city_types.csv (column final_type).";
  stats(cities, counts);
  bars(cities.length, counts);
  confidence(cities);
  sizes(cities);
}

export function renderError(err: unknown): void {
  const msg = err instanceof Error ? err.message : String(err);
  byId("res-status").textContent = `Could not load the results: ${msg}`;
}
