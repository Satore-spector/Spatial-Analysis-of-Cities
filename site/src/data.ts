import { parseCSV } from "./csv";

export const TYPES = ["monocentric", "polycentric", "unclear"] as const;
export type FinalType = (typeof TYPES)[number];

/** Only the columns the site uses. The type always comes from final_type. */
export interface City {
  type: FinalType;
  confidence: string; // "high" | "medium" | "low" | "" (blank for unclear cities)
  sizeClass: string;
}

const REQUIRED = ["final_type", "confidence", "size_class"];
const isType = (s: string): s is FinalType => (TYPES as readonly string[]).includes(s);

export async function loadCities(): Promise<City[]> {
  const res = await fetch(`${import.meta.env.BASE_URL}city_types.csv`);
  if (!res.ok) throw new Error(`city_types.csv returned HTTP ${res.status}`);
  const { cols, rows } = parseCSV(await res.text());
  const missing = REQUIRED.filter(c => !cols.includes(c));
  if (missing.length) throw new Error(`city_types.csv is missing column(s): ${missing.join(", ")}`);
  if (!rows.length) throw new Error("city_types.csv has no rows");
  return rows.map(r => {
    const type = r["final_type"];
    if (!isType(type)) throw new Error(`Unexpected final_type value: "${type}"`);
    return { type, confidence: r["confidence"], sizeClass: r["size_class"] };
  });
}

/** Count cities per key. */
export function countBy(cities: City[], key: (c: City) => string): Map<string, number> {
  const m = new Map<string, number>();
  for (const c of cities) m.set(key(c), (m.get(key(c)) ?? 0) + 1);
  return m;
}
