const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const calm = matchMedia("(prefers-reduced-motion: reduce)").matches;

/** Hero: a small grid of cells, a few lit up like city centres. */
export function heroGrid(): void {
  const g = $("gb");
  for (let i = 0; i < 60; i++) {
    const c = document.createElement("i");
    const r = Math.random();
    if (r > 0.9) c.style.background = "#fff";
    else if (r > 0.7) c.style.background = "rgba(255,255,255,.4)";
    g.appendChild(c);
  }
}

/** Pipeline carousel: buttons, dots, arrow keys (when visible) and swipe. */
export function carousel(): void {
  const trk = $("trk"), car = $("car"), dt = $("dt"), ct = $("ct");
  const n = trk.children.length;
  const dots: HTMLElement[] = [];
  let ix = 0;
  const go = (i: number): void => {
    ix = (i + n) % n;
    trk.style.transform = `translateX(-${ix * 100}%)`;
    dots.forEach((d, k) => d.classList.toggle("on", k === ix));
    ct.textContent = `${ix + 1} / ${n}`;
  };
  for (let k = 0; k < n; k++) {
    const d = document.createElement("i");
    d.onclick = () => go(k);
    dt.appendChild(d);
    dots.push(d);
  }
  $("pv").onclick = () => go(ix - 1);
  $("nx").onclick = () => go(ix + 1);
  document.addEventListener("keydown", e => {
    if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
    const r = car.getBoundingClientRect();
    if (r.bottom < 0 || r.top > innerHeight) return;
    e.preventDefault();
    go(ix + (e.key === "ArrowRight" ? 1 : -1));
  });
  let sx: number | null = null;
  car.addEventListener("touchstart", e => { sx = e.touches[0].clientX; }, { passive: true });
  car.addEventListener("touchend", e => {
    if (sx === null) return;
    const d = e.changedTouches[0].clientX - sx;
    if (Math.abs(d) > 50) go(ix + (d < 0 ? 1 : -1));
    sx = null;
  });
  go(0);
}

/** Every .fade block drifts and fades as it moves away from the screen centre. */
export function scrollFade(): void {
  const els = document.querySelectorAll<HTMLElement>(".fade");
  let busy = false;
  const update = (): void => {
    const vh = innerHeight;
    els.forEach(el => {
      if (calm) { el.style.opacity = "1"; el.style.transform = "none"; return; }
      const r = el.getBoundingClientRect();
      const o = r.top + r.height / 2 - vh / 2;
      const t = Math.min(Math.abs(o) / (vh * (o > 0 ? 0.8 : 1)), 1);
      el.style.opacity = String(0.2 + 0.8 * (1 - t * t));
      el.style.transform = `translateY(${Math.sign(o) * t * 40}px)`;
    });
  };
  const queue = (): void => {
    if (busy) return;
    busy = true;
    requestAnimationFrame(() => { update(); busy = false; });
  };
  addEventListener("scroll", queue, { passive: true });
  addEventListener("resize", queue);
  addEventListener("load", update);
  update();
}
