import {
  Component, ChangeDetectionStrategy, ElementRef, afterNextRender, inject, DestroyRef, viewChild,
} from '@angular/core';
import { RouterLink } from '@angular/router';
import { CHAPTERS } from '../../core/sections';
import { REPO } from '../../core/repo';
import { Math as MathTex } from '../../shared/math';

/**
 * Home / overview. Frames the whole site: the two families (diffusion vs
 * autoregressive), the "next-frame intuition is backwards" reframing, the
 * Stable-Diffusion pipeline at a glance, and the chapter index. The hero runs
 * a live canvas that resolves a field of noise into structure and back —
 * the core diffusion metaphor, animated.
 */
@Component({
  selector: 'app-overview',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [RouterLink, MathTex],
  templateUrl: './overview.html',
  styleUrl: './overview.scss',
})
export class Overview {
  readonly chapters = CHAPTERS;
  readonly repo = REPO;

  private readonly canvasRef = viewChild.required<ElementRef<HTMLCanvasElement>>('field');
  private readonly destroyRef = inject(DestroyRef);

  constructor() {
    afterNextRender(() => this.runField());
  }

  /** A particle field that denoises into a logarithmic spiral, holds, then re-noises — forever. */
  private runField(): void {
    const canvas = this.canvasRef().nativeElement;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const N = 460;
    const pts = Array.from({ length: N }, (_, i) => {
      const f = i / N;
      const arms = 3;
      const a = f * Math.PI * 2 * 6 + (i % arms) * ((Math.PI * 2) / arms);
      const r = 0.08 + f * 0.92;
      return {
        // target position on a 3-arm spiral, in normalized [-1,1] space
        tx: Math.cos(a) * r,
        ty: Math.sin(a) * r,
        // current + noise seed
        nx: Math.random() * 2 - 1,
        ny: Math.random() * 2 - 1,
        x: 0, y: 0,
        hue: f,
        seed: Math.random() * 1000,
      };
    });

    let raf = 0;
    let running = true;
    let start = performance.now();
    const ease = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
    const lerp = (a: number, b: number, t: number) => a + (b - a) * t;

    const resize = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.max(1, Math.floor(rect.width * dpr));
      canvas.height = Math.max(1, Math.floor(rect.height * dpr));
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    resize();
    const onResize = () => resize();
    window.addEventListener('resize', onResize);

    const colors = ['#7c5cff', '#41d6ff', '#ff5c8a'];

    const frame = (now: number) => {
      if (!running) return;
      const rect = canvas.getBoundingClientRect();
      const w = rect.width, h = rect.height;
      const cx = w * 0.5, cy = h * 0.5;
      const scale = Math.min(w, h) * 0.44;

      // cycle: 0..1 resolve, hold, 1..0 scatter — period ~7.2s
      const period = 7200;
      const tphase = ((now - start) % period) / period;
      let p: number;
      if (tphase < 0.42) p = ease(tphase / 0.42);
      else if (tphase < 0.58) p = 1;
      else p = ease(1 - (tphase - 0.58) / 0.42);

      const rot = (now - start) * 0.00006;
      const cosR = Math.cos(rot), sinR = Math.sin(rot);

      ctx.clearRect(0, 0, w, h);
      ctx.globalCompositeOperation = 'lighter';

      for (let i = 0; i < N; i++) {
        const pt = pts[i];
        // rotate the resolved spiral target slowly
        const rx = pt.tx * cosR - pt.ty * sinR;
        const ry = pt.tx * sinR + pt.ty * cosR;
        // jitter shrinks as we resolve
        const jitter = (1 - p) * 0.5;
        const jx = Math.sin(now * 0.0011 + pt.seed) * jitter;
        const jy = Math.cos(now * 0.0013 + pt.seed * 1.7) * jitter;
        const x = lerp(pt.nx, rx, p) + jx;
        const y = lerp(pt.ny, ry, p) + jy;
        const sx = cx + x * scale;
        const sy = cy + y * scale;

        const col = colors[i % 3];
        const rad = lerp(0.7, 1.8, p) + 0.6 * p;
        ctx.beginPath();
        ctx.fillStyle = col;
        ctx.globalAlpha = lerp(0.22, 0.9, p);
        ctx.arc(sx, sy, rad, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.globalAlpha = 1;
      ctx.globalCompositeOperation = 'source-over';
      raf = requestAnimationFrame(frame);
    };
    raf = requestAnimationFrame(frame);

    this.destroyRef.onDestroy(() => {
      running = false;
      cancelAnimationFrame(raf);
      window.removeEventListener('resize', onResize);
    });
  }
}
