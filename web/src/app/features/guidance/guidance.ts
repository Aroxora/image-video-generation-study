import {
  Component, ChangeDetectionStrategy, ElementRef, afterNextRender, inject, DestroyRef,
  viewChild, signal, computed, effect,
} from '@angular/core';
import { Chapter } from '../../shared/chapter';
import { CodeRef } from '../../shared/code-ref';
import { Math as MathTex } from '../../shared/math';
import { FirebaseService } from '../../core/firebase.service';

/**
 * Chapter — Classifier-free guidance: the prompt amplifier.
 *
 * Two figures, both driven by the same guidance scale w:
 *   1. Vector view: ε_uncond, ε_cond, and the guided ε = ε_uncond + w·(ε_cond − ε_uncond)
 *      drawn from the origin on a 2D plane; the guided vector extends along the
 *      (cond − uncond) direction as w grows. Animated via an eased current-w that
 *      chases the slider value, so the arrow glides.
 *   2. Effect illustration: a synthetic "sample" whose contrast/saturation are pushed
 *      as a function of w — a stand-in for the on-prompt-but-oversaturated trade-off.
 *
 * Code-refs quote classifier_free_guidance and drop_context from
 * pytorch/diffusion/guidance.py.
 */
@Component({
  selector: 'app-guidance',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [Chapter, CodeRef, MathTex],
  template: `
<app-chapter slug="guidance">

  <p class="lede">
    A diffusion model that has learned to follow prompts will, left to its own devices,
    follow them <em>politely</em> — often too politely. Classifier-free guidance is the
    one-line trick that turns up the volume: run the denoiser <strong>twice</strong> at
    every step, once with the prompt and once with nothing, and push the prediction
    <em>away</em> from the prompt-less guess. It is the single most important knob between
    "technically conditioned" and "unmistakably on-prompt" — and it costs you nothing but
    a second forward pass.
  </p>

  <h2>Two predictions, one extrapolation</h2>

  <p>
    A conditional diffusion model is a network <app-math expr="\\varepsilon_\\theta(x_t, t, c)" />
    that, given a noisy sample <app-math expr="x_t" />, a timestep <app-math expr="t" />, and a
    condition <app-math expr="c" /> (text embeddings), predicts the noise mixed in. The
    insight of Ho &amp; Salimans (2022) is to also let that <em>same</em> network run with no
    condition — the empty / null prompt — giving two predictions at every step:
  </p>

  <app-math display
    expr="\\varepsilon_{\\text{cond}} = \\varepsilon_\\theta(x_t, t, c), \\qquad \\varepsilon_{\\text{uncond}} = \\varepsilon_\\theta(x_t, t, \\varnothing)." />

  <p>
    Think of <app-math expr="\\varepsilon_{\\text{uncond}}" /> as "what any plausible image looks
    like" and <app-math expr="\\varepsilon_{\\text{cond}}" /> as "what <em>this prompt</em> wants."
    Guidance doesn't average them — it <strong>extrapolates</strong>. We start at the
    unconditional guess and walk past the conditional one, by a factor of the guidance scale
    <app-math expr="w" />:
  </p>

  <app-math display
    expr="\\varepsilon = \\varepsilon_{\\text{uncond}} + w\\,(\\varepsilon_{\\text{cond}} - \\varepsilon_{\\text{uncond}})." />

  <p>
    The difference vector <app-math expr="\\varepsilon_{\\text{cond}} - \\varepsilon_{\\text{uncond}}" />
    points "toward the prompt." Ho &amp; Salimans showed it is, up to a constant, the score of an
    implicit classifier <app-math expr="p(c\\mid x_t)" /> — so this <em>is</em> classifier guidance,
    minus the classifier. Hence "classifier-<strong>free</strong>." Drag <app-math expr="w" />
    below and watch the guided arrow shoot out along that direction.
  </p>

  <!-- FIGURE 1 — vector view -->
  <figure class="fig">
    <div class="fig__stage">
      <canvas #vec class="fig__canvas fig__canvas--sq"></canvas>
    </div>
    <div class="ctl">
      <div class="ctl__row">
        <label class="ctl__label" for="wslider">
          guidance scale <span class="mono">w = {{ fmt(wScale()) }}</span>
        </label>
        <input id="wslider" class="ctl__range" type="range" min="0" max="15" step="0.1"
          [value]="wScale()" (input)="onW($event)" />
      </div>
      <div class="ctl__readout">
        <span class="chip"><span class="swatch swatch--uncond"></span>ε<sub>uncond</sub></span>
        <span class="chip"><span class="swatch swatch--cond"></span>ε<sub>cond</sub></span>
        <span class="chip chip--accent"><span class="swatch swatch--guided"></span>guided ε</span>
        <span class="chip"><span class="chip__k">‖guided ε‖</span><span class="chip__v">{{ fmt(guidedNorm()) }}×</span></span>
      </div>
      <div class="ctl__row ctl__row--btns">
        <button type="button" class="btn" [class.btn--primary]="wScale() === 0" (click)="setW(0)">w = 0 (ignore)</button>
        <button type="button" class="btn" [class.btn--primary]="wScale() === 1" (click)="setW(1)">w = 1 (plain)</button>
        <button type="button" class="btn" [class.btn--primary]="wScale() === 7.5" (click)="setW(7.5)">w = 7.5 (SD)</button>
      </div>
    </div>
    <figcaption class="fig__cap">
      The guidance step as vector arithmetic. From the origin: <strong>ε<sub>uncond</sub></strong>
      (any image), <strong>ε<sub>cond</sub></strong> (the prompt's pull), and the
      <strong>guided ε</strong> = <app-math expr="\\varepsilon_{\\text{uncond}} + w(\\varepsilon_{\\text{cond}} - \\varepsilon_{\\text{uncond}})" />.
      As <app-math expr="w" /> grows the guided arrow extends along the dashed
      <app-math expr="\\varepsilon_{\\text{cond}} - \\varepsilon_{\\text{uncond}}" /> direction.
    </figcaption>
  </figure>

  <p>
    Read the special cases straight off the arrow. At <app-math expr="w = 0" /> the guided
    prediction collapses onto <app-math expr="\\varepsilon_{\\text{uncond}}" /> — the prompt is
    ignored entirely. At <app-math expr="w = 1" /> the difference term is added back exactly once
    and the guided vector lands precisely on <app-math expr="\\varepsilon_{\\text{cond}}" />:
    ordinary conditional sampling, no extra push. For <app-math expr="w > 1" /> we
    <em>over</em>-shoot past the conditional guess, and that overshoot is what makes the image
    snap to the prompt. Stable Diffusion typically lives around
    <app-math expr="w \\in [5, 12]" />.
  </p>

  <h3>How the code actually computes it</h3>

  <p>
    In the repo, <code>classifier_free_guidance</code> stacks the two contexts along the batch
    axis and runs <strong>one doubled-batch forward</strong> — exactly how production samplers
    keep CFG cheap — then splits the result and applies the extrapolation. The last line is the
    equation above, verbatim:
  </p>

  <app-code-ref
    file="pytorch/diffusion/guidance.py"
    lang="python"
    [code]="snipCfg"
    [lines]="[183, 212]"
    caption="classifier_free_guidance — one doubled-batch forward gives both ε's; the final line is ε_uncond + scale·(ε_cond − ε_uncond)." />

  <p>
    Two details worth noticing. The conditional and unconditional contexts are concatenated
    (<code>torch.cat([uncond_context, context], 0)</code>) so cross-attention and convolutions
    run <em>once</em> over a batch of size <app-math expr="2B" /> instead of twice — guidance
    roughly doubles the cost of a step, not more. And when <code>scale == 1.0</code> the function
    short-circuits to a single conditional forward, because the algebra collapses to
    <app-math expr="\\varepsilon_{\\text{cond}}" /> and the second pass would be wasted compute.
  </p>

  <h2>The price of pushing harder</h2>

  <p>
    Extrapolation is not free lunch. The same overshoot that makes an image more on-prompt also
    drags its statistics out of the distribution the model was trained on: high
    <app-math expr="w" /> tends to produce <strong>over-saturated, high-contrast</strong> images
    with less diversity across seeds. You are trading <em>variety</em> for <em>fidelity to the
    prompt</em>. The canvas below is an <strong>illustration</strong> of that trade-off, not a
    real denoiser: as you raise <app-math expr="w" /> it pushes the sample's contrast and
    saturation, the way real CFG does to color channels at large scales.
  </p>

  <!-- FIGURE 2 — effect illustration -->
  <figure class="fig">
    <div class="fig__stage">
      <canvas #eff class="fig__canvas fig__canvas--sq"></canvas>
    </div>
    <div class="ctl">
      <div class="ctl__row">
        <label class="ctl__label" for="w2slider">
          guidance scale <span class="mono">w = {{ fmt(wScale()) }}</span>
          <span class="ctl__tail">{{ effLabel() }}</span>
        </label>
        <input id="w2slider" class="ctl__range" type="range" min="0" max="15" step="0.1"
          [value]="wScale()" (input)="onW($event)" />
      </div>
      <p class="ctl__note">
        Illustration only — a fixed scene whose saturation &amp; contrast are pushed as a function
        of <app-math expr="w" />. A real model would also change <em>which</em> image it draws;
        here we hold the content fixed to isolate the over-saturation effect.
      </p>
    </div>
    <figcaption class="fig__cap">
      The quality / diversity trade-off, illustrated. Low <app-math expr="w" />: muted, "average,"
      faithful colors. High <app-math expr="w" />: punchy and on-prompt, but contrast and
      saturation blow out — the classic CFG artifact.
    </figcaption>
  </figure>

  <h2>One network, two jobs</h2>

  <p>
    There's a chicken-and-egg problem hiding here: where does
    <app-math expr="\\varepsilon_{\\text{uncond}}" /> come from? If the network has only ever seen
    real prompts, asking it to denoise with the <em>empty</em> prompt is undefined behaviour. The
    fix is built into training, not sampling. On roughly <strong>10%</strong> of training
    examples we simply <em>drop the prompt</em> — replace the condition with the null context and
    blank its attention mask — so the same weights learn to produce a sensible unconditional
    prediction too:
  </p>

  <app-code-ref
    file="pytorch/diffusion/guidance.py"
    lang="python"
    [code]="snipDrop"
    [lines]="[102, 122]"
    caption="drop_context — flip a per-row coin with probability p (~0.1); on heads, zero the context and mask out every token." />

  <p>
    That single hyperparameter <app-math expr="p \\approx 0.1" /> is what makes guidance possible.
    Each row of the batch gets an independent Bernoulli draw; on heads, that example's context is
    multiplied to zero and every token in its mask is flipped to padding, so the network trains on
    a genuine "no prompt" signal. After training, one model answers both questions —
    "what does this prompt want?" and "what does any image look like?" — and CFG simply subtracts
    the second from the first.
  </p>

  <h2>Where guidance fits</h2>

  <p>
    Keep the mental model straight. Guidance is <em>not</em> a separate generation pass and the
    prompt is <em>not</em> something the model emits alongside the image. The text is a continuous
    steering signal injected by cross-attention at <strong>every</strong> reverse step; CFG just
    amplifies how hard that signal pulls, by re-running the step once without it and extrapolating.
    The whole image (or, for video, the whole spacetime clip) is refined at once at each noise
    level — guidance is a per-step correction layered on top of that, not a left-to-right
    "next-frame" prediction. That global, all-at-once refinement is exactly why diffusion stays
    coherent, and guidance is the dial that decides how loudly the prompt gets to speak.
  </p>

</app-chapter>
  `,
  styles: [`
    :host { display: block; }
    .lede { font-size: var(--step-1); color: var(--ink-1); }
    .lede em, p em { color: var(--ink-0); font-style: italic; }
    code {
      font-family: var(--font-mono); font-size: 0.86em;
      background: var(--bg-3); border: 1px solid var(--line);
      border-radius: 6px; padding: 0.05em 0.4em; color: #cdbcff;
    }

    .fig {
      margin: 2rem 0;
      border: 1px solid var(--line);
      border-radius: var(--radius);
      background: var(--bg-1);
      box-shadow: var(--shadow-1);
      overflow: hidden;
    }
    .fig__stage {
      display: flex; justify-content: center; align-items: center;
      padding: 1.4rem; background:
        radial-gradient(120% 120% at 50% 0%, rgba(124,92,255,0.07), transparent 60%), var(--bg-0);
    }
    .fig__canvas { display: block; width: 100%; image-rendering: auto; }
    .fig__canvas--sq { max-width: 420px; aspect-ratio: 1 / 1; border-radius: var(--radius-sm); border: 1px solid var(--line-strong); }

    .ctl { padding: 1rem 1.1rem; border-top: 1px solid var(--line); display: grid; gap: 0.75rem; }
    .ctl__row { display: grid; gap: 0.5rem; }
    .ctl__row--btns { display: flex; flex-wrap: wrap; align-items: center; gap: 0.5rem; }
    .ctl__label { font-family: var(--font-mono); font-size: 0.8rem; color: var(--ink-2); }
    .ctl__tail { color: var(--ink-3); margin-left: 0.5rem; }
    .mono { font-family: var(--font-mono); color: var(--ink-0); }

    .ctl__range { -webkit-appearance: none; appearance: none; width: 100%; height: 6px;
      border-radius: 999px; background: linear-gradient(90deg, var(--plasma-a), var(--plasma-b));
      outline: none; cursor: pointer; }
    .ctl__range::-webkit-slider-thumb { -webkit-appearance: none; appearance: none;
      width: 18px; height: 18px; border-radius: 50%; background: #fff;
      border: 2px solid var(--plasma-a); box-shadow: 0 2px 8px rgba(0,0,0,0.5); cursor: pointer; }
    .ctl__range::-moz-range-thumb { width: 18px; height: 18px; border-radius: 50%; background: #fff;
      border: 2px solid var(--plasma-a); cursor: pointer; }

    .ctl__readout { display: flex; flex-wrap: wrap; gap: 0.5rem; }
    .chip {
      display: inline-flex; align-items: center; gap: 0.4rem;
      font-family: var(--font-mono); font-size: 0.74rem;
      padding: 0.3em 0.65em; border-radius: 999px;
      border: 1px solid var(--line); background: var(--bg-2); color: var(--ink-1);
    }
    .chip__k { color: var(--ink-3); }
    .chip__v { color: var(--ink-0); }
    .chip--accent { border-color: rgba(124,92,255,0.4); background: var(--accent-soft); }
    .swatch { width: 12px; height: 3px; border-radius: 2px; display: inline-block; }
    .swatch--uncond { background: #8a93a6; }
    .swatch--cond { background: var(--plasma-b); }
    .swatch--guided { background: var(--plasma-c); }

    .ctl__note { margin: 0; font-size: 0.8rem; color: var(--ink-2); }
    .fig__cap {
      padding: 0.75rem 1.1rem; border-top: 1px solid var(--line);
      font-size: 0.84rem; color: var(--ink-2); background: rgba(255,255,255,0.012);
    }
    .fig__cap strong { color: var(--ink-1); }
  `],
})
export class Guidance {
  private readonly fb = inject(FirebaseService);
  private readonly destroyRef = inject(DestroyRef);

  // ---- reactive control: guidance scale w in [0, 15] ----
  readonly wScale = signal(7.5);

  // animated value that eases toward wScale() so the vector glides
  private readonly wAnim = signal(7.5);

  // ---- canvases ----
  private readonly vecCv = viewChild.required<ElementRef<HTMLCanvasElement>>('vec');
  private readonly effCv = viewChild.required<ElementRef<HTMLCanvasElement>>('eff');

  // ---- fixed epsilon vectors (in a unit-ish "noise plane"), chosen so the
  //      difference (cond - uncond) is a clear, mostly-horizontal direction ----
  private readonly epsUncond = { x: -0.55, y: 0.62 };
  private readonly epsCond = { x: 0.78, y: 0.20 };

  // norm of the guided vector relative to the conditional one, for the readout
  readonly guidedNorm = computed(() => {
    const w = this.wScale();
    const gx = this.epsUncond.x + w * (this.epsCond.x - this.epsUncond.x);
    const gy = this.epsUncond.y + w * (this.epsCond.y - this.epsUncond.y);
    const cn = Math.hypot(this.epsCond.x, this.epsCond.y) || 1;
    return Math.hypot(gx, gy) / cn;
  });

  readonly effLabel = computed(() => {
    const w = this.wScale();
    if (w < 0.5) return '· prompt ignored';
    if (w <= 1.05) return '· plain conditional';
    if (w < 6) return '· balanced';
    if (w < 11) return '· on-prompt';
    return '· over-saturated';
  });

  // ---- scene for figure 2 (a small RGB field in [0,1]) ----
  private readonly GRID = 96;
  private base!: Float32Array;

  constructor() {
    this.base = this.makeScene(this.GRID);

    afterNextRender(() => {
      this.startVectorLoop();
      this.drawEffect();
    });

    // figure 2 is static per-w: redraw whenever the slider changes
    effect(() => { this.wScale(); this.drawEffect(); });
  }

  // ====================================================================
  // FIGURE 1 — vector view (animated via rAF, draws directly to canvas)
  // ====================================================================
  private startVectorLoop(): void {
    const cv = this.vecCv().nativeElement;
    const ctx = cv.getContext('2d');
    if (!ctx) return;

    const resize = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const r = cv.getBoundingClientRect();
      cv.width = Math.floor(r.width * dpr);
      cv.height = Math.floor(r.height * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    resize();
    window.addEventListener('resize', resize);

    let raf = 0;
    const loop = () => {
      // ease the animated w toward the target
      const target = this.wScale();
      const cur = this.wAnim();
      const next = cur + (target - cur) * 0.16;
      this.wAnim.set(Math.abs(next - target) < 1e-3 ? target : next);
      this.drawVector(ctx, cv);
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);

    this.destroyRef.onDestroy(() => {
      cancelAnimationFrame(raf);
      window.removeEventListener('resize', resize);
    });
  }

  private drawVector(ctx: CanvasRenderingContext2D, cv: HTMLCanvasElement): void {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = cv.width / dpr, h = cv.height / dpr;
    ctx.clearRect(0, 0, w, h);

    // origin near lower-left; scale so vectors fit at high w (clamp visual length)
    const ox = w * 0.30, oy = h * 0.72;
    const scale = Math.min(w, h) * 0.34;

    // grid backdrop
    ctx.strokeStyle = 'rgba(255,255,255,0.05)';
    ctx.lineWidth = 1;
    const gstep = scale * 0.5;
    for (let gx = ox % gstep; gx < w; gx += gstep) {
      ctx.beginPath(); ctx.moveTo(gx, 0); ctx.lineTo(gx, h); ctx.stroke();
    }
    for (let gy = oy % gstep; gy < h; gy += gstep) {
      ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(w, gy); ctx.stroke();
    }

    // axes through origin
    ctx.strokeStyle = 'rgba(255,255,255,0.16)';
    ctx.beginPath(); ctx.moveTo(0, oy); ctx.lineTo(w, oy); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(ox, 0); ctx.lineTo(ox, h); ctx.stroke();

    const u = this.epsUncond, c = this.epsCond;
    const wv = this.wAnim();
    // guided vector
    const g = { x: u.x + wv * (c.x - u.x), y: u.y + wv * (c.y - u.y) };

    // clamp the guided arrow's drawn length so it stays on-canvas, but keep direction
    const maxLen = Math.min(w, h) * 0.46;
    const gLen = Math.hypot(g.x, g.y) * scale;
    const clamp = gLen > maxLen ? maxLen / gLen : 1;

    // map noise-plane coords to screen (y up)
    const P = (v: { x: number; y: number }, k = 1) => ({
      x: ox + v.x * scale * k,
      y: oy - v.y * scale * k,
    });

    // dashed (cond - uncond) direction line from uncond tip, the extrapolation axis
    const pu = P(u), pc = P(c), pg = P(g, clamp);
    ctx.save();
    ctx.strokeStyle = 'rgba(255,255,255,0.22)';
    ctx.setLineDash([5, 6]);
    ctx.lineWidth = 1.4;
    // extend the line well past for context
    const dirx = c.x - u.x, diry = c.y - u.y;
    const far = P({ x: u.x + dirx * 16, y: u.y + diry * 16 });
    const near = P({ x: u.x - dirx * 2, y: u.y - diry * 2 });
    ctx.beginPath(); ctx.moveTo(near.x, near.y); ctx.lineTo(far.x, far.y); ctx.stroke();
    ctx.restore();

    // arrows
    this.arrow(ctx, ox, oy, pu.x, pu.y, '#8a93a6', 'ε_uncond');
    this.arrow(ctx, ox, oy, pc.x, pc.y, '#41d6ff', 'ε_cond');
    this.arrow(ctx, ox, oy, pg.x, pg.y, '#ff5c8a', `guided ε  (w=${this.fmt(this.wScale())})`, true);

    // origin dot
    ctx.fillStyle = '#fff';
    ctx.beginPath(); ctx.arc(ox, oy, 3, 0, Math.PI * 2); ctx.fill();
  }

  private arrow(
    ctx: CanvasRenderingContext2D,
    x0: number, y0: number, x1: number, y1: number,
    color: string, label: string, bold = false,
  ): void {
    const ang = Math.atan2(y1 - y0, x1 - x0);
    const head = 11;
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = bold ? 3 : 2;
    ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
    // arrowhead
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x1 - head * Math.cos(ang - 0.4), y1 - head * Math.sin(ang - 0.4));
    ctx.lineTo(x1 - head * Math.cos(ang + 0.4), y1 - head * Math.sin(ang + 0.4));
    ctx.closePath();
    ctx.fill();
    // label
    ctx.font = `${bold ? '600 ' : ''}12px ui-monospace, SFMono-Regular, monospace`;
    ctx.fillStyle = color;
    const lx = x1 + Math.cos(ang) * 8 + (Math.cos(ang) < 0 ? -ctx.measureText(label).width : 6);
    const ly = y1 + Math.sin(ang) * 8 + (Math.sin(ang) > 0 ? 14 : -6);
    ctx.fillText(label, lx, ly);
  }

  // ====================================================================
  // FIGURE 2 — effect illustration (static per w)
  // ====================================================================
  private drawEffect(): void {
    const cv = this.effCv().nativeElement;
    const ctx = cv.getContext('2d');
    if (!ctx) return;
    const n = this.GRID;
    if (cv.width !== n) { cv.width = n; cv.height = n; }

    const w = this.wScale();
    // saturation gain rises with w; contrast gain rises with w.
    // w=1 -> identity-ish; high w -> exaggerated.
    const sat = 1 + 0.16 * Math.max(0, w - 1);
    const con = 1 + 0.085 * Math.max(0, w - 1);

    const img = ctx.createImageData(n, n);
    const d = img.data;
    for (let p = 0; p < n * n; p++) {
      const j = p * 3, k = p * 4;
      let r = this.base[j], gC = this.base[j + 1], b = this.base[j + 2];
      // luma for saturation push
      const lum = 0.299 * r + 0.587 * gC + 0.114 * b;
      r = lum + (r - lum) * sat;
      gC = lum + (gC - lum) * sat;
      b = lum + (b - lum) * sat;
      // contrast around mid-gray
      r = 0.5 + (r - 0.5) * con;
      gC = 0.5 + (gC - 0.5) * con;
      b = 0.5 + (b - 0.5) * con;
      d[k] = Math.max(0, Math.min(255, r * 255));
      d[k + 1] = Math.max(0, Math.min(255, gC * 255));
      d[k + 2] = Math.max(0, Math.min(255, b * 255));
      d[k + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }

  /** Hand-drawn dusk scene: sky gradient, sun, two hills. RGB in [0,1]. */
  private makeScene(n: number): Float32Array {
    const buf = new Float32Array(n * n * 3);
    const sunX = n * 0.68, sunY = n * 0.32, sunR = n * 0.11;
    const hill1Y = n * 0.6, hill1A = n * 0.16;
    const hill2Y = n * 0.78, hill2A = n * 0.12;
    for (let y = 0; y < n; y++) {
      for (let x = 0; x < n; x++) {
        const i = (y * n + x) * 3;
        const fy = y / n;
        let r = 0.20 + 0.5 * fy, g = 0.14 + 0.3 * fy, b = 0.42 - 0.16 * fy;
        const dx = x - sunX, dy = y - sunY;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < sunR) { r = 0.98; g = 0.82; b = 0.5; }
        else {
          const glow = Math.max(0, 1 - (dist - sunR) / (n * 0.32));
          r += glow * 0.46; g += glow * 0.3; b += glow * 0.1;
        }
        const ridge1 = hill1Y - Math.sin((x / n) * Math.PI * 1.4) * hill1A;
        const ridge2 = hill2Y - Math.sin((x / n) * Math.PI * 2.1 + 1.0) * hill2A;
        if (y > ridge2) { r = 0.07; g = 0.21; b = 0.16; }
        else if (y > ridge1) { r = 0.11; g = 0.29; b = 0.22; }
        buf[i] = Math.min(1, r); buf[i + 1] = Math.min(1, g); buf[i + 2] = Math.min(1, b);
      }
    }
    return buf;
  }

  // ====================================================================
  // control handlers (zoneless: .set / .update only)
  // ====================================================================
  onW(ev: Event): void {
    const v = +(ev.target as HTMLInputElement).value;
    this.wScale.set(v);
    this.fb.event('interact', { section: 'guidance', control: 'w-slider', value: v });
  }
  setW(v: number): void {
    this.wScale.set(v);
    this.fb.event('interact', { section: 'guidance', control: 'w-preset', value: v });
  }

  // ---- formatting ----
  fmt(v: number): string {
    return Number.isInteger(v) ? v.toFixed(0) : v.toFixed(1);
  }

  // ---- code snippets quoted verbatim from pytorch/diffusion/guidance.py ----
  readonly snipCfg = `# --- Try the cheap path: one doubled-batch forward. ---------------------
# This works when the conditional and unconditional contexts share the same
# sequence length L so we can concatenate along the batch axis.
can_batch = uncond_context.shape[1] == L
if can_batch:
    x_in = torch.cat([x_t, x_t], dim=0)              # [2B, ...]
    t_in = torch.cat([t, t], dim=0)                  # [2B]
    c_in = torch.cat([uncond_context, context], 0)   # [2B, L, dim]
    # ...
    eps = model(x_in, t_in, context=c_in, mask=m_in)
    eps_uncond, eps_cond = eps[:B], eps[B:]
else:
    # Fallback: differing sequence lengths -> two separate forwards.
    eps_uncond = model(x_t, t, context=uncond_context, mask=uncond_mask)
    eps_cond = model(x_t, t, context=context, mask=mask)

# The guidance step: extrapolate from "any image" toward "this prompt".
return eps_uncond + scale * (eps_cond - eps_uncond)`;

  readonly snipDrop = `if context is None or p <= 0.0:
    return context, mask

B = context.shape[0]
# One Bernoulli draw per batch row. ...
coin = torch.rand(B, generator=generator).to(context.device)
drop = coin < p  # [B] bool: True where we wipe the prompt

if not bool(drop.any()):
    return context, mask

# Broadcast the per-row decision over (L, dim) and zero those rows.
keep = (~drop).to(context.dtype).view(B, *([1] * (context.dim() - 1)))
context = context * keep

if mask is not None:
    # Where we dropped the prompt, ALL tokens become padding (False).
    mask = mask & (~drop).view(B, *([1] * (mask.dim() - 1)))

return context, mask`;
}
