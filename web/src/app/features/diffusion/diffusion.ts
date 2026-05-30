import {
  Component, ChangeDetectionStrategy, ElementRef, afterNextRender, inject, DestroyRef,
  viewChild, signal, computed, effect,
} from '@angular/core';
import { Chapter } from '../../shared/chapter';
import { CodeRef } from '../../shared/code-ref';
import { Math as MathTex } from '../../shared/math';
import { FirebaseService } from '../../core/firebase.service';

/**
 * Chapter 01 — Forward & reverse: sculpting an image out of static.
 *
 * Three figures, all driven by the SAME cosine/linear schedule we recompute in
 * JS so the page matches `pytorch/diffusion/schedule.py` exactly:
 *   1. forward q(x_t|x_0): a slider over t dissolves a hand-drawn scene into noise;
 *   2. schedule visualizer: ᾱ_t / β_t / SNR for cosine vs linear;
 *   3. reverse (illustrative): replay the forward frames backwards to convey
 *      "predict noise, subtract, repeat", then send readers to the live playground.
 */
@Component({
  selector: 'app-diffusion',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [Chapter, CodeRef, MathTex],
  template: `
<app-chapter slug="diffusion">

  <p class="lede">
    A diffusion model never paints a picture stroke by stroke. It learns to do
    one small, dumb-sounding thing — <em>look at a noisy image and guess the
    noise</em> — and then does it a few hundred times, starting from pure static.
    To understand the reverse trick you first have to watch the <strong>forward</strong>
    process that defines it: a fixed recipe for turning any image <em>into</em> static.
    No network. No learning. Just controlled vandalism with Gaussian noise.
  </p>

  <h2>The forward process is pure math</h2>

  <p>
    Pick a clean image <app-math expr="x_0" />. The forward process adds a touch of
    Gaussian noise, then a touch more, and a touch more, for <app-math expr="T" /> steps.
    Each step is tiny — <app-math expr="q(x_t\\mid x_{t-1}) = \\mathcal{N}\\!\\big(x_t;\\,\\sqrt{1-\\beta_t}\\,x_{t-1},\\,\\beta_t \\mathbf{I}\\big)" /> —
    where <app-math expr="\\beta_t" /> is a fixed <em>schedule</em> that says how much
    fresh noise to inject at step <app-math expr="t" />. After enough steps the image
    is indistinguishable from <app-math expr="\\mathcal{N}(0,\\mathbf{I})" />: snow.
  </p>

  <p>
    The crucial property: because every step is Gaussian and the schedule is fixed,
    you can <strong>skip straight to step <app-math expr="t" /></strong> in closed form.
    Let <app-math expr="\\alpha_t = 1-\\beta_t" /> and let
    <app-math expr="\\bar\\alpha_t = \\prod_{s\\le t}\\alpha_s" /> be the total signal
    that survives to step <app-math expr="t" />. Then
  </p>

  <app-math display
    expr="x_t = \\sqrt{\\bar\\alpha_t}\\,x_0 \\;+\\; \\sqrt{1-\\bar\\alpha_t}\\,\\varepsilon,\\qquad \\varepsilon\\sim\\mathcal{N}(0,\\mathbf{I})." />

  <p>
    That single affine blend is the entire forward process. The figure below is
    <em>literally</em> that equation, evaluated per pixel: drag <app-math expr="t" />
    and watch <app-math expr="\\sqrt{\\bar\\alpha_t}" /> fade the scene out while
    <app-math expr="\\sqrt{1-\\bar\\alpha_t}" /> fades the static in.
  </p>

  <!-- FIGURE 1 — forward process -->
  <figure class="fig">
    <div class="fig__stage">
      <canvas #fwd class="fig__canvas fig__canvas--sq"></canvas>
    </div>
    <div class="ctl">
      <div class="ctl__row">
        <label class="ctl__label" for="tslider">
          step <span class="mono">t = {{ tStep() }}</span> / {{ T }}
        </label>
        <input id="tslider" class="ctl__range" type="range" min="0" [max]="T"
          [value]="tStep()" (input)="onT($event)" />
      </div>
      <div class="ctl__readout">
        <span class="chip"><span class="chip__k">ᾱ<sub>t</sub></span><span class="chip__v">{{ fmt(alphaBarAtT()) }}</span></span>
        <span class="chip"><span class="chip__k">√ᾱ<sub>t</sub></span><span class="chip__v">{{ fmt(Math.sqrt(alphaBarAtT())) }}</span></span>
        <span class="chip"><span class="chip__k">√(1−ᾱ<sub>t</sub>)</span><span class="chip__v">{{ fmt(Math.sqrt(1 - alphaBarAtT())) }}</span></span>
        <span class="chip chip--accent"><span class="chip__k">SNR</span><span class="chip__v">{{ fmtSnr(snrAtT()) }}</span></span>
      </div>
      <div class="ctl__row ctl__row--btns">
        <button type="button" class="btn" [class.btn--primary]="schedule() === 'cosine'" (click)="setSchedule('cosine')">cosine</button>
        <button type="button" class="btn" [class.btn--primary]="schedule() === 'linear'" (click)="setSchedule('linear')">linear</button>
        <button type="button" class="btn" (click)="reseed()">reseed ε</button>
      </div>
    </div>
    <figcaption class="fig__cap">
      Forward diffusion on a fixed scene. The slider is the timestep <app-math expr="t" />;
      each pixel becomes <app-math expr="\\sqrt{\\bar\\alpha_t}\\,x_0+\\sqrt{1-\\bar\\alpha_t}\\,\\varepsilon" />.
      The signal-to-noise ratio is <app-math expr="\\mathrm{SNR}=\\bar\\alpha_t/(1-\\bar\\alpha_t)" />.
    </figcaption>
  </figure>

  <p>
    Notice the readouts. At <app-math expr="t=0" /> the SNR is enormous (all signal);
    by the end it collapses toward zero. The whole game of training is: hand the
    network an <app-math expr="x_t" /> from some random <app-math expr="t" /> and ask
    it to name the <app-math expr="\\varepsilon" /> that was mixed in. That is the
    only target it ever sees.
  </p>

  <p>
    In code, the forward jump is a four-line method. We precompute
    <app-math expr="\\sqrt{\\bar\\alpha_t}" /> and <app-math expr="\\sqrt{1-\\bar\\alpha_t}" />
    once, then <code>q_sample</code> just indexes and blends — exactly what the
    canvas does:
  </p>

  <app-code-ref
    file="pytorch/diffusion/schedule.py"
    lang="python"
    [code]="snipQSample"
    [lines]="[204, 220]"
    caption="NoiseSchedule.q_sample — the forward process in one closed-form step. No model, no gradients." />

  <p>
    There is genuinely no learning here. <code>noise</code> is the regression target,
    not an output. This is why diffusion training is cheap: you never simulate the
    <app-math expr="T" />-step chain, you teleport to a random <app-math expr="t" />
    and backprop a single MSE.
  </p>

  <h2>The schedule decides where the steps go</h2>

  <p>
    The <app-math expr="\\beta_t" /> schedule is the one design knob of the forward
    process, and it matters more than it looks. The original DDPM used a
    <strong>linear</strong> ramp of <app-math expr="\\beta_t" />. The trouble: at high
    resolution that destroys the image too quickly — the last fifth of steps are
    already near-pure noise, so the network wastes capacity on timesteps that carry
    almost no information.
  </p>

  <p>
    Nichol &amp; Dhariwal's <strong>cosine</strong> schedule fixes this by defining the
    survival curve <app-math expr="\\bar\\alpha_t" /> directly with a cosine and
    <em>deriving</em> <app-math expr="\\beta_t" /> from it:
  </p>

  <app-math display
    expr="\\bar\\alpha_t = \\frac{f(t)}{f(0)},\\quad f(t)=\\cos^2\\!\\left(\\frac{t/T + s}{1+s}\\cdot\\frac{\\pi}{2}\\right),\\quad \\beta_t = 1-\\frac{\\bar\\alpha_t}{\\bar\\alpha_{t-1}}." />

  <p>
    The small offset <app-math expr="s=0.008" /> stops <app-math expr="\\beta_t" /> from
    vanishing near <app-math expr="t=0" />. The payoff: signal decays gently through
    the middle steps instead of falling off a cliff, so more steps land where the
    network can actually learn something. Toggle the curves below.
  </p>

  <!-- FIGURE 2 — schedule visualizer -->
  <figure class="fig">
    <div class="fig__stage fig__stage--chart">
      <canvas #chart class="fig__canvas fig__canvas--wide"></canvas>
    </div>
    <div class="ctl">
      <div class="ctl__row ctl__row--btns">
        <span class="ctl__group">curve:</span>
        <button type="button" class="btn" [class.btn--primary]="curve() === 'alphaBar'" (click)="setCurve('alphaBar')">ᾱ<sub>t</sub></button>
        <button type="button" class="btn" [class.btn--primary]="curve() === 'beta'" (click)="setCurve('beta')">β<sub>t</sub></button>
        <button type="button" class="btn" [class.btn--primary]="curve() === 'snr'" (click)="setCurve('snr')">log SNR</button>
      </div>
      <div class="ctl__readout ctl__readout--legend">
        <span class="chip"><span class="swatch swatch--cos"></span>cosine</span>
        <span class="chip"><span class="swatch swatch--lin"></span>linear</span>
        <span class="chip chip--ghost">marker = current t</span>
      </div>
    </div>
    <figcaption class="fig__cap">
      Schedule curves vs <app-math expr="t" />. <strong>ᾱ<sub>t</sub></strong> is surviving
      signal, <strong>β<sub>t</sub></strong> is per-step noise, <strong>log SNR</strong> is
      <app-math expr="\\log\\!\\big(\\bar\\alpha_t/(1-\\bar\\alpha_t)\\big)" />. The vertical
      marker tracks the same <app-math expr="t" /> as Figure 1.
    </figcaption>
  </figure>

  <p>
    Watch the cosine <app-math expr="\\bar\\alpha_t" /> curve sit higher than linear
    across the whole middle of the range: at the same <app-math expr="t" />, cosine
    has kept more signal. On the <strong>log SNR</strong> view the cosine line is a
    near-straight descent, which is exactly the property that makes timestep weighting
    well-behaved — every step contributes a comparable slice of difficulty.
  </p>

  <p>The schedule is built once, from <code>betas</code>, via <code>torch.cumprod</code>:</p>

  <app-code-ref
    file="pytorch/diffusion/schedule.py"
    lang="python"
    [code]="snipCosine"
    [lines]="[75, 88]"
    caption="make_beta_schedule (cosine branch) — define ᾱ_t with a cosine, then read off β_t from neighboring ratios." />

  <h2>Reverse: predict the noise, subtract, repeat</h2>

  <p>
    Everything above is fixed. The <strong>only</strong> learned object in a diffusion
    model is a network <app-math expr="\\varepsilon_\\theta(x_t, t)" /> that, given a
    noisy <app-math expr="x_t" /> and the timestep, predicts the noise inside it.
    Training is one line of intent: minimize
    <app-math expr="\\lVert \\varepsilon - \\varepsilon_\\theta(x_t,t)\\rVert^2" />.
  </p>

  <p>
    Sampling runs that network backwards. Start from pure static
    <app-math expr="x_T\\sim\\mathcal{N}(0,\\mathbf{I})" />. At each step, predict the
    noise, back out an estimate of the clean image, and take one small step toward it
    (adding a pinch of fresh noise except at the very end). The ancestral DDPM update is
  </p>

  <app-math display
    expr="x_{t-1} = \\tilde\\mu_\\theta(x_t,t) + \\sigma_t z,\\qquad \\hat x_0 = \\sqrt{\\tfrac{1}{\\bar\\alpha_t}}\\,x_t - \\sqrt{\\tfrac{1}{\\bar\\alpha_t}-1}\\,\\varepsilon_\\theta,\\qquad z\\sim\\mathcal{N}(0,\\mathbf{I})." />

  <p>
    where <app-math expr="\\tilde\\mu_\\theta" /> is the posterior mean built from the
    predicted <app-math expr="\\hat x_0" />. The strip below is <strong>illustrative</strong>:
    it replays Figure 1's forward frames in reverse to convey the <em>shape</em> of the
    process — static condensing into structure — but it is <em>not</em> a trained
    network. A real denoiser hallucinates new detail at each step; here we are just
    rewinding. For the genuine learned reverse loop, see the
    <a class="link" href="/playground">live playground ↗</a>.
  </p>

  <!-- FIGURE 3 — reverse (illustrative) -->
  <figure class="fig">
    <div class="fig__stage">
      <canvas #rev class="fig__canvas fig__canvas--sq"></canvas>
    </div>
    <div class="ctl">
      <div class="ctl__row ctl__row--btns">
        <button type="button" class="btn btn--primary" (click)="toggleReverse()">
          {{ revPlaying() ? '❚❚ pause' : '▶ play reverse' }}
        </button>
        <span class="ctl__group">denoise step <span class="mono">{{ revLabel() }}</span></span>
      </div>
      <p class="ctl__note">
        Illustrative rewind, not a trained model. Each visible step stands in for
        “predict <app-math expr="\\varepsilon_\\theta" />, form <app-math expr="\\hat x_0" />, step toward it.”
      </p>
    </div>
    <figcaption class="fig__cap">
      Reverse process, conceptually: from static at <app-math expr="t=T" />, repeatedly
      estimate and remove noise until a clean <app-math expr="x_0" /> remains.
    </figcaption>
  </figure>

  <p>
    One reverse step, faithfully, is <code>p_sample</code>. Read the four numbered
    moves in the docstring against the equation above — predict <app-math expr="\\varepsilon" />,
    recover <app-math expr="\\hat x_0" />, form the posterior, add noise unless
    <app-math expr="t=0" />:
  </p>

  <app-code-ref
    file="pytorch/diffusion/ddpm.py"
    lang="python"
    [code]="snipPSample"
    [lines]="[173, 183]"
    caption="GaussianDiffusion.p_sample — one ancestral reverse step. The network's only job is the eps in line one." />

  <h2>Why this is not “next-frame prediction”</h2>

  <p>
    A common misconception imagines diffusion guessing pixels left-to-right, or frames
    one after the next. It does not. The denoiser sees the <strong>entire</strong> image
    (or, for video, the whole clip) at every step and refines all of it at once — the
    iteration is over <em>noise level</em>, not over space or time. That global, all-at-once
    refinement is exactly why diffusion gives coherent results, and it is the opposite of
    the autoregressive “predict the next token” family covered later in this site.
  </p>

  <p>
    Two more facts that the rest of these chapters build on: Stable Diffusion runs this
    whole loop inside a small <strong>VAE latent</strong>, not in pixels, which is what
    made it affordable; and a text prompt enters not as a starting frame but as a
    continuous steering signal injected by <strong>cross-attention</strong> at every
    reverse step. The static you started from never knew what it would become — the
    schedule and the network sculpt it, step by step, out of noise.
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
    .link { color: var(--plasma-b); text-decoration: none; border-bottom: 1px solid rgba(65,214,255,0.4); }
    .link:hover { color: #fff; }

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
    .fig__stage--chart { padding: 1.1rem 1rem 0.6rem; }
    .fig__canvas { display: block; width: 100%; image-rendering: auto; }
    .fig__canvas--sq { max-width: 380px; aspect-ratio: 1 / 1; border-radius: var(--radius-sm); border: 1px solid var(--line-strong); }
    .fig__canvas--wide { max-width: 720px; aspect-ratio: 16 / 7; border-radius: var(--radius-sm); }

    .ctl { padding: 1rem 1.1rem; border-top: 1px solid var(--line); display: grid; gap: 0.75rem; }
    .ctl__row { display: grid; gap: 0.5rem; }
    .ctl__row--btns { display: flex; flex-wrap: wrap; align-items: center; gap: 0.5rem; }
    .ctl__label { font-family: var(--font-mono); font-size: 0.8rem; color: var(--ink-2); }
    .ctl__group { font-family: var(--font-mono); font-size: 0.78rem; color: var(--ink-3); }
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
    .ctl__readout--legend { font-family: var(--font-mono); font-size: 0.74rem; }
    .chip {
      display: inline-flex; align-items: center; gap: 0.4rem;
      font-family: var(--font-mono); font-size: 0.74rem;
      padding: 0.3em 0.65em; border-radius: 999px;
      border: 1px solid var(--line); background: var(--bg-2); color: var(--ink-1);
    }
    .chip__k { color: var(--ink-3); }
    .chip__v { color: var(--ink-0); }
    .chip--accent { border-color: rgba(124,92,255,0.4); background: var(--accent-soft); }
    .chip--accent .chip__v { color: #cdbcff; }
    .chip--ghost { color: var(--ink-3); border-style: dashed; }
    .swatch { width: 12px; height: 3px; border-radius: 2px; display: inline-block; }
    .swatch--cos { background: var(--plasma-b); }
    .swatch--lin { background: var(--plasma-c); }

    .ctl__note { margin: 0; font-size: 0.8rem; color: var(--ink-2); }
    .fig__cap {
      padding: 0.75rem 1.1rem; border-top: 1px solid var(--line);
      font-size: 0.84rem; color: var(--ink-2); background: rgba(255,255,255,0.012);
    }
    .fig__cap strong { color: var(--ink-1); }
  `],
})
export class Diffusion {
  // expose Math to the template for the numeric readouts
  protected readonly Math = Math;

  private readonly fb = inject(FirebaseService);
  private readonly destroyRef = inject(DestroyRef);

  // ---- schedule parameters (kept in lockstep with schedule.py) ----
  readonly T = 1000;
  private readonly GRID = 96;            // forward image resolution (px per side)
  private readonly s = 0.008;            // cosine offset, matches make_beta_schedule

  // ---- reactive controls ----
  readonly tStep = signal(0);
  readonly schedule = signal<'cosine' | 'linear'>('cosine');
  readonly curve = signal<'alphaBar' | 'beta' | 'snr'>('alphaBar');
  readonly revPlaying = signal(false);
  readonly revStep = signal(this.T);     // illustrative reverse position (T..0)
  private seed = signal(1);              // bump to resample epsilon

  // ---- canvases ----
  private readonly fwdCv = viewChild.required<ElementRef<HTMLCanvasElement>>('fwd');
  private readonly chartCv = viewChild.required<ElementRef<HTMLCanvasElement>>('chart');
  private readonly revCv = viewChild.required<ElementRef<HTMLCanvasElement>>('rev');

  // ---- precomputed ᾱ tables for both schedules (length T+1, index = t) ----
  private readonly cosTable = this.buildAlphaBar('cosine');
  private readonly linTable = this.buildAlphaBar('linear');

  // ---- per-pixel fields for the forward scene ----
  private base!: Float32Array;   // x0 in [0,1], length GRID*GRID*3
  private noise!: Float32Array;  // epsilon ~ N(0,1) per channel

  // ---- derived (computed) values shown in the template ----
  private readonly table = computed(() => (this.schedule() === 'cosine' ? this.cosTable : this.linTable));
  readonly alphaBarAtT = computed(() => this.table()[this.tStep()]);
  readonly snrAtT = computed(() => {
    const ab = this.alphaBarAtT();
    return ab / Math.max(1e-12, 1 - ab);
  });
  readonly revLabel = computed(() => {
    const t = this.revStep();
    return t >= this.T ? `${this.T} (static)` : t <= 0 ? '0 (clean)' : `${t}`;
  });

  constructor() {
    this.base = this.makeScene(this.GRID);
    this.noise = this.makeNoise(this.GRID, this.seed());

    afterNextRender(() => {
      this.drawForward();
      this.drawChart();
      this.drawReverse();
      this.startReverseLoop();
    });

    // redraw forward whenever t / schedule / seed change
    effect(() => { this.tStep(); this.schedule(); this.seed(); this.drawForward(); });
    // redraw chart whenever curve / schedule / t change (marker tracks t)
    effect(() => { this.curve(); this.schedule(); this.tStep(); this.drawChart(); });
    // redraw reverse strip whenever its step changes (loop drives this)
    effect(() => { this.revStep(); this.seed(); this.drawReverse(); });
  }

  // ====================================================================
  // schedule math (mirrors pytorch/diffusion/schedule.py)
  // ====================================================================
  /** ᾱ_t table of length T+1 (index t in [0, T]) for either schedule. */
  private buildAlphaBar(kind: 'cosine' | 'linear'): Float64Array {
    const T = this.T;
    const out = new Float64Array(T + 1);
    if (kind === 'cosine') {
      const f0 = Math.cos((this.s / (1 + this.s)) * Math.PI * 0.5) ** 2;
      for (let t = 0; t <= T; t++) {
        const f = Math.cos(((t / T + this.s) / (1 + this.s)) * Math.PI * 0.5) ** 2;
        out[t] = f / f0;
      }
    } else {
      // linear betas in [1e-4, 2e-2]; ᾱ_t = prod(1 - beta_s)
      const bStart = 1e-4, bEnd = 2e-2;
      out[0] = 1;
      let acc = 1;
      for (let t = 1; t <= T; t++) {
        const beta = bStart + ((bEnd - bStart) * (t - 1)) / (T - 1);
        acc *= 1 - beta;
        out[t] = acc;
      }
    }
    return out;
  }

  // ====================================================================
  // scene + noise generation
  // ====================================================================
  /** Hand-drawn scene: sky gradient, sun, two hills. Returns RGB in [0,1]. */
  private makeScene(n: number): Float32Array {
    const buf = new Float32Array(n * n * 3);
    const sunX = n * 0.7, sunY = n * 0.3, sunR = n * 0.12;
    const hill1Y = n * 0.62, hill1A = n * 0.18;
    const hill2Y = n * 0.78, hill2A = n * 0.12;
    for (let y = 0; y < n; y++) {
      for (let x = 0; x < n; x++) {
        const i = (y * n + x) * 3;
        const fy = y / n;
        // sky: dusk gradient, indigo -> warm
        let r = 0.18 + 0.55 * fy, g = 0.12 + 0.32 * fy, b = 0.42 - 0.18 * fy;
        // sun glow + disc
        const dx = x - sunX, dy = y - sunY;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < sunR) { r = 1.0; g = 0.86; b = 0.5; }
        else {
          const glow = Math.max(0, 1 - (dist - sunR) / (n * 0.34));
          r += glow * 0.5; g += glow * 0.34; b += glow * 0.12;
        }
        // hills (sine ridges), front hill darker
        const ridge1 = hill1Y - Math.sin((x / n) * Math.PI * 1.4) * hill1A;
        const ridge2 = hill2Y - Math.sin((x / n) * Math.PI * 2.1 + 1.0) * hill2A;
        if (y > ridge2) { r = 0.06; g = 0.22; b = 0.16; }
        else if (y > ridge1) { r = 0.10; g = 0.30; b = 0.22; }
        buf[i] = Math.min(1, r); buf[i + 1] = Math.min(1, g); buf[i + 2] = Math.min(1, b);
      }
    }
    return buf;
  }

  /** Per-channel standard normal noise via Box–Muller, seeded for reproducibility. */
  private makeNoise(n: number, seed: number): Float32Array {
    const out = new Float32Array(n * n * 3);
    let st = (seed * 2654435761) >>> 0;
    const rng = () => { st ^= st << 13; st ^= st >>> 17; st ^= st << 5; st >>>= 0; return st / 4294967296; };
    for (let i = 0; i < out.length; i++) {
      const u1 = Math.max(1e-9, rng()), u2 = rng();
      out[i] = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }
    return out;
  }

  // ====================================================================
  // FIGURE 1 — forward q(x_t | x_0)
  // ====================================================================
  private drawForward(): void {
    const cv = this.fwdCv().nativeElement;
    const ctx = cv.getContext('2d');
    if (!ctx) return;
    const n = this.GRID;
    if (cv.width !== n) { cv.width = n; cv.height = n; }

    const ab = this.alphaBarAtT();
    const sa = Math.sqrt(ab);
    const so = Math.sqrt(Math.max(0, 1 - ab));

    const img = ctx.createImageData(n, n);
    const d = img.data;
    for (let p = 0; p < n * n; p++) {
      const j = p * 3, k = p * 4;
      // x_t = sqrt(aBar) * x0 + sqrt(1-aBar) * eps,  with x0,eps centered around 0.5
      for (let c = 0; c < 3; c++) {
        const x0 = this.base[j + c] - 0.5;        // center so mean is preserved
        const eps = this.noise[j + c] * 0.5;       // scale noise into image units
        const v = 0.5 + sa * x0 + so * eps;
        d[k + c] = Math.max(0, Math.min(255, v * 255));
      }
      d[k + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }

  // ====================================================================
  // FIGURE 2 — schedule visualizer
  // ====================================================================
  private drawChart(): void {
    const cv = this.chartCv().nativeElement;
    const ctx = cv.getContext('2d');
    if (!ctx) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const rect = cv.getBoundingClientRect();
    const W = Math.max(1, Math.floor(rect.width * dpr));
    const H = Math.max(1, Math.floor(rect.height * dpr));
    if (cv.width !== W || cv.height !== H) { cv.width = W; cv.height = H; }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const w = rect.width, h = rect.height;
    const padL = 44, padR = 12, padT = 12, padB = 26;
    const plotW = w - padL - padR, plotH = h - padT - padB;

    ctx.clearRect(0, 0, w, h);

    const which = this.curve();

    // build series arrays
    const N = 200;
    const cos = this.cosTable, lin = this.linTable;
    const sampleAb = (tbl: Float64Array, f: number) => tbl[Math.round(f * this.T)];
    const sampleBeta = (tbl: Float64Array, f: number) => {
      const t = Math.max(1, Math.round(f * this.T));
      return 1 - tbl[t] / Math.max(1e-12, tbl[t - 1]);
    };

    let lo = Infinity, hi = -Infinity;
    const seriesVal = (tbl: Float64Array, f: number): number => {
      if (which === 'beta') return sampleBeta(tbl, f);
      if (which === 'snr') {
        const ab = sampleAb(tbl, f);
        return Math.log(Math.max(1e-9, ab / Math.max(1e-12, 1 - ab)));
      }
      return sampleAb(tbl, f);
    };
    for (let s = 0; s <= N; s++) {
      const f = s / N;
      lo = Math.min(lo, seriesVal(cos, f), seriesVal(lin, f));
      hi = Math.max(hi, seriesVal(cos, f), seriesVal(lin, f));
    }
    if (which === 'alphaBar') { lo = 0; hi = 1; }
    if (hi - lo < 1e-9) hi = lo + 1;
    const pad = (hi - lo) * 0.06;
    lo -= pad; hi += pad;

    const px = (f: number) => padL + f * plotW;
    const py = (v: number) => padT + plotH * (1 - (v - lo) / (hi - lo));

    // grid + axes
    ctx.strokeStyle = 'rgba(255,255,255,0.06)';
    ctx.lineWidth = 1;
    ctx.font = '10px ui-monospace, monospace';
    ctx.fillStyle = 'rgba(255,255,255,0.4)';
    for (let g = 0; g <= 4; g++) {
      const yy = padT + (plotH * g) / 4;
      ctx.beginPath(); ctx.moveTo(padL, yy); ctx.lineTo(w - padR, yy); ctx.stroke();
      const val = hi - ((hi - lo) * g) / 4;
      ctx.fillText(this.axisFmt(val), 4, yy + 3);
    }
    for (let g = 0; g <= 4; g++) {
      const xx = padL + (plotW * g) / 4;
      ctx.fillText(String(Math.round((this.T * g) / 4)), xx - 8, h - 8);
    }
    ctx.fillStyle = 'rgba(255,255,255,0.35)';
    ctx.fillText('t →', w - 28, h - 8);

    // current-t marker
    const fT = this.tStep() / this.T;
    ctx.strokeStyle = 'rgba(124,92,255,0.55)';
    ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(px(fT), padT); ctx.lineTo(px(fT), padT + plotH); ctx.stroke();
    ctx.setLineDash([]);

    const plot = (tbl: Float64Array, color: string) => {
      ctx.strokeStyle = color; ctx.lineWidth = 2.2; ctx.beginPath();
      for (let s = 0; s <= N; s++) {
        const f = s / N;
        const X = px(f), Y = py(seriesVal(tbl, f));
        if (s === 0) ctx.moveTo(X, Y); else ctx.lineTo(X, Y);
      }
      ctx.stroke();
      // dot at current t
      const Xc = px(fT), Yc = py(seriesVal(tbl, fT));
      ctx.fillStyle = color; ctx.beginPath(); ctx.arc(Xc, Yc, 3.4, 0, Math.PI * 2); ctx.fill();
    };
    plot(lin, '#ff5c8a');
    plot(cos, '#41d6ff');
  }

  // ====================================================================
  // FIGURE 3 — reverse (illustrative rewind)
  // ====================================================================
  private drawReverse(): void {
    const cv = this.revCv().nativeElement;
    const ctx = cv.getContext('2d');
    if (!ctx) return;
    const n = this.GRID;
    if (cv.width !== n) { cv.width = n; cv.height = n; }

    // use the cosine table so the rewind matches the default forward look
    const ab = this.cosTable[Math.max(0, Math.min(this.T, this.revStep()))];
    const sa = Math.sqrt(ab);
    const so = Math.sqrt(Math.max(0, 1 - ab));

    const img = ctx.createImageData(n, n);
    const d = img.data;
    for (let p = 0; p < n * n; p++) {
      const j = p * 3, k = p * 4;
      for (let c = 0; c < 3; c++) {
        const x0 = this.base[j + c] - 0.5;
        const eps = this.noise[j + c] * 0.5;
        const v = 0.5 + sa * x0 + so * eps;
        d[k + c] = Math.max(0, Math.min(255, v * 255));
      }
      d[k + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }

  private startReverseLoop(): void {
    let raf = 0;
    let last = performance.now();
    const stepEvery = 28; // ms per visible denoise step
    let acc = 0;
    const loop = (now: number) => {
      const dt = now - last; last = now;
      if (this.revPlaying()) {
        acc += dt;
        while (acc >= stepEvery) {
          acc -= stepEvery;
          const cur = this.revStep();
          if (cur <= 0) { this.revPlaying.set(false); }
          else { this.revStep.set(Math.max(0, cur - Math.ceil(this.T / 80))); }
        }
      } else {
        acc = 0;
      }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    this.destroyRef.onDestroy(() => cancelAnimationFrame(raf));
  }

  // ====================================================================
  // control handlers (zoneless: .set / .update only)
  // ====================================================================
  onT(ev: Event): void {
    const v = +(ev.target as HTMLInputElement).value;
    this.tStep.set(v);
    this.fb.event('interact', { section: 'diffusion', control: 'forward-t', value: v });
  }
  setSchedule(k: 'cosine' | 'linear'): void {
    this.schedule.set(k);
    this.fb.event('interact', { section: 'diffusion', control: 'schedule', value: k });
  }
  setCurve(c: 'alphaBar' | 'beta' | 'snr'): void {
    this.curve.set(c);
    this.fb.event('interact', { section: 'diffusion', control: 'curve', value: c });
  }
  reseed(): void {
    this.seed.update((v) => v + 1);
    this.noise = this.makeNoise(this.GRID, this.seed());
    // bump seed signal already triggers the effects; refresh noise field first
    this.drawForward(); this.drawReverse();
    this.fb.event('interact', { section: 'diffusion', control: 'reseed' });
  }
  toggleReverse(): void {
    if (!this.revPlaying() && this.revStep() <= 0) this.revStep.set(this.T);
    this.revPlaying.update((p) => !p);
    this.fb.event('interact', { section: 'diffusion', control: 'reverse-play', value: this.revPlaying() });
  }

  // ====================================================================
  // formatting helpers
  // ====================================================================
  fmt(v: number): string {
    if (v >= 0.9995) return '1.000';
    if (v < 0.001 && v > 0) return v.toExponential(1);
    return v.toFixed(3);
  }
  fmtSnr(v: number): string {
    if (v > 999) return v.toExponential(1);
    if (v < 0.001) return v.toExponential(1);
    return v.toFixed(2);
  }
  private axisFmt(v: number): string {
    if (Math.abs(v) >= 100 || (Math.abs(v) < 0.01 && v !== 0)) return v.toExponential(0);
    return v.toFixed(2);
  }

  // ---- code snippets quoted from the repo (kept verbatim) ----
  readonly snipQSample = `def q_sample(self, x0: Tensor, t: Tensor, noise: Tensor | None = None) -> Tensor:
    """Sample x_t ~ q(x_t | x_0): the forward diffusion in ONE step.

    Implements the closed form
        x_t = sqrt(alphas_cumprod[t]) * x0 + sqrt(1 - alphas_cumprod[t]) * eps.
    # ...
    """
    if noise is None:
        noise = torch.randn_like(x0)
    sqrt_acp = extract(self.sqrt_alphas_cumprod, t, x0.shape)
    sqrt_1m_acp = extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
    return sqrt_acp * x0 + sqrt_1m_acp * noise`;

  readonly snipCosine = `if kind == "cosine":
    # Nichol & Dhariwal define a smooth f(t) and set
    #   alphas_cumprod(t) = f(t) / f(0),   f(t) = cos^2( ((t/T + s) / (1+s)) * pi/2 ).
    # ...
    s = 0.008
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    f = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = f / f[0]  # normalize so alphas_cumprod[0] == 1
    # beta_t = 1 - alphas_cumprod[t] / alphas_cumprod[t-1]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    # Clamp: the upper bound 0.999 avoids a degenerate (zero-signal) final step.
    return betas.clamp(min=1e-8, max=0.999).float()`;

  readonly snipPSample = `eps = self._model_eps(model, x_t, t, context, mask, guidance_scale, uncond_context)
x0_hat = sched.predict_x0_from_eps(x_t, t, eps)
# Clamp the predicted x0 to a sane range; latents/images live ~[-1, 1] ...
x0_hat = x0_hat.clamp(-3.0, 3.0)

mean, _var, log_var = sched.posterior(x0_hat, x_t, t)
noise = torch.randn_like(x_t)
# No noise at the last step (t == 0): mask out the noise term per batch row.
nonzero = (t != 0).float().reshape(-1, *((1,) * (x_t.dim() - 1)))
return mean + nonzero * (0.5 * log_var).exp() * noise`;
}
