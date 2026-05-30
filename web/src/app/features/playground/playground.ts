import {
  Component, ChangeDetectionStrategy, ElementRef, afterNextRender, inject, DestroyRef,
  viewChild, signal, computed, effect,
} from '@angular/core';
import { Chapter } from '../../shared/chapter';
import { CodeRef } from '../../shared/code-ref';
import { Math as MathTex } from '../../shared/math';
import { FirebaseService } from '../../core/firebase.service';

/**
 * Chapter — Live diffusion playground.
 *
 * The centerpiece is a REAL reverse-diffusion sampler running in the browser on
 * a 2-D point cloud. The target is a mixture of Gaussians placed to form a
 * selectable shape ("two moons", "spiral", "ring", "GEN"). Because the target is
 * a Gaussian mixture, the forward-noised distribution q(x_t) is ALSO a Gaussian
 * mixture (inflated covariance), so its score ∇log q(x_t) is available in closed
 * form — no trained network needed. We convert that analytic score into an eps
 * prediction and run the exact ancestral DDPM update from `pytorch/diffusion/ddpm.py`
 * with the cosine schedule from `pytorch/diffusion/schedule.py`. This is the same
 * algorithm as `pytorch/toy/toy_diffusion_2d.py`; there a tiny MLP LEARNS the
 * score, here we use the exact score so it runs live and is genuinely correct.
 */
@Component({
  selector: 'app-playground',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [Chapter, CodeRef, MathTex],
  template: `
<app-chapter slug="playground">

  <p class="lede">
    Every other chapter explains the reverse process and then shows you a
    <em>picture</em> of it. Here you can run it. The cloud below is doing honest
    DDPM sampling — start from pure static, predict the noise, subtract a little,
    add a pinch back, repeat — and resolving into a shape. The only trick is that
    we picked a target whose score we can write down exactly, so there is no
    network to train. Same loop as Stable Diffusion. Two dimensions instead of a
    million.
  </p>

  <!-- ===================== THE PLAYGROUND ===================== -->
  <figure class="fig">
    <div class="fig__stage">
      <canvas #cv class="fig__canvas"></canvas>
    </div>

    <div class="progress">
      <div class="progress__bar" [style.width.%]="progressPct()"></div>
    </div>

    <div class="ctl">
      <div class="ctl__row ctl__row--btns">
        <span class="ctl__group">target</span>
        @for (tg of targets; track tg.id) {
          <button type="button" class="btn" [class.btn--primary]="target() === tg.id"
            (click)="setTarget(tg.id)">{{ tg.label }}</button>
        }
      </div>

      <div class="ctl__row ctl__row--btns">
        <button type="button" class="btn btn--primary" (click)="togglePlay()">
          {{ playing() ? '❚❚ pause' : (done() ? '▶ replay' : '▶ play') }}
        </button>
        <button type="button" class="btn" (click)="stepOnce()" [disabled]="done() && !playing()">step</button>
        <button type="button" class="btn" (click)="reset()">⟲ reset</button>
        <span class="readout">
          <span class="chip"><span class="chip__k">t</span><span class="chip__v">{{ tNow() }}</span></span>
          <span class="chip"><span class="chip__k">ᾱ<sub>t</sub></span><span class="chip__v">{{ fmt(alphaBarNow()) }}</span></span>
          <span class="chip chip--accent"><span class="chip__k">step</span><span class="chip__v">{{ stepsDone() }} / {{ steps() }}</span></span>
        </span>
      </div>

      <div class="ctl__grid">
        <div class="ctl__field">
          <label class="ctl__label" for="pts">points <span class="mono">{{ nPoints() }}</span></label>
          <input id="pts" class="ctl__range" type="range" min="200" max="4000" step="100"
            [value]="nPoints()" (input)="onPoints($event)" />
        </div>
        <div class="ctl__field">
          <label class="ctl__label" for="stp">reverse steps <span class="mono">{{ steps() }}</span></label>
          <input id="stp" class="ctl__range" type="range" min="20" max="240" step="10"
            [value]="steps()" (input)="onSteps($event)" />
        </div>
        <div class="ctl__field">
          <label class="ctl__label" for="tmp">
            guidance / temperature <span class="mono">{{ fmt(guidance()) }}</span>
          </label>
          <input id="tmp" class="ctl__range" type="range" min="0.5" max="3" step="0.05"
            [value]="guidance()" (input)="onGuidance($event)" />
        </div>
      </div>
    </div>

    <figcaption class="fig__cap">
      <strong>Live reverse diffusion.</strong> {{ nPoints() }} points start as
      <app-math expr="\\mathcal{N}(0,\\mathbf{I})" /> static and are denoised over
      {{ steps() }} ancestral DDPM steps into a mixture of Gaussians shaped like
      <em>{{ targetLabel() }}</em>. The score is computed
      <strong>analytically</strong> at every step — this is genuine sampling, not a
      replayed animation. Guidance &gt; 1 sharpens the cloud onto the shape (lower
      temperature); guidance &lt; 1 leaves it looser.
    </figcaption>
  </figure>

  <h2>What you are looking at</h2>

  <p>
    A point cloud <em>is</em> a probability distribution over the plane. The dots
    you see are samples; where they crowd, density is high. The shape you picked
    defines a target distribution <app-math expr="p_{\\text{data}}(x)" />, and the
    job of reverse diffusion is to turn a featureless Gaussian blob back into that
    distribution — to walk samples from noise to data. Watch the static condense
    into structure: that condensation is the whole of what a diffusion model does,
    minus the U-Net and the million pixels.
  </p>

  <p>
    We build the target as a <strong>mixture of Gaussians</strong>: scatter
    <app-math expr="K" /> anchor points <app-math expr="a_k" /> along the shape, and
    put a tiny isotropic blob of variance <app-math expr="\\sigma^2" /> on each. So
  </p>

  <app-math display
    expr="p_{\\text{data}}(x) = \\frac{1}{K}\\sum_{k=1}^{K}\\mathcal{N}\\!\\big(x;\\,a_k,\\,\\sigma^2\\mathbf{I}\\big)." />

  <p>
    The anchors are sampled exactly the way the PyTorch reference builds its 2-D
    targets — two interleaving half-circles for the moons, an Archimedean arm for
    the spiral, a ring of blobs — plus a stroked <span class="mono">GEN</span>
    point cloud for fun. In the repo those points are the <em>training data</em> for
    a tiny network; here we never train, because for a Gaussian mixture we can
    write the score down.
  </p>

  <app-code-ref
    file="pytorch/toy/toy_diffusion_2d.py"
    lang="python"
    [code]="snipTarget"
    [lines]="[107, 135]"
    caption="make_target — the same shapes this playground samples. In the repo these points are data to learn; here they become mixture anchors." />

  <h2>Why the score is exact (no network needed)</h2>

  <p>
    The forward process noises a clean point: at timestep <app-math expr="t" />,
    <app-math expr="x_t = \\sqrt{\\bar\\alpha_t}\\,x_0 + \\sqrt{1-\\bar\\alpha_t}\\,\\varepsilon" />.
    Apply that to a single Gaussian and you get another Gaussian — its mean scales
    by <app-math expr="\\sqrt{\\bar\\alpha_t}" /> and its variance inflates by
    <app-math expr="1-\\bar\\alpha_t" />. Apply it to a <em>mixture</em> and you get a
    mixture: each component noises independently. So the marginal the sampler
    actually faces, <app-math expr="q(x_t)" />, is itself a Gaussian mixture with
    component variance <app-math expr="v_t = \\bar\\alpha_t\\,\\sigma^2 + (1-\\bar\\alpha_t)" />:
  </p>

  <app-math display
    expr="q(x_t) = \\frac{1}{K}\\sum_{k=1}^{K}\\mathcal{N}\\!\\big(x_t;\\,\\sqrt{\\bar\\alpha_t}\\,a_k,\\,v_t\\,\\mathbf{I}\\big)." />

  <p>
    The score of a Gaussian mixture has a closed form: a
    <strong>softmax-weighted average</strong> of the per-component scores, where
    each component pulls <app-math expr="x_t" /> toward its (scaled) anchor:
  </p>

  <app-math display
    expr="\\nabla_{x}\\log q(x_t) = \\sum_{k=1}^{K} w_k(x_t)\\,\\frac{\\sqrt{\\bar\\alpha_t}\\,a_k - x_t}{v_t},\\qquad w_k \\propto \\exp\\!\\Big(\\!-\\tfrac{\\lVert x_t-\\sqrt{\\bar\\alpha_t}\\,a_k\\rVert^2}{2 v_t}\\Big)." />

  <p>
    That is the line every diffusion model spends millions of parameters trying to
    approximate — and here we just compute it. A learned denoiser predicts noise
    <app-math expr="\\varepsilon_\\theta" />; score and noise are two views of the
    same thing, related by
    <app-math expr="\\varepsilon_\\theta(x_t,t) = -\\sqrt{1-\\bar\\alpha_t}\\,\\nabla_x\\log q(x_t)" />.
    So we convert our exact score into an exact <app-math expr="\\varepsilon" /> and
    hand it to the standard reverse update.
  </p>

  <h2>The reverse step is the repo's <code>p_sample</code>, verbatim</h2>

  <p>
    With an <app-math expr="\\varepsilon" /> in hand, each reverse step is the
    ancestral DDPM update: back out a clean estimate
    <app-math expr="\\hat x_0" />, form the posterior mean for the previous, less
    noisy timestep, and add a little fresh noise (except at the very last step):
  </p>

  <app-math display
    expr="\\hat x_0 = \\tfrac{1}{\\sqrt{\\bar\\alpha_t}}\\big(x_t - \\sqrt{1-\\bar\\alpha_t}\\,\\varepsilon\\big),\\qquad x_{t-1} = \\tilde\\mu(\\hat x_0, x_t) + \\mathbb{1}[t>0]\\,\\sigma_t\\,z,\\quad z\\sim\\mathcal{N}(0,\\mathbf{I})." />

  <p>
    The JavaScript loop driving the canvas implements exactly these four moves on
    each of the <app-math expr="N" /> 2-D points. It mirrors line-for-line the
    PyTorch reverse step — the only difference is where <app-math expr="\\varepsilon" />
    comes from (our analytic score versus a network's prediction):
  </p>

  <app-code-ref
    file="pytorch/diffusion/ddpm.py"
    lang="python"
    [code]="snipPSample"
    [lines]="[172, 183]"
    caption="GaussianDiffusion.p_sample — one ancestral reverse step. The canvas runs this identically; only the source of eps differs." />

  <p>
    In the reference file, that <app-math expr="\\varepsilon" /> is produced by a
    laptop-sized MLP standing in for a U-Net. Its job is to <em>learn</em> the
    very score we computed in closed form above — which is exactly why the toy file
    is the honest miniature of the whole field. The denoiser is just a function
    from <span class="mono">(noisy point, timestep)</span> to noise:
  </p>

  <app-code-ref
    file="pytorch/toy/toy_diffusion_2d.py"
    lang="python"
    [code]="snipMlp"
    [lines]="[92, 101]"
    caption="MLPDenoiser.forward — the learned stand-in for a U-Net: featurize t, concat with x, predict eps. The playground replaces this with the exact score." />

  <h2>The knobs, and what they actually do</h2>

  <p>
    <strong>Reverse steps</strong> is the number of timesteps <app-math expr="T" />
    in the schedule. More steps means each one removes less noise, so the cloud
    settles more smoothly — the classic quality-versus-speed trade of every
    sampler. <strong>Points</strong> is just how many samples we draw in parallel;
    diffusion denoises them all at once, independently, which is why it stays
    smooth as you crank it up.
  </p>

  <p>
    The <strong>guidance / temperature</strong> knob scales the score before the
    update, <app-math expr="\\nabla\\log q \\rightarrow \\gamma\\,\\nabla\\log q" />.
    It is the toy analogue of <strong>classifier-free guidance</strong>: pushing
    harder along the score sharpens samples onto the high-density core of each mode
    (effectively sampling a lower-temperature, "more confident" distribution),
    while <app-math expr="\\gamma<1" /> relaxes the pull and leaves a fuzzier,
    higher-temperature cloud. Turn it up and watch the moons get crisp; turn it
    down and they blur. In a real text-to-image model this same dial is what makes
    a prompt "stick."
  </p>

  <h2>This is the same engine as everything else</h2>

  <p>
    Nothing here is a special 2-D algorithm. The schedule is the cosine
    <app-math expr="\\bar\\alpha_t" /> from <code>schedule.py</code>; the update is
    <code>p_sample</code> from <code>ddpm.py</code>; the shapes come from
    <code>toy_diffusion_2d.py</code>. Swap the 2-D point for a <app-math expr="64\\times64\\times4" />
    latent grid, swap our closed-form score for a U-Net or a DiT, and inject a text
    prompt through cross-attention at each step — and you have Stable Diffusion. The
    iteration is over <em>noise level</em>, never over space or time: every point
    (every pixel, every frame) is refined together, all at once. That global
    refinement is the source of diffusion's coherence, and it is precisely what the
    "next-frame prediction" mental model — which belongs to the autoregressive
    family — gets wrong.
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
    .mono { font-family: var(--font-mono); color: var(--ink-0); }

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
      padding: 0; background:
        radial-gradient(120% 120% at 50% 0%, rgba(124,92,255,0.08), transparent 60%), #06070d;
    }
    .fig__canvas {
      display: block; width: 100%; aspect-ratio: 16 / 10; image-rendering: auto;
    }

    .progress { height: 3px; width: 100%; background: var(--bg-3); }
    .progress__bar {
      height: 100%; width: 0%;
      background: linear-gradient(90deg, var(--plasma-a), var(--plasma-b));
      transition: width .08s linear;
    }

    .ctl { padding: 1rem 1.1rem; border-top: 1px solid var(--line); display: grid; gap: 0.85rem; }
    .ctl__row--btns { display: flex; flex-wrap: wrap; align-items: center; gap: 0.5rem; }
    .ctl__group {
      font-family: var(--font-mono); font-size: 0.72rem; letter-spacing: 0.1em;
      text-transform: uppercase; color: var(--ink-3); margin-right: 0.2rem;
    }
    .btn:disabled { opacity: 0.4; cursor: not-allowed; }

    .readout { display: inline-flex; flex-wrap: wrap; gap: 0.45rem; margin-left: auto; }
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

    .ctl__grid {
      display: grid; gap: 0.9rem 1.4rem;
      grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
      padding-top: 0.2rem; border-top: 1px solid var(--line);
    }
    .ctl__field { display: grid; gap: 0.5rem; padding-top: 0.7rem; }
    .ctl__label { font-family: var(--font-mono); font-size: 0.78rem; color: var(--ink-2); }

    .ctl__range { -webkit-appearance: none; appearance: none; width: 100%; height: 6px;
      border-radius: 999px; background: linear-gradient(90deg, var(--plasma-a), var(--plasma-b));
      outline: none; cursor: pointer; }
    .ctl__range::-webkit-slider-thumb { -webkit-appearance: none; appearance: none;
      width: 18px; height: 18px; border-radius: 50%; background: #fff;
      border: 2px solid var(--plasma-a); box-shadow: 0 2px 8px rgba(0,0,0,0.5); cursor: pointer; }
    .ctl__range::-moz-range-thumb { width: 18px; height: 18px; border-radius: 50%; background: #fff;
      border: 2px solid var(--plasma-a); cursor: pointer; }

    .fig__cap {
      padding: 0.85rem 1.1rem; border-top: 1px solid var(--line);
      font-size: 0.84rem; color: var(--ink-2); background: rgba(255,255,255,0.012);
    }
    .fig__cap strong { color: var(--ink-1); }
    .fig__cap em { color: #cdbcff; font-style: normal; }
  `],
})
export class Playground {
  private readonly fb = inject(FirebaseService);
  private readonly destroyRef = inject(DestroyRef);
  private readonly cvRef = viewChild.required<ElementRef<HTMLCanvasElement>>('cv');
  private viewReady = false;          // canvas exists only after afterNextRender

  // ---- target catalog ----
  readonly targets = [
    { id: 'moons', label: 'two moons' },
    { id: 'spiral', label: 'spiral' },
    { id: 'ring', label: 'ring' },
    { id: 'gen', label: 'GEN' },
  ] as const;

  // ---- reactive controls ----
  readonly target = signal<'moons' | 'spiral' | 'ring' | 'gen'>('moons');
  readonly nPoints = signal(1500);
  readonly steps = signal(120);
  readonly guidance = signal(1.4);
  readonly playing = signal(false);

  // ---- live progress state (written by the sampler, read by the template) ----
  readonly stepsDone = signal(0);     // how many reverse steps applied
  readonly tNow = signal(0);          // current timestep index t (T..0)

  // ---- derived readouts ----
  readonly done = computed(() => this.stepsDone() >= this.steps());
  readonly progressPct = computed(() => Math.min(100, (this.stepsDone() / this.steps()) * 100));
  readonly targetLabel = computed(() => this.targets.find((t) => t.id === this.target())!.label);
  readonly alphaBarNow = computed(() => {
    const t = this.tNow();
    return t <= 0 ? 1 : this.alphaBarAt(t, this.steps());
  });

  // ---- mutable simulation buffers (NOT signals; the RAF loop owns them) ----
  private anchors: Float64Array = new Float64Array(0); // [K*2] mixture anchors a_k
  private K = 0;
  private sigma = 0.06;                                 // base mixture std (in data units)
  private px = new Float64Array(0);                     // current x positions [N]
  private py = new Float64Array(0);                     // current y positions [N]
  private rngState = 0x9e3779b9 >>> 0;

  constructor() {
    afterNextRender(() => {
      this.viewReady = true;
      this.rebuildTarget();
      this.reset();          // seed cloud + draw frame 0
      this.startLoop();
    });

    // rebuild anchors + restart whenever the target changes
    effect(() => { this.target(); if (this.anchors.length) { this.rebuildTarget(); this.reset(); } });
    // resizing the point count restarts the sampler so buffers match
    effect(() => { this.nPoints(); if (this.px.length) this.reset(); });
  }

  // ====================================================================
  // schedule (cosine ᾱ_t, mirrors pytorch/diffusion/schedule.py)
  // ====================================================================
  /** ᾱ_t for a schedule of length T at integer step t in [0, T]. */
  private alphaBarAt(t: number, T: number): number {
    const s = 0.008;
    const f0 = Math.cos((s / (1 + s)) * Math.PI * 0.5) ** 2;
    const f = Math.cos(((t / T + s) / (1 + s)) * Math.PI * 0.5) ** 2;
    return Math.max(1e-6, Math.min(1, f / f0));
  }

  // ====================================================================
  // RNG + Gaussian (seeded xorshift + Box–Muller)
  // ====================================================================
  private rand(): number {
    let x = this.rngState;
    x ^= x << 13; x ^= x >>> 17; x ^= x << 5; x >>>= 0;
    this.rngState = x;
    return x / 4294967296;
  }
  private gauss(): number {
    const u1 = Math.max(1e-9, this.rand());
    const u2 = this.rand();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }

  // ====================================================================
  // target construction — anchors a_k for each shape
  // (mirrors make_target in pytorch/toy/toy_diffusion_2d.py)
  // ====================================================================
  private rebuildTarget(): void {
    const name = this.target();
    const K = name === 'gen' ? 360 : 220;
    const a = new Float64Array(K * 2);

    if (name === 'moons') {
      const half = K >> 1;
      for (let k = 0; k < K; k++) {
        let x: number, y: number;
        if (k < half) {                       // outer upper half-circle
          const th = (k / half) * Math.PI;
          x = Math.cos(th); y = Math.sin(th);
        } else {                              // inner lower half-circle, shifted
          const th = ((k - half) / (K - half)) * Math.PI;
          x = 1 - Math.cos(th); y = 0.5 - Math.sin(th);
        }
        a[k * 2] = (x - 0.5) * 1.4;
        a[k * 2 + 1] = (y - 0.25) * 1.4;
      }
      this.sigma = 0.05;
    } else if (name === 'spiral') {
      for (let k = 0; k < K; k++) {
        const tt = Math.sqrt(k / K) * 3 * Math.PI;   // denser near center
        const r = (tt / (3 * Math.PI)) * 1.8;
        a[k * 2] = r * Math.cos(tt);
        a[k * 2 + 1] = r * Math.sin(tt);
      }
      this.sigma = 0.04;
    } else if (name === 'ring') {
      const modes = 8;
      for (let k = 0; k < K; k++) {
        const m = k % modes;
        const ang = (2 * Math.PI * m) / modes;
        a[k * 2] = Math.cos(ang) * 1.5;
        a[k * 2 + 1] = Math.sin(ang) * 1.5;
      }
      this.sigma = 0.1;
    } else {
      this.sampleGEN(a, K);
      this.sigma = 0.045;
    }

    this.anchors = a;
    this.K = K;
  }

  /** Stroke the letters G E N as a point cloud, centered to ~[-1.8,1.8]. */
  private sampleGEN(a: Float64Array, K: number): void {
    // each letter is a list of polyline strokes in a local 0..1 box
    const G: number[][][] = [[
      [0.95, 0.1], [0.5, 0.0], [0.1, 0.18], [0.0, 0.5], [0.1, 0.82], [0.5, 1.0],
      [0.9, 0.9], [1.0, 0.62], [0.55, 0.62], [0.55, 0.5], [1.0, 0.5],
    ]];
    const E: number[][][] = [
      [[1.0, 1.0], [0.0, 1.0], [0.0, 0.5], [0.0, 0.0], [1.0, 0.0]],
      [[0.0, 0.5], [0.78, 0.5]],
    ];
    const N: number[][][] = [[[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]];
    const letters = [G, E, N];
    const slotW = 1.25, slotH = 1.9, gap = 0.35;
    const totalW = letters.length * slotW + (letters.length - 1) * gap;
    let cursor = -totalW / 2;
    const placed: number[][] = [];           // collected [x,y] points
    for (const strokes of letters) {
      for (const stroke of strokes) {
        for (let s = 0; s < stroke.length - 1; s++) {
          const [x0, y0] = stroke[s], [x1, y1] = stroke[s + 1];
          const seg = 16;
          for (let q = 0; q <= seg; q++) {
            const f = q / seg;
            const lx = x0 + (x1 - x0) * f;
            const ly = y0 + (y1 - y0) * f;
            placed.push([cursor + lx * slotW, (ly - 0.5) * slotH]);
          }
        }
      }
      cursor += slotW + gap;
    }
    // resample to exactly K anchors (with replacement is fine)
    for (let k = 0; k < K; k++) {
      const p = placed[Math.floor(this.rand() * placed.length)];
      a[k * 2] = p[0];
      a[k * 2 + 1] = p[1];
    }
  }

  // ====================================================================
  // sampler — start from N(0,I) static
  // ====================================================================
  /**
   * Apply ONE ancestral DDPM reverse step at the current timestep, in place.
   * eps comes from the EXACT mixture score; everything else is the repo's
   * p_sample arithmetic (predict x0, posterior mean, add noise unless t==0).
   */
  private reverseStep(): void {
    const T = this.steps();
    const stepIdx = this.stepsDone();
    if (stepIdx >= T) return;

    const t = T - stepIdx;                    // current (1-indexed) timestep
    const tPrev = t - 1;
    const abT = this.alphaBarAt(t, T);
    const abPrev = this.alphaBarAt(tPrev, T);
    const alphaT = abT / abPrev;              // alpha_t = ᾱ_t / ᾱ_{t-1}
    const betaT = 1 - alphaT;

    const sqrtAbT = Math.sqrt(abT);
    const oneMinusAbT = Math.max(1e-8, 1 - abT);
    const sqrtOneMinusAbT = Math.sqrt(oneMinusAbT);
    const vT = abT * this.sigma * this.sigma + oneMinusAbT;   // noised mixture variance

    // posterior mean coefficients (Ho et al. eq. 7)
    const coef1 = (betaT * Math.sqrt(abPrev)) / oneMinusAbT;          // on x0_hat
    const coef2 = ((1 - abPrev) * Math.sqrt(alphaT)) / oneMinusAbT;   // on x_t
    // posterior variance β̃_t; no noise at the last step (t==1 here => tPrev==0)
    const postVar = tPrev > 0 ? (betaT * (1 - abPrev)) / oneMinusAbT : 0;
    const postStd = Math.sqrt(Math.max(0, postVar));

    const gamma = this.guidance();
    const N = this.px.length;
    const a = this.anchors, K = this.K;
    const invTwoV = 1 / (2 * vT);

    for (let i = 0; i < N; i++) {
      const xt = this.px[i], yt = this.py[i];

      // ---- analytic mixture score ∇log q(x_t) = Σ softmax_k (sqrt(ᾱ) a_k - x_t)/v_t ----
      // softmax over -||x_t - sqrt(ᾱ) a_k||^2 / (2 v_t), log-sum-exp stable.
      let maxLog = -Infinity;
      // first pass: find max log-weight
      for (let k = 0; k < K; k++) {
        const ax = sqrtAbT * a[k * 2] - xt;
        const ay = sqrtAbT * a[k * 2 + 1] - yt;
        const lw = -(ax * ax + ay * ay) * invTwoV;
        if (lw > maxLog) maxLog = lw;
      }
      let wsum = 0, sx = 0, sy = 0;
      for (let k = 0; k < K; k++) {
        const ax = sqrtAbT * a[k * 2] - xt;
        const ay = sqrtAbT * a[k * 2 + 1] - yt;
        const w = Math.exp(-(ax * ax + ay * ay) * invTwoV - maxLog);
        wsum += w;
        sx += w * ax;   // (sqrt(ᾱ) a_k - x_t), weight w; divide by v_t after
        sy += w * ay;
      }
      const scoreX = (sx / wsum) / vT;
      const scoreY = (sy / wsum) / vT;

      // guidance/temperature: scale the score (toy classifier-free guidance)
      const gScoreX = gamma * scoreX;
      const gScoreY = gamma * scoreY;

      // eps = -sqrt(1-ᾱ_t) * score
      const epsX = -sqrtOneMinusAbT * gScoreX;
      const epsY = -sqrtOneMinusAbT * gScoreY;

      // x0_hat = (x_t - sqrt(1-ᾱ_t) eps) / sqrt(ᾱ_t), clamped like the repo
      let x0x = (xt - sqrtOneMinusAbT * epsX) / sqrtAbT;
      let x0y = (yt - sqrtOneMinusAbT * epsY) / sqrtAbT;
      x0x = x0x < -3 ? -3 : x0x > 3 ? 3 : x0x;
      x0y = x0y < -3 ? -3 : x0y > 3 ? 3 : x0y;

      // posterior mean + (maybe) fresh noise
      let nx = coef1 * x0x + coef2 * xt;
      let ny = coef1 * x0y + coef2 * yt;
      if (postStd > 0) { nx += postStd * this.gauss(); ny += postStd * this.gauss(); }

      this.px[i] = nx;
      this.py[i] = ny;
    }

    this.stepsDone.set(stepIdx + 1);
    this.tNow.set(tPrev);
  }

  // ====================================================================
  // draw — plasma-colored cloud, anchors faint underneath
  // ====================================================================
  private draw(): void {
    const cv = this.cvRef().nativeElement;
    const ctx = cv.getContext('2d');
    if (!ctx) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const rect = cv.getBoundingClientRect();
    const W = Math.max(1, Math.floor(rect.width * dpr));
    const H = Math.max(1, Math.floor(rect.height * dpr));
    if (cv.width !== W || cv.height !== H) { cv.width = W; cv.height = H; }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const w = rect.width, h = rect.height;

    ctx.clearRect(0, 0, w, h);

    // world [-2.6, 2.6] -> screen, square aspect centered
    const span = 5.2;
    const scale = Math.min(w, h) / span;
    const cx = w / 2, cy = h / 2;
    const sx = (x: number) => cx + x * scale;
    const sy = (y: number) => cy - y * scale;

    // faint target anchors so you can see what the cloud is converging to
    ctx.fillStyle = 'rgba(124,92,255,0.16)';
    const a = this.anchors, K = this.K;
    for (let k = 0; k < K; k++) {
      ctx.beginPath();
      ctx.arc(sx(a[k * 2]), sy(a[k * 2 + 1]), 1.4, 0, Math.PI * 2);
      ctx.fill();
    }

    // the live cloud: progress drives color from cyan (noise) -> magenta (data)
    const prog = this.steps() > 0 ? this.stepsDone() / this.steps() : 0;
    const N = this.px.length;
    ctx.globalCompositeOperation = 'lighter';
    const r = 1.7;
    for (let i = 0; i < N; i++) {
      // tiny per-point hue jitter for life
      const jitter = (i % 7) / 7 - 0.5;
      const tcol = Math.max(0, Math.min(1, prog + jitter * 0.12));
      // lerp #41d6ff -> #ff5c8a through #7c5cff
      const col = this.plasma(tcol);
      ctx.fillStyle = col;
      ctx.beginPath();
      ctx.arc(sx(this.px[i]), sy(this.py[i]), r, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.globalCompositeOperation = 'source-over';
  }

  /** plasma palette ramp: 0 -> cyan, 0.5 -> violet, 1 -> magenta. */
  private plasma(t: number): string {
    const c0 = [0x41, 0xd6, 0xff];   // --plasma-b cyan
    const c1 = [0x7c, 0x5c, 0xff];   // --plasma-a violet
    const c2 = [0xff, 0x5c, 0x8a];   // --plasma-c magenta
    let r: number, g: number, b: number;
    if (t < 0.5) {
      const f = t / 0.5;
      r = c0[0] + (c1[0] - c0[0]) * f; g = c0[1] + (c1[1] - c0[1]) * f; b = c0[2] + (c1[2] - c0[2]) * f;
    } else {
      const f = (t - 0.5) / 0.5;
      r = c1[0] + (c2[0] - c1[0]) * f; g = c1[1] + (c2[1] - c1[1]) * f; b = c1[2] + (c2[2] - c1[2]) * f;
    }
    return `rgba(${r | 0},${g | 0},${b | 0},0.62)`;
  }

  // ====================================================================
  // RAF loop — steps the sampler while playing, always redraws
  // ====================================================================
  private startLoop(): void {
    let raf = 0;
    let last = performance.now();
    let acc = 0;
    const resize = () => this.draw();
    window.addEventListener('resize', resize);

    const loop = (now: number) => {
      const dt = now - last; last = now;
      if (this.playing() && !this.done()) {
        // pace steps so the run takes a couple seconds regardless of step count
        acc += dt;
        const budget = Math.max(1, (2200 / this.steps()));   // ms per reverse step
        let guard = 0;
        while (acc >= budget && !this.done() && guard < 4) {
          acc -= budget;
          this.reverseStep();
          guard++;
        }
        if (this.done()) this.playing.set(false);
      } else {
        acc = 0;
      }
      this.draw();
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    this.destroyRef.onDestroy(() => {
      cancelAnimationFrame(raf);
      window.removeEventListener('resize', resize);
    });
  }

  // ====================================================================
  // control handlers (zoneless: .set / .update only)
  // ====================================================================
  setTarget(id: 'moons' | 'spiral' | 'ring' | 'gen'): void {
    this.target.set(id);   // effect rebuilds + resets
    this.fb.event('interact', { section: 'playground', control: 'target', value: id });
  }
  togglePlay(): void {
    if (this.done()) this.reset();
    this.playing.update((p) => !p);
    this.fb.event('interact', { section: 'playground', control: 'play', value: this.playing() });
  }
  stepOnce(): void {
    this.playing.set(false);
    if (this.done()) return;
    this.reverseStep();
    this.draw();
    this.fb.event('interact', { section: 'playground', control: 'step', value: this.stepsDone() });
  }
  reset(): void {
    this.playing.set(false);
    this.seedAndDraw();
    this.fb.event('interact', { section: 'playground', control: 'reset' });
  }
  onPoints(ev: Event): void {
    this.nPoints.set(+(ev.target as HTMLInputElement).value);   // effect resets buffers
    this.fb.event('interact', { section: 'playground', control: 'points', value: this.nPoints() });
  }
  onSteps(ev: Event): void {
    this.steps.set(+(ev.target as HTMLInputElement).value);
    this.seedAndDraw();
    this.fb.event('interact', { section: 'playground', control: 'steps', value: this.steps() });
  }
  onGuidance(ev: Event): void {
    this.guidance.set(+(ev.target as HTMLInputElement).value);
    this.fb.event('interact', { section: 'playground', control: 'guidance', value: this.guidance() });
  }

  /** Re-seed the cloud from static and draw frame 0 (shared by reset/onSteps). */
  private seedAndDraw(): void {
    const N = this.nPoints();
    this.px = new Float64Array(N);
    this.py = new Float64Array(N);
    this.rngState = 0x9e3779b9 >>> 0;
    for (let i = 0; i < N; i++) { this.px[i] = this.gauss(); this.py[i] = this.gauss(); }
    this.stepsDone.set(0);
    this.tNow.set(this.steps());
    if (this.viewReady) this.draw();
  }

  // ====================================================================
  // formatting
  // ====================================================================
  fmt(v: number): string {
    if (v >= 0.9995) return '1.00';
    if (v < 0.01 && v > 0) return v.toExponential(1);
    return v.toFixed(2);
  }

  // ====================================================================
  // code snippets quoted verbatim from the repo
  // ====================================================================
  readonly snipTarget = `def make_target(name: str, n: int) -> Tensor:
    # ... returns [N, 2] points from a named 2-D distribution.
    rng = np.random.default_rng()

    if name == "moons":
        # Two interleaving half-circles (the classic sklearn "two moons").
        theta_out = rng.uniform(0, math.pi, n_out)
        outer = np.stack([np.cos(theta_out), np.sin(theta_out)], axis=1)
        theta_in = rng.uniform(0, math.pi, n_in)
        inner = np.stack([1.0 - np.cos(theta_in), 0.5 - np.sin(theta_in)], axis=1)
        pts = np.concatenate([outer, inner], axis=0)

    elif name == "spiral":
        t = np.sqrt(rng.uniform(0, 1, n)) * 3.0 * math.pi  # denser near center
        r = t / (3.0 * math.pi)
        pts = np.stack([r * np.cos(t), r * np.sin(t)], axis=1)`;

  readonly snipMlp = `def forward(self, x: Tensor, t: Tensor, context=None, mask=None) -> Tensor:
    """x: [B, 2], t: [B] long -> predicted eps: [B, 2]."""
    t_emb = self.time_mlp(self.time_sinusoid(t))  # [B, time_dim]
    h = torch.cat([x, t_emb], dim=-1)             # [B, 2 + time_dim]
    return self.net(h)`;

  readonly snipPSample = `sched = self.schedule
eps = self._model_eps(model, x_t, t, context, mask, guidance_scale, uncond_context)
x0_hat = sched.predict_x0_from_eps(x_t, t, eps)
# Clamp the predicted x0 to a sane range ...
x0_hat = x0_hat.clamp(-3.0, 3.0)

mean, _var, log_var = sched.posterior(x0_hat, x_t, t)
noise = torch.randn_like(x_t)
# No noise at the last step (t == 0): mask out the noise term per batch row.
nonzero = (t != 0).float().reshape(-1, *((1,) * (x_t.dim() - 1)))
return mean + nonzero * (0.5 * log_var).exp() * noise`;
}
