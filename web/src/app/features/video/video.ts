import {
  Component, ChangeDetectionStrategy, ElementRef, afterNextRender, inject, DestroyRef,
  viewChild, signal, computed, effect,
} from '@angular/core';
import { Chapter } from '../../shared/chapter';
import { CodeRef } from '../../shared/code-ref';
import { Math as MathTex } from '../../shared/math';
import { FirebaseService } from '../../core/firebase.service';

/**
 * Chapter 06 — Spacetime latents: generating the whole clip at once.
 *
 * Three interactive figures, each tied to the real PyTorch under pytorch/video/:
 *   1. spacetime volume: a frames×H×W latent brick chopped into (pt,ph,pw) patches,
 *      each becoming one token; token count = (T/pt)(H/ph)(W/pw)  — patchify_video.
 *   2. coherence contrast: an "all-at-once diffusion" blob (stable) vs an
 *      "autoregressive frame-by-frame" blob that drifts as error compounds.
 *   3. factorized attention: a diagram of within-frame spatial vs across-frame
 *      temporal attention — FactorizedSpacetimeBlock / TemporalAttention.
 */
@Component({
  selector: 'app-video',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [Chapter, CodeRef, MathTex],
  template: `
<app-chapter slug="video">

  <p class="lede">
    An image diffusion model denoises one frame. A video model denoises a
    <em>stack</em> of them — and the whole trick is to refuse to think of them as
    separate frames at all. Stable Diffusion runs in a small VAE latent; the video
    version makes that latent a <strong>spacetime volume</strong> of shape
    <app-math expr="T\\times H\\times W" />, chops it into little bricks of pixels
    that span both space <em>and</em> time, and runs a diffusion Transformer over
    all of them <strong>at once</strong>. Every frame is denoised jointly, so motion
    holds together instead of flickering.
  </p>

  <h2>A clip is one big sequence of spacetime patches</h2>

  <p>
    A Vision Transformer turns an image <app-math expr="[C,H,W]" /> into a sequence
    of <app-math expr="P\\times P" /> patch tokens. A video DiT does the exact same
    thing, except the patch is a little <em>brick</em> spanning
    <app-math expr="p_t" /> frames by <app-math expr="p_h" /> rows by
    <app-math expr="p_w" /> columns. Flatten every brick and you get one token
    sequence that mixes appearance and motion:
  </p>

  <app-math display
    expr="\\text{video}\\;[B,C,T,H,W] \\;\\longrightarrow\\; \\text{tokens}\\;[B,N,\\,C\\cdot p_t p_h p_w],\\qquad N=\\frac{T}{p_t}\\cdot\\frac{H}{p_h}\\cdot\\frac{W}{p_w}." />

  <p>
    Drag the patch-size controls below. The volume re-chops live, and the token
    count <app-math expr="N" /> updates with it. Bigger patches mean fewer, fatter
    tokens — cheaper attention, coarser detail. This single number is the knob that
    decides how much a video Transformer has to chew on.
  </p>

  <!-- FIGURE 1 — spacetime volume -->
  <figure class="fig">
    <div class="fig__stage">
      <canvas #vol class="fig__canvas fig__canvas--wide"></canvas>
    </div>
    <div class="ctl">
      <div class="ctl__grid">
        <label class="ctl__cell">
          <span class="ctl__label">p<sub>t</sub> · frames <span class="mono">{{ pt() }}</span></span>
          <input class="ctl__range" type="range" min="1" max="4" step="1" [value]="pt()" (input)="onP('t', $event)" />
        </label>
        <label class="ctl__cell">
          <span class="ctl__label">p<sub>h</sub> · rows <span class="mono">{{ ph() }}</span></span>
          <input class="ctl__range" type="range" min="1" max="4" step="1" [value]="ph()" (input)="onP('h', $event)" />
        </label>
        <label class="ctl__cell">
          <span class="ctl__label">p<sub>w</sub> · cols <span class="mono">{{ pw() }}</span></span>
          <input class="ctl__range" type="range" min="1" max="4" step="1" [value]="pw()" (input)="onP('w', $event)" />
        </label>
      </div>
      <div class="ctl__readout">
        <span class="chip"><span class="chip__k">T×H×W</span><span class="chip__v">{{ T }}×{{ H }}×{{ W }}</span></span>
        <span class="chip"><span class="chip__k">grid</span><span class="chip__v">{{ gt() }}×{{ gh() }}×{{ gw() }}</span></span>
        <span class="chip chip--accent"><span class="chip__k">N tokens</span><span class="chip__v">{{ tokenCount() }}</span></span>
        <span class="chip"><span class="chip__k">D = C·p<sub>t</sub>p<sub>h</sub>p<sub>w</sub></span><span class="chip__v">{{ tokenDim() }}</span></span>
      </div>
      <div class="ctl__row ctl__row--btns">
        <button type="button" class="btn" [class.btn--primary]="explode()" (click)="toggleExplode()">
          {{ explode() ? '◳ pack patches' : '◱ explode patches' }}
        </button>
        <span class="ctl__group">a diffusion Transformer attends over all {{ tokenCount() }} tokens together</span>
      </div>
    </div>
    <figcaption class="fig__cap">
      A latent clip drawn as a stack of <strong>{{ T }}</strong> frames forming a
      <app-math expr="T\\times H\\times W" /> volume, chopped into spacetime patches of size
      <app-math expr="(p_t,p_h,p_w)" />. Each brick is one token; token count is
      <app-math expr="N=(T/p_t)(H/p_h)(W/p_w)" />.
    </figcaption>
  </figure>

  <p>
    The split itself is not learned — it is a pure <code>reshape</code> +
    <code>permute</code>, which is why it has an exact inverse. The token ordering
    is row-major over <app-math expr="(g_t, g_h, g_w)" />: time outermost, then
    height, then width. That is <code>patchify_video</code> verbatim:
  </p>

  <app-code-ref
    file="pytorch/video/spacetime.py"
    lang="python"
    [code]="snipPatchify"
    [lines]="[51, 66]"
    caption="patchify_video — split each axis into (grid, patch), reorder, and flatten the bricks into one [B, N, C·pt·ph·pw] sequence." />

  <p>
    In a real model the reshape is folded into a strided 3D convolution — a
    <code>Conv3d</code> whose kernel and stride both equal the patch size is
    <em>exactly</em> a per-patch linear projection, the standard ViT/DiT
    "patch embed" extended to time. Same bricks, now learnable and mapped straight
    to the model width.
  </p>

  <h2>Why all-at-once beats frame-by-frame</h2>

  <p>
    Here is the heart of it. Because every patch is just a token in one sequence,
    attention can relate a patch in frame 0 to a patch in frame 7. The model
    denoises the <strong>entire clip jointly</strong> — and that is the opposite of
    the "next-frame prediction" mental model, which belongs to the autoregressive
    family, not diffusion. Diffusion iterates over <em>noise level</em>, not over
    time.
  </p>

  <p>
    The contrast below makes the failure mode concrete. On the left, an all-at-once
    diffusion clip: the blob's motion is fixed once, globally, so it stays smooth.
    On the right, a frame-by-frame generator that conditions each new frame only on
    the last — every tiny prediction error feeds the next step, so the trajectory
    visibly <strong>drifts</strong> and the shape wobbles. Seeing all frames
    together is what keeps a diffusion video temporally coherent.
  </p>

  <!-- FIGURE 2 — coherence contrast -->
  <figure class="fig">
    <div class="fig__stage fig__stage--pair">
      <div class="pair">
        <span class="pair__tag pair__tag--good">all-at-once diffusion</span>
        <canvas #diff class="fig__canvas fig__canvas--sq"></canvas>
      </div>
      <div class="pair">
        <span class="pair__tag pair__tag--warn">autoregressive, frame-by-frame</span>
        <canvas #auto class="fig__canvas fig__canvas--sq"></canvas>
      </div>
    </div>
    <div class="ctl">
      <div class="ctl__row ctl__row--btns">
        <button type="button" class="btn btn--primary" (click)="togglePlay()">
          {{ playing() ? '❚❚ pause' : '▶ play' }}
        </button>
        <button type="button" class="btn" (click)="resetClip()">↻ restart clip</button>
        <span class="ctl__group">frame <span class="mono">{{ frame() }}</span> / {{ FRAMES - 1 }} · drift Δ <span class="mono">{{ driftLabel() }}</span></span>
      </div>
      <p class="ctl__note">
        Both blobs follow the <em>same</em> intended path (the dashed circle). The
        right one only ever sees its previous frame, so error accumulates — a
        compounding-error simulation, exactly the weakness joint denoising avoids.
      </p>
    </div>
    <figcaption class="fig__cap">
      Left: one clip denoised together stays on the path. Right: predicting each
      frame from the last lets small errors compound into visible drift.
    </figcaption>
  </figure>

  <p>
    This is the Sora-style recipe in one line: spacetime patches go into a diffusion
    Transformer, the whole clip is denoised in one shared computation, and temporal
    attention stitches the frames. The backbone's <code>forward</code> embeds the
    clip, runs a stack of factorized blocks, and folds per-token noise predictions
    back into a <app-math expr="[B,C,T,H,W]" /> clip:
  </p>

  <app-code-ref
    file="pytorch/video/video_dit.py"
    lang="python"
    [code]="snipForward"
    [lines]="[312, 332]"
    caption="VideoDiT.forward (tail) — reshape tokens to [B, T, S, D], run factorized spatial+temporal blocks, then unpatch into eps for the whole clip." />

  <h2>Factorized attention: within a frame, then across frames</h2>

  <p>
    Full 3D attention over every token is <app-math expr="\\mathcal{O}(N^2)" /> with
    <app-math expr="N = T\\cdot H\\cdot W" />, which blows up fast. The standard fix
    (ViViT, video DiT, AnimateDiff-style temporal layers) is to <strong>factorize</strong>
    attention into two cheaper passes per block:
  </p>

  <ul class="facts">
    <li><strong>Spatial</strong> attention — within each frame, tokens attend to other tokens of the
      <em>same</em> frame; time is held fixed. Reshape to <app-math expr="[B\\cdot T,\\,S,\\,D]" />.</li>
    <li><strong>Temporal</strong> attention — for each spatial location, tokens attend
      <em>across</em> frames; space is held fixed. Reshape to <app-math expr="[B\\cdot S,\\,T,\\,D]" />.</li>
  </ul>

  <p>
    Stacking the two gives every token an indirect path to every other token, at a
    fraction of dense 3D cost. The diagram is interactive: hover or tap to highlight
    one token and watch which neighbours each pass mixes it with. The
    <strong>temporal</strong> pass is the one that enforces motion consistency.
  </p>

  <!-- FIGURE 3 — factorized attention diagram -->
  <figure class="fig">
    <div class="fig__stage">
      <canvas #attn class="fig__canvas fig__canvas--wide" (pointermove)="onHover($event)" (pointerleave)="clearHover()"></canvas>
    </div>
    <div class="ctl">
      <div class="ctl__row ctl__row--btns">
        <span class="ctl__group">attention pass:</span>
        <button type="button" class="btn" [class.btn--primary]="pass() === 'spatial'" (click)="setPass('spatial')">spatial (within frame)</button>
        <button type="button" class="btn" [class.btn--primary]="pass() === 'temporal'" (click)="setPass('temporal')">temporal (across frames)</button>
      </div>
      <div class="ctl__readout ctl__readout--legend">
        <span class="chip"><span class="dot dot--q"></span>query token</span>
        <span class="chip"><span class="dot dot--a"></span>attends to</span>
        <span class="chip chip--ghost">grid = frames (→) × spatial slots (↓)</span>
      </div>
    </div>
    <figcaption class="fig__cap">
      Factorized spacetime attention. <strong>Spatial</strong> mixes within a column
      (one frame); <strong>temporal</strong> mixes along a row (one spatial slot across
      frames). Highlighted token shows its attention set for the selected pass.
    </figcaption>
  </figure>

  <p>
    One block is literally spatial self-attention, then temporal self-attention,
    then an MLP — each wrapped in adaLN-Zero conditioning from the timestep (plus
    pooled text). <code>TemporalAttention</code> just moves the spatial axis into the
    batch so the same attention kernel runs along time:
  </p>

  <app-code-ref
    file="pytorch/video/temporal_attention.py"
    lang="python"
    [code]="snipTemporal"
    [lines]="[93, 100]"
    caption="TemporalAttention.forward — treat each spatial location as its own length-T sequence by folding S into the batch dimension." />

  <app-code-ref
    file="pytorch/video/temporal_attention.py"
    lang="python"
    [code]="snipBlock"
    [lines]="[171, 187]"
    caption="FactorizedSpacetimeBlock.forward — spatial pass (frames into batch), temporal pass, then MLP; each gated by zero-init adaLN so the block starts as identity (the DiT recipe)." />

  <h2>Conditioning: image-to-video and beyond</h2>

  <p>
    Once the whole clip is a token sequence, conditioning is just extra context.
    Text steers exactly as in image diffusion — cross-attention from the video
    tokens to the prompt embeddings at every block, amplified by classifier-free
    guidance. <strong>Image-to-video</strong> fixes the first frame: you overwrite
    the leading time-steps with the clean conditioning frame and add a binary
    "this token is known" channel, so the model must denoise the rest to be
    consistent with what it was handed. <strong>Video-to-video</strong> and
    ControlNet-style structure (condition on depth or edge maps) ride the same
    rails — extra channels or extra context tokens, never a change to the
    all-at-once core.
  </p>

  <p>
    In <code>VideoDiT</code>, I2V is "replace + mask": splice the clean frames in,
    flag them, and let the joint denoiser fill the future. The patch embed even
    reads a mask channel telling it which time-steps are observed — so a single
    forward pass handles unconditional generation, I2V, and inpainting alike.
  </p>

  <p>
    The throughline of this whole site: diffusion never predicts the next frame. It
    holds the entire spacetime latent in view and refines all of it together, step
    after step, until coherent motion falls out. The autoregressive family — tokens
    predicted one at a time, text and frames interleaved in one sequence — is the
    <a class="link" href="/autoregressive">other way a machine dreams a clip ↗</a>,
    and hybrids (autoregressive across chunks, diffusion within) are where a lot of
    current research lives.
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
    .facts { margin: 1rem 0; padding-left: 1.1rem; display: grid; gap: 0.55rem; color: var(--ink-1); }
    .facts strong { color: var(--ink-0); }

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
    .fig__stage--pair { gap: 1.4rem; flex-wrap: wrap; }
    .fig__canvas { display: block; width: 100%; image-rendering: auto; }
    .fig__canvas--sq { max-width: 280px; aspect-ratio: 1 / 1; border-radius: var(--radius-sm); border: 1px solid var(--line-strong); }
    .fig__canvas--wide { max-width: 720px; aspect-ratio: 16 / 8; border-radius: var(--radius-sm); }

    .pair { display: grid; gap: 0.55rem; justify-items: center; }
    .pair__tag {
      font-family: var(--font-mono); font-size: 0.72rem; letter-spacing: 0.04em;
      padding: 0.22em 0.7em; border-radius: 999px; border: 1px solid var(--line);
    }
    .pair__tag--good { color: var(--good); border-color: rgba(70,224,160,0.4); background: rgba(70,224,160,0.08); }
    .pair__tag--warn { color: var(--warn); border-color: rgba(255,207,92,0.4); background: rgba(255,207,92,0.08); }

    .ctl { padding: 1rem 1.1rem; border-top: 1px solid var(--line); display: grid; gap: 0.75rem; }
    .ctl__row { display: grid; gap: 0.5rem; }
    .ctl__row--btns { display: flex; flex-wrap: wrap; align-items: center; gap: 0.5rem; }
    .ctl__grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.8rem; }
    .ctl__cell { display: grid; gap: 0.4rem; }
    .ctl__label { font-family: var(--font-mono); font-size: 0.78rem; color: var(--ink-2); }
    .ctl__group { font-family: var(--font-mono); font-size: 0.76rem; color: var(--ink-3); }
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
    .dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
    .dot--q { background: var(--plasma-c); }
    .dot--a { background: var(--plasma-b); }

    .ctl__note { margin: 0; font-size: 0.8rem; color: var(--ink-2); }
    .fig__cap {
      padding: 0.75rem 1.1rem; border-top: 1px solid var(--line);
      font-size: 0.84rem; color: var(--ink-2); background: rgba(255,255,255,0.012);
    }
    .fig__cap strong { color: var(--ink-1); }

    @media (max-width: 520px) { .ctl__grid { grid-template-columns: 1fr; } }
  `],
})
export class Video {
  private readonly fb = inject(FirebaseService);
  private readonly destroyRef = inject(DestroyRef);

  // ---- FIGURE 1: spacetime volume dims (latent grid, in patch-able units) ----
  readonly T = 8;   // frames
  readonly H = 8;   // latent rows
  readonly W = 8;   // latent cols
  private readonly C = 4; // latent channels (Stable-Diffusion-style)

  readonly pt = signal(1);
  readonly ph = signal(2);
  readonly pw = signal(2);
  readonly explode = signal(false);

  // grid counts per axis (floor so the volume always tiles)
  readonly gt = computed(() => Math.floor(this.T / this.pt()));
  readonly gh = computed(() => Math.floor(this.H / this.ph()));
  readonly gw = computed(() => Math.floor(this.W / this.pw()));
  readonly tokenCount = computed(() => this.gt() * this.gh() * this.gw());
  readonly tokenDim = computed(() => this.C * this.pt() * this.ph() * this.pw());

  // ---- FIGURE 2: coherence contrast ----
  readonly FRAMES = 96;            // length of the looping clip
  readonly playing = signal(true);
  readonly frame = signal(0);
  private driftSig = signal(0);
  readonly driftLabel = computed(() => this.driftSig().toFixed(2));
  // precomputed drift offsets for the autoregressive blob (compounding error)
  private autoPath: { x: number; y: number }[] = this.buildAutoPath(7);

  // ---- FIGURE 3: factorized attention ----
  readonly pass = signal<'spatial' | 'temporal'>('temporal');
  private readonly hoverTok = signal<number | null>(null); // index in the T×S grid
  private readonly aFrames = 6;    // columns in the diagram
  private readonly aSlots = 5;     // rows (spatial slots) in the diagram

  // ---- canvases ----
  private readonly volCv = viewChild.required<ElementRef<HTMLCanvasElement>>('vol');
  private readonly diffCv = viewChild.required<ElementRef<HTMLCanvasElement>>('diff');
  private readonly autoCv = viewChild.required<ElementRef<HTMLCanvasElement>>('auto');
  private readonly attnCv = viewChild.required<ElementRef<HTMLCanvasElement>>('attn');

  constructor() {
    afterNextRender(() => {
      this.drawVolume();
      this.drawAttn();
      this.startClipLoop();
    });
    // static figures react to their controls
    effect(() => { this.pt(); this.ph(); this.pw(); this.explode(); this.drawVolume(); });
    effect(() => { this.pass(); this.hoverTok(); this.drawAttn(); });
  }

  // ====================================================================
  // FIGURE 1 — spacetime volume drawn as an isometric stack of frames
  // ====================================================================
  private drawVolume(): void {
    const cv = this.volCv().nativeElement;
    const ctx = cv.getContext('2d');
    if (!ctx) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const rect = cv.getBoundingClientRect();
    const Wd = Math.max(1, Math.floor(rect.width * dpr));
    const Hd = Math.max(1, Math.floor(rect.height * dpr));
    if (cv.width !== Wd || cv.height !== Hd) { cv.width = Wd; cv.height = Hd; }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const w = rect.width, h = rect.height;
    ctx.clearRect(0, 0, w, h);

    // isometric basis: each frame is a HxW grid in the XY plane, frames stacked along time (depth)
    const cell = Math.min(w, h) / 16;            // base cell size in px
    const depth = this.explode() ? cell * 2.6 : cell * 1.15; // gap between frames
    const skew = cell * 0.55;                    // isometric shear per depth unit
    const pt = this.pt(), ph = this.ph(), pw = this.pw();

    // overall footprint to center the drawing
    const drawW = this.W * cell + (this.T - 1) * skew;
    const drawH = this.H * cell + (this.T - 1) * depth * 0.0;
    const totalDepthShift = (this.T - 1) * depth;
    const ox = (w - drawW) / 2;
    const oy = (h - this.H * cell - totalDepthShift * 0.55) / 2 + totalDepthShift * 0.55;

    // project (col, row, frame) -> screen; frames recede up-right
    const proj = (col: number, row: number, fr: number) => ({
      x: ox + col * cell + fr * skew,
      y: oy + row * cell - fr * depth * 0.55,
    });

    // draw far frames first (back to front along time)
    for (let f = this.T - 1; f >= 0; f--) {
      const pIdx = Math.floor(f / pt); // which temporal patch this frame belongs to
      // frame backing panel
      const a = proj(0, 0, f), b = proj(this.W, 0, f), c = proj(this.W, this.H, f), d = proj(0, this.H, f);
      ctx.beginPath();
      ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.lineTo(c.x, c.y); ctx.lineTo(d.x, d.y); ctx.closePath();
      const tint = 0.06 + 0.04 * (1 - f / this.T);
      ctx.fillStyle = `rgba(124,92,255,${tint})`;
      ctx.fill();
      ctx.strokeStyle = 'rgba(255,255,255,0.10)';
      ctx.lineWidth = 1;
      ctx.stroke();

      // patch tiles on this frame: color by patch index so equal patches share a hue
      for (let gr = 0; gr < this.gh(); gr++) {
        for (let gc = 0; gc < this.gw(); gc++) {
          const x0 = gc * pw, y0 = gr * ph;
          const p0 = proj(x0, y0, f), p1 = proj(x0 + pw, y0, f);
          const p2 = proj(x0 + pw, y0 + ph, f), p3 = proj(x0, y0 + ph, f);
          // a deterministic hue per (temporal patch, gr, gc) so a brick reads as one token
          const hueSeed = (pIdx * 131 + gr * 17 + gc * 7) % 360;
          const onTime = f % pt === 0; // first frame of its temporal patch -> brighter
          const lift = onTime ? 0.55 : 0.32;
          ctx.beginPath();
          ctx.moveTo(p0.x, p0.y); ctx.lineTo(p1.x, p1.y); ctx.lineTo(p2.x, p2.y); ctx.lineTo(p3.x, p3.y); ctx.closePath();
          ctx.fillStyle = `hsla(${180 + hueSeed * 0.5}, 70%, 60%, ${lift * 0.5})`;
          ctx.fill();
          ctx.strokeStyle = `hsla(${180 + hueSeed * 0.5}, 80%, 70%, ${onTime ? 0.85 : 0.4})`;
          ctx.lineWidth = onTime ? 1.4 : 0.7;
          ctx.stroke();
        }
      }
    }

    // time arrow label
    ctx.fillStyle = 'rgba(255,255,255,0.5)';
    ctx.font = '11px ui-monospace, monospace';
    const tip = proj(this.W * 0.2, -0.6, this.T - 1);
    ctx.fillText('time →', tip.x, tip.y);
    ctx.fillText('one brick = one token', ox, oy + this.H * cell + 18);
  }

  // ====================================================================
  // FIGURE 2 — coherence contrast (diffusion vs autoregressive drift)
  // ====================================================================
  /** Precompute the autoregressive trajectory: ideal circle + compounding noise. */
  private buildAutoPath(seed: number): { x: number; y: number }[] {
    let st = (seed * 2654435761) >>> 0;
    const rng = () => { st ^= st << 13; st ^= st >>> 17; st ^= st << 5; st >>>= 0; return st / 4294967296; };
    const path: { x: number; y: number }[] = [];
    let ex = 0, ey = 0; // accumulated error in normalized units
    for (let i = 0; i < this.FRAMES; i++) {
      // each step adds a small biased error that is never corrected (drift)
      ex += (rng() - 0.45) * 0.010;
      ey += (rng() - 0.5) * 0.012;
      path.push({ x: ex, y: ey });
    }
    return path;
  }

  private idealPos(i: number): { x: number; y: number } {
    const a = (i / this.FRAMES) * Math.PI * 2;
    return { x: Math.cos(a) * 0.26, y: Math.sin(a) * 0.26 };
  }

  private drawBlob(cv: HTMLCanvasElement, cx: number, cy: number, color: string, wobble: number): void {
    const ctx = cv.getContext('2d');
    if (!ctx) return;
    const n = 140;
    if (cv.width !== n) { cv.width = n; cv.height = n; }
    ctx.clearRect(0, 0, n, n);

    // dashed reference path (the intended circle)
    ctx.strokeStyle = 'rgba(255,255,255,0.16)';
    ctx.setLineDash([3, 4]);
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(n / 2, n / 2, 0.26 * n, 0, Math.PI * 2);
    ctx.stroke();
    ctx.setLineDash([]);

    // blob: a soft, slightly wobbly disc
    const px = n / 2 + cx * n, py = n / 2 + cy * n;
    const grad = ctx.createRadialGradient(px, py, 2, px, py, 22);
    grad.addColorStop(0, color);
    grad.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.fillStyle = grad;
    ctx.beginPath();
    for (let k = 0; k <= 32; k++) {
      const a = (k / 32) * Math.PI * 2;
      const r = 14 + Math.sin(a * 3 + this.frame() * 0.3) * wobble;
      const x = px + Math.cos(a) * r, y = py + Math.sin(a) * r;
      if (k === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.fill();
  }

  private renderClip(): void {
    const i = this.frame();
    const ideal = this.idealPos(i);
    // left: diffusion blob sits exactly on the intended path (joint denoising)
    this.drawBlob(this.diffCv().nativeElement, ideal.x, ideal.y, 'rgba(70,224,160,0.9)', 1.3);
    // right: autoregressive blob = intended path + accumulated drift
    const err = this.autoPath[i];
    this.drawBlob(this.autoCv().nativeElement, ideal.x + err.x, ideal.y + err.y, 'rgba(255,207,92,0.9)', 2.6);
    this.driftSig.set(Math.hypot(err.x, err.y) * 100);
  }

  private startClipLoop(): void {
    this.renderClip();
    let raf = 0;
    let last = performance.now();
    const msPerFrame = 70;
    let acc = 0;
    const loop = (now: number) => {
      const dt = now - last; last = now;
      if (this.playing()) {
        acc += dt;
        while (acc >= msPerFrame) {
          acc -= msPerFrame;
          this.frame.set((this.frame() + 1) % this.FRAMES);
        }
        this.renderClip();
      } else {
        acc = 0;
      }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    this.destroyRef.onDestroy(() => cancelAnimationFrame(raf));
  }

  // ====================================================================
  // FIGURE 3 — factorized attention diagram
  // ====================================================================
  private attnLayout(w: number, h: number) {
    const cols = this.aFrames, rows = this.aSlots;
    const padX = 60, padY = 36;
    const gx = (w - padX * 2) / (cols - 1);
    const gy = (h - padY * 2) / (rows - 1);
    return { cols, rows, padX, padY, gx, gy };
  }

  private drawAttn(): void {
    const cv = this.attnCv().nativeElement;
    const ctx = cv.getContext('2d');
    if (!ctx) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const rect = cv.getBoundingClientRect();
    const Wd = Math.max(1, Math.floor(rect.width * dpr));
    const Hd = Math.max(1, Math.floor(rect.height * dpr));
    if (cv.width !== Wd || cv.height !== Hd) { cv.width = Wd; cv.height = Hd; }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const w = rect.width, h = rect.height;
    ctx.clearRect(0, 0, w, h);

    const { cols, rows, padX, padY, gx, gy } = this.attnLayout(w, h);
    const pos = (c: number, r: number) => ({ x: padX + c * gx, y: padY + r * gy });

    // axis labels
    ctx.fillStyle = 'rgba(255,255,255,0.45)';
    ctx.font = '11px ui-monospace, monospace';
    ctx.fillText('frames (time) →', padX, h - 8);
    ctx.save();
    ctx.translate(16, padY + 4);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('← spatial slots', -((rows - 1) * gy), 0);
    ctx.restore();

    // default query = center token if none hovered
    const q = this.hoverTok() ?? (Math.floor(rows / 2) * cols + Math.floor(cols / 2));
    const qc = q % cols, qr = Math.floor(q / cols);
    const isSpatial = this.pass() === 'spatial';

    // draw attention links for the chosen pass first (under the nodes)
    ctx.strokeStyle = 'rgba(65,214,255,0.5)';
    ctx.lineWidth = 1.6;
    const qp = pos(qc, qr);
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        if (c === qc && r === qr) continue;
        const inSet = isSpatial ? (c === qc) : (r === qr);
        if (!inSet) continue;
        const p = pos(c, r);
        ctx.beginPath(); ctx.moveTo(qp.x, qp.y); ctx.lineTo(p.x, p.y); ctx.stroke();
      }
    }

    // nodes
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const p = pos(c, r);
        const isQ = c === qc && r === qr;
        const inSet = isSpatial ? (c === qc) : (r === qr);
        ctx.beginPath();
        ctx.arc(p.x, p.y, isQ ? 9 : 6.5, 0, Math.PI * 2);
        if (isQ) ctx.fillStyle = '#ff5c8a';
        else if (inSet) ctx.fillStyle = 'rgba(65,214,255,0.95)';
        else ctx.fillStyle = 'rgba(255,255,255,0.18)';
        ctx.fill();
      }
    }
  }

  onHover(ev: PointerEvent): void {
    const cv = this.attnCv().nativeElement;
    const rect = cv.getBoundingClientRect();
    const x = ev.clientX - rect.left, y = ev.clientY - rect.top;
    const { cols, rows, padX, padY, gx, gy } = this.attnLayout(rect.width, rect.height);
    const c = Math.round((x - padX) / gx), r = Math.round((y - padY) / gy);
    if (c >= 0 && c < cols && r >= 0 && r < rows) {
      this.hoverTok.set(r * cols + c);
    }
  }
  clearHover(): void { this.hoverTok.set(null); }

  // ====================================================================
  // control handlers (zoneless: .set / .update only)
  // ====================================================================
  onP(axis: 't' | 'h' | 'w', ev: Event): void {
    const v = Math.max(1, +(ev.target as HTMLInputElement).value);
    if (axis === 't') this.pt.set(v);
    else if (axis === 'h') this.ph.set(v);
    else this.pw.set(v);
    this.fb.event('interact', { section: 'video', control: `patch-${axis}`, value: v });
  }
  toggleExplode(): void {
    this.explode.update((e) => !e);
    this.fb.event('interact', { section: 'video', control: 'explode', value: this.explode() });
  }
  togglePlay(): void {
    this.playing.update((p) => !p);
    this.fb.event('interact', { section: 'video', control: 'clip-play', value: this.playing() });
  }
  resetClip(): void {
    this.frame.set(0);
    this.renderClip();
    this.fb.event('interact', { section: 'video', control: 'clip-reset' });
  }
  setPass(p: 'spatial' | 'temporal'): void {
    this.pass.set(p);
    this.fb.event('interact', { section: 'video', control: 'attn-pass', value: p });
  }

  // ---- code snippets quoted verbatim from the repo ----
  readonly snipPatchify = `b, c, t, h, w = x.shape
if t % pt or h % ph or w % pw:
    raise ValueError(
        f"video dims (T={t},H={h},W={w}) must be divisible by patch (pt={pt},ph={ph},pw={pw})"
    )
gt, gh, gw = t // pt, h // ph, w // pw

# Split each axis into (grid, patch): [B, C, gt, pt, gh, ph, gw, pw]
x = x.reshape(b, c, gt, pt, gh, ph, gw, pw)
# Reorder so all the grid axes come first (token index) and all the per-patch
# content axes (C, pt, ph, pw) come last (token feature vector):
#   [B, gt, gh, gw, C, pt, ph, pw]
x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
# Merge grid -> N, and content -> D = C*pt*ph*pw.
tokens = x.reshape(b, gt * gh * gw, c * pt * ph * pw)
return tokens, (gt, gh, gw)`;

  readonly snipForward = `# 3) reshape flat tokens to [B, T(grid), S, D] for the factorized blocks.
gt, gh, gw = grid
s = gh * gw
h_tok = tokens.view(b, gt, s, self.hidden)

for i, block in enumerate(self.blocks):
    h_tok = block(h_tok, cond)  # spatial + temporal self-attn + MLP
    # optional cross-attention to text tokens (zero-gated at init).
    if self.cross_attns is not None and context is not None:
        flat = h_tok.reshape(b, gt * s, self.hidden)
        ca = self.cross_attns[i](self.cross_norms[i](flat), context, mask=mask)
        flat = flat + self.cross_gates[i] * ca
        h_tok = flat.view(b, gt, s, self.hidden)

# 4) adaLN-Zero output head -> per-token eps patches -> fold back to a clip.
tokens = h_tok.reshape(b, self.num_tokens, self.hidden)
shift, scale = self.final_ada(cond).chunk(2, dim=-1)
tokens = _modulate(self.final_norm(tokens), shift, scale)
tokens = self.final_linear(tokens)  # [B, N, out_dim]
eps = self._unpatch_tokens(tokens)  # [B, C, T, H, W]
return eps`;

  readonly snipTemporal = `def forward(self, x: Tensor) -> Tensor:
    b, t, s, d = x.shape
    # [B, T, S, D] -> [B, S, T, D] -> [B*S, T, D]  (each spatial loc = one T-sequence)
    x = x.permute(0, 2, 1, 3).reshape(b * s, t, d)
    x = self.attn(x)
    # back to [B, T, S, D]
    x = x.reshape(b, s, t, d).permute(0, 2, 1, 3).contiguous()
    return x`;

  readonly snipBlock = `# ---- spatial self-attention (within each frame) ----
h = _modulate(self.norm_spatial(x), shift_sa, scale_sa)  # [B,T,S,D]
h = h.reshape(b * t, s, d)            # frames into batch
h = self.spatial_attn(h)
h = h.reshape(b, t, s, d)
x = x + gate_sa.unsqueeze(1).unsqueeze(1) * h

# ---- temporal self-attention (across frames) ----
h = _modulate(self.norm_temporal(x), shift_ta, scale_ta)
h = self.temporal_attn(h)             # [B,T,S,D]
x = x + gate_ta.unsqueeze(1).unsqueeze(1) * h

# ---- per-token MLP ----
h = _modulate(self.norm_mlp(x), shift_mlp, scale_mlp)
h = self.mlp(h)
x = x + gate_mlp.unsqueeze(1).unsqueeze(1) * h
return x`;
}
