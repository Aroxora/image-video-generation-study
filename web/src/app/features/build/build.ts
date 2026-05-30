import {
  Component, ChangeDetectionStrategy, ElementRef, afterNextRender, inject, DestroyRef,
  viewChild, signal,
} from '@angular/core';
import { Chapter } from '../../shared/chapter';
import { CodeRef } from '../../shared/code-ref';
import { Math as MathTex } from '../../shared/math';
import { FirebaseService } from '../../core/firebase.service';

/**
 * Chapter 10 — Build it yourself in PyTorch.
 *
 * The bridge from concepts to the runnable repo. Two interactive figures:
 *   1. an assembly diagram (canvas) of the pipeline — dataset -> [VAE] -> schedule
 *      + GaussianDiffusion -> backbone(+text/CFG) -> AdamW -> ckpt -> sample.py,
 *      animated with data flowing along the arrows; click a box to focus it;
 *   2. a tabbed code reader that shows the REAL excerpt for each stage
 *      (p_losses, p_sample, ddim_sample, CFG context-dropout, VAE encode).
 * Plus a verified quickstart and an honest "teaching-scale" closing note.
 */

interface Stage {
  readonly id: string;
  readonly label: string;
  readonly sub: string;
  readonly file: string;
}

@Component({
  selector: 'app-build',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [Chapter, CodeRef, MathTex],
  template: `
<app-chapter slug="build">

  <p class="lede">
    Every other chapter took one idea apart in isolation. This one snaps them
    back together. The companion repo is a single, deliberately tiny diffusion
    pipeline you can run on a laptop CPU in seconds — same architecture as the
    billion-parameter systems, just shrunk until you can read all of it. Below
    is the whole machine as one diagram, then the exact lines of code that make
    each box do its job.
  </p>

  <h2>The whole machine, one diagram</h2>

  <p>
    Training a generative model is plumbing: route data through a fixed forward
    process to manufacture targets, hand those to a learned denoiser, average
    the error, and step an optimizer. Inference reverses the flow — start from
    static, ask the network to subtract noise a few dozen times, decode. The
    figure animates a batch flowing left-to-right through training, then loops
    the saved checkpoint back into the sampler. Click any box to pin it and pull
    up its source.
  </p>

  <!-- FIGURE 1 — assembly diagram -->
  <figure class="fig">
    <div class="fig__stage fig__stage--diagram">
      <canvas #wire class="fig__canvas fig__canvas--wide"></canvas>
    </div>
    <div class="ctl">
      <div class="ctl__row ctl__row--btns">
        <button type="button" class="btn" [class.btn--primary]="flow() === 'train'" (click)="setFlow('train')">training flow</button>
        <button type="button" class="btn" [class.btn--primary]="flow() === 'sample'" (click)="setFlow('sample')">sampling flow</button>
        <button type="button" class="btn" (click)="toggleRun()">{{ running() ? 'pause' : 'play' }}</button>
        @if (pinned(); as p) {
          <span class="chip chip--accent"><span class="chip__k">pinned</span><span class="chip__v">{{ p }}</span></span>
        } @else {
          <span class="chip chip--ghost">click a box to inspect</span>
        }
      </div>
    </div>
    <figcaption class="fig__cap">
      The pipeline as wired in <code>pytorch/train.py</code> (train) and
      <code>pytorch/sample.py</code> (sample). In <strong>training flow</strong>
      a batch streams forward into the loss; in <strong>sampling flow</strong>
      the checkpoint feeds the reverse chain back to an image. Boxes are clickable.
    </figcaption>
  </figure>

  <p>
    Notice the one-way valve in the middle. The <strong>forward</strong> arrow
    (dataset → noisy <app-math expr="x_t" />) is pure arithmetic from the
    <code>NoiseSchedule</code> — no parameters. Only the
    <strong>backbone</strong> is learned, and it only ever does one thing: read a
    noisy tensor and a timestep and predict the noise that was added. Everything
    else in <code>train.py</code> is glue that keeps the shapes lined up.
  </p>

  <h2>Five stages, the real code</h2>

  <p>
    The tabs below are the load-bearing methods of the repo, quoted verbatim with
    their true line numbers. Read them in order and you have read the entire
    learning loop: build a target, regress on it, then walk the chain backwards.
  </p>

  <!-- FIGURE 2 — tabbed code reader -->
  <figure class="fig">
    <div class="tabs" role="tablist">
      @for (s of stages; track s.id) {
        <button
          type="button"
          role="tab"
          class="tabs__tab"
          [class.tabs__tab--on]="stage() === s.id"
          [attr.aria-selected]="stage() === s.id"
          (click)="setStage(s.id)">
          <span class="tabs__lab">{{ s.label }}</span>
          <span class="tabs__sub">{{ s.sub }}</span>
        </button>
      }
    </div>

    <div class="tabs__panel">
      @if (stage() === 'loss') {
        <p class="tabs__prose">
          <strong>The training step.</strong> One uniformly random timestep
          <app-math expr="t" /> per example, jump the clean batch to <app-math expr="x_t" />
          with the closed-form forward blend
          <app-math expr="x_t=\\sqrt{\\bar\\alpha_t}\\,x_0+\\sqrt{1-\\bar\\alpha_t}\\,\\varepsilon" />,
          ask the model for <app-math expr="\\hat\\varepsilon" />, and minimize
          <app-math expr="\\lVert\\varepsilon-\\hat\\varepsilon\\rVert^2" />. That single MSE
          — Ho et al.'s <em>L<sub>simple</sub></em> — is the whole objective.
        </p>
        <app-code-ref
          file="pytorch/diffusion/ddpm.py"
          lang="python"
          [code]="snipPLosses"
          [lines]="[81, 103]"
          caption="GaussianDiffusion.p_losses — forward-diffuse to x_t, predict the noise, return MSE. The forward process here is only a target factory." />
      }
      @if (stage() === 'dropout') {
        <p class="tabs__prose">
          <strong>Classifier-free guidance, trained for.</strong> To let inference
          amplify the prompt, the model must also know how to denoise <em>without</em>
          one. During training we randomly blank the text context with probability
          <app-math expr="p" /> (here <span class="mono">0.1</span>) so the same network
          learns both the conditional and unconditional score. The dropout happens in
          the training loop right before the loss:
        </p>
        <app-code-ref
          file="pytorch/train.py"
          lang="python"
          [code]="snipDropout"
          [lines]="[321, 337]"
          caption="train.py inner loop — encode (optional VAE) to x0, build the text context, then drop_context() blanks it with probability guidance_dropout for CFG." />
      }
      @if (stage() === 'ddpm') {
        <p class="tabs__prose">
          <strong>One ancestral DDPM step.</strong> Predict the noise (optionally
          with CFG), back out a clean estimate <app-math expr="\\hat{x}_0" />, form the
          true posterior mean, and add a dab of fresh noise — except at
          <app-math expr="t=0" />, where we return the mean so the final sample is clean.
          Run this for every <app-math expr="t=T-1\\ldots0" /> and static becomes an image.
        </p>
        <app-code-ref
          file="pytorch/diffusion/ddpm.py"
          lang="python"
          [code]="snipPSample"
          [lines]="[152, 183]"
          caption="GaussianDiffusion.p_sample — the learned reverse step (Ho et al. Algorithm 2). The full sampler calls this T times." />
      }
      @if (stage() === 'ddim') {
        <p class="tabs__prose">
          <strong>DDIM: the same network, far fewer steps.</strong> Keep the trained
          <app-math expr="\\varepsilon" />-predictor but use a non-Markovian update that
          jumps across a sub-sequence of timesteps. With <app-math expr="\\eta=0" /> the
          noise term vanishes and sampling becomes a deterministic ODE — good images in
          20–50 steps instead of 1000. This is what <code>--steps 50</code> selects.
        </p>
        <app-code-ref
          file="pytorch/diffusion/ddpm.py"
          lang="python"
          [code]="snipDdim"
          [lines]="[266, 288]"
          caption="GaussianDiffusion.ddim_sample — the few-step deterministic update. x_{t_prev} = √ᾱ_prev·x̂₀ + direction·ε (+ σ·z when η>0)." />
      }
      @if (stage() === 'sample') {
        <p class="tabs__prose">
          <strong>Inference, end to end.</strong> <code>sample.py</code> rebuilds the
          exact model from the checkpoint config, encodes a prompt into a context plus
          a zeroed <em>null</em> context (the same empty prompt dropout trained on),
          runs DDPM or DDIM with a guidance scale, and VAE-decodes if the model is
          latent. Guidance is forced to <span class="mono">1.0</span> when there is no
          condition to push toward.
        </p>
        <app-code-ref
          file="pytorch/sample.py"
          lang="python"
          [code]="snipSample"
          [lines]="[141, 170]"
          caption="sample.py sample_images — build (context, mask, uncond), pick ddim/ddpm, apply guidance, decode latents, map [-1,1] -> [0,1]." />
      }
      @if (stage() === 'vae') {
        <p class="tabs__prose">
          <strong>Optional latent compression.</strong> With <code>--latent</code> a
          frozen VAE squeezes each <span class="mono">32×32×3</span> image down to a
          <span class="mono">8×8×4</span> latent, and diffusion runs <em>there</em>
          instead of on pixels — exactly the trick that made Stable Diffusion cheap.
          The encode is one <code>torch.no_grad()</code> call in the training loop:
        </p>
        <app-code-ref
          file="pytorch/train.py"
          lang="python"
          [code]="snipVae"
          [lines]="[202, 216]"
          caption="train.py build_models — when --latent is set, a frozen VAE decides the channels/size diffusion runs on (4 channels, 8×8 here)." />
      }
    </div>
    <figcaption class="fig__cap">
      Each tab is a real method from the repo. Click through them top-to-bottom to
      trace one batch from data to loss, then one sample from static to picture.
    </figcaption>
  </figure>

  <h2>Run it</h2>

  <p>
    Nothing here downloads a dataset or needs a GPU. The default
    <code>shapes</code> dataset is rasterized synthetically, so a full one-epoch
    train + sample finishes in seconds. Copy the block, paste it at the repo root:
  </p>

  <app-code-ref
    file="pytorch/requirements.txt"
    lang="bash"
    [code]="snipQuickstart"
    caption="The full path: install, smoke-test, train one epoch on shapes, sample with DDIM + guidance, and run the 2-D toy. Flags verified against train.py / sample.py." />

  <p>
    Those flags are real. <code>--epochs</code>, <code>--dataset</code>,
    <code>--out</code> live in <code>train.py</code>; <code>--ckpt</code>,
    <code>--steps</code>, <code>--guidance</code>, <code>--sampler</code> live in
    <code>sample.py</code>. The dependency list itself is short on purpose — the
    point is that nothing exotic is required to reproduce the core mechanism:
  </p>

  <app-code-ref
    file="pytorch/requirements.txt"
    lang="bash"
    [code]="snipReqs"
    [lines]="[6, 10]"
    caption="pytorch/requirements.txt — five packages, CPU-friendly. torchvision is optional and only needed for --dataset mnist." />

  <h2>Same shape, different scale</h2>

  <p>
    Be honest about what you just ran. This is a <strong>teaching-scale</strong>
    model: a few hundred thousand parameters, a 32-pixel canvas, nine toy classes,
    a randomly-initialized VAE that exists only to make the latent shapes line up.
    A production text-to-image system is orders of magnitude bigger — billions of
    parameters, a VAE pretrained on hundreds of millions of images, a
    transformer denoiser, a real text encoder, and weeks on a cluster.
  </p>

  <p>
    But re-read the tabs above. <em>Architecturally it is the same machine.</em>
    The forward process is still
    <app-math expr="x_t=\\sqrt{\\bar\\alpha_t}\\,x_0+\\sqrt{1-\\bar\\alpha_t}\\,\\varepsilon" />.
    The objective is still <app-math expr="\\lVert\\varepsilon-\\hat\\varepsilon\\rVert^2" />.
    The sampler still backs out <app-math expr="\\hat{x}_0" /> and steps toward the
    previous timestep; guidance is still <app-math expr="\\hat\\varepsilon_{\\varnothing}+w\\,(\\hat\\varepsilon_c-\\hat\\varepsilon_{\\varnothing})" />.
    Scale changes the picture quality, not the plot. Once you can read these
    fifty lines, the headline systems are just bigger boxes on the same diagram.
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
    .fig__stage--diagram { padding: 1rem; }
    .fig__canvas { display: block; width: 100%; image-rendering: auto; }
    .fig__canvas--wide { max-width: 760px; aspect-ratio: 16 / 9; border-radius: var(--radius-sm); cursor: pointer; }

    .ctl { padding: 1rem 1.1rem; border-top: 1px solid var(--line); display: grid; gap: 0.75rem; }
    .ctl__row--btns { display: flex; flex-wrap: wrap; align-items: center; gap: 0.5rem; }
    .mono { font-family: var(--font-mono); color: var(--ink-0); }

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

    .fig__cap {
      padding: 0.75rem 1.1rem; border-top: 1px solid var(--line);
      font-size: 0.84rem; color: var(--ink-2); background: rgba(255,255,255,0.012);
    }
    .fig__cap strong { color: var(--ink-1); }
    .fig__cap code { font-size: 0.78em; }

    /* tabbed code reader */
    .tabs {
      display: flex; flex-wrap: wrap; gap: 0.3rem;
      padding: 0.7rem 0.7rem 0; background: var(--bg-0);
      border-bottom: 1px solid var(--line);
    }
    .tabs__tab {
      display: grid; gap: 0.1rem; text-align: left;
      padding: 0.5rem 0.8rem; cursor: pointer;
      background: var(--bg-2); border: 1px solid var(--line);
      border-bottom: none;
      border-radius: var(--radius-sm) var(--radius-sm) 0 0;
      color: var(--ink-2); transition: color .15s, background .15s, border-color .15s;
    }
    .tabs__tab:hover { color: var(--ink-0); border-color: var(--line-strong); }
    .tabs__tab--on {
      background: var(--bg-1); color: var(--ink-0);
      border-color: rgba(124,92,255,0.5);
      box-shadow: inset 0 2px 0 0 var(--plasma-a);
    }
    .tabs__lab { font-family: var(--font-display); font-weight: 600; font-size: 0.86rem; }
    .tabs__sub { font-family: var(--font-mono); font-size: 0.66rem; color: var(--ink-3); }
    .tabs__panel { padding: 1.1rem 1.1rem 0.4rem; }
    .tabs__prose { margin: 0 0 0.4rem; font-size: 0.92rem; color: var(--ink-1); }
    .tabs__prose strong { color: var(--ink-0); }
  `],
})
export class Build {
  private readonly fb = inject(FirebaseService);
  private readonly cv = viewChild.required<ElementRef<HTMLCanvasElement>>('wire');
  private readonly destroyRef = inject(DestroyRef);

  // ----- diagram state -----
  readonly flow = signal<'train' | 'sample'>('train');
  readonly running = signal(true);
  readonly pinned = signal<string | null>(null);

  // ----- tab state -----
  readonly stages: readonly Stage[] = [
    { id: 'loss', label: 'training step', sub: 'ddpm.py · p_losses', file: 'pytorch/diffusion/ddpm.py' },
    { id: 'dropout', label: 'CFG dropout', sub: 'train.py · drop_context', file: 'pytorch/train.py' },
    { id: 'ddpm', label: 'DDPM step', sub: 'ddpm.py · p_sample', file: 'pytorch/diffusion/ddpm.py' },
    { id: 'ddim', label: 'DDIM step', sub: 'ddpm.py · ddim_sample', file: 'pytorch/diffusion/ddpm.py' },
    { id: 'sample', label: 'inference', sub: 'sample.py · sample_images', file: 'pytorch/sample.py' },
    { id: 'vae', label: 'latent (VAE)', sub: 'train.py · build_models', file: 'pytorch/train.py' },
  ];
  readonly stage = signal<string>('loss');

  setFlow(f: 'train' | 'sample'): void {
    this.flow.set(f);
    this.fb.event('interact', { section: 'build', control: 'flow', value: f });
  }
  toggleRun(): void {
    this.running.update((r) => !r);
    this.fb.event('interact', { section: 'build', control: 'run', value: this.running() });
  }
  setStage(id: string): void {
    this.stage.set(id);
    this.fb.event('interact', { section: 'build', control: 'tab', value: id });
  }

  // ----- diagram geometry (computed once; boxes for the active flow) -----
  private readonly trainBoxes = [
    { id: 'dataset', t: 'dataset', s: 'shapes / mnist' },
    { id: 'vae', t: 'VAE.encode', s: '(optional) latent', dim: true },
    { id: 'forward', t: 'q_sample', s: 'add noise → xₜ' },
    { id: 'backbone', t: 'UNet | DiT', s: '+ text · CFG drop' },
    { id: 'loss', t: 'MSE loss', s: '‖ε − ε̂‖²' },
    { id: 'adamw', t: 'AdamW', s: 'step → ckpt' },
  ];
  private readonly sampleBoxes = [
    { id: 'ckpt', t: 'checkpoint', s: 'runs/model.pt' },
    { id: 'noise', t: 'x_T ∼ 𝒩(0,I)', s: 'pure static' },
    { id: 'reverse', t: 'p_sample / ddim', s: 'subtract ε̂ ×N' },
    { id: 'guidance', t: 'CFG', s: 'guidance_scale' },
    { id: 'decode', t: 'VAE.decode', s: '(optional)', dim: true },
    { id: 'image', t: 'image', s: '[0,1] grid PNG' },
  ];

  constructor() {
    afterNextRender(() => this.draw());
  }

  private draw(): void {
    const canvas = this.cv().nativeElement;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let W = 0;
    let H = 0;
    const resize = () => {
      const dpr = Math.min(devicePixelRatio || 1, 2);
      const r = canvas.getBoundingClientRect();
      W = r.width;
      H = r.height;
      canvas.width = Math.floor(r.width * dpr);
      canvas.height = Math.floor(r.height * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    resize();
    addEventListener('resize', resize);

    // map a click to the nearest box and pin it
    const onClick = (ev: MouseEvent) => {
      const r = canvas.getBoundingClientRect();
      const mx = ev.clientX - r.left;
      const my = ev.clientY - r.top;
      const layout = this.layout(W, H);
      for (const b of layout) {
        if (mx >= b.x && mx <= b.x + b.w && my >= b.y && my <= b.y + b.h) {
          this.pinned.set(b.t);
          this.fb.event('interact', { section: 'build', control: 'box', value: b.id });
          return;
        }
      }
      this.pinned.set(null);
    };
    canvas.addEventListener('click', onClick);

    let raf = 0;
    let phase = 0;
    const loop = () => {
      if (this.running()) phase += 0.012;
      this.render(ctx, W, H, phase);
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);

    this.destroyRef.onDestroy(() => {
      cancelAnimationFrame(raf);
      removeEventListener('resize', resize);
      canvas.removeEventListener('click', onClick);
    });
  }

  /** Compute box rectangles for the currently selected flow. */
  private layout(W: number, H: number) {
    const boxes = this.flow() === 'train' ? this.trainBoxes : this.sampleBoxes;
    const n = boxes.length;
    const padX = 14;
    const gap = 14;
    const usable = W - padX * 2 - gap * (n - 1);
    const w = usable / n;
    const h = Math.min(74, H * 0.34);
    const y = H / 2 - h / 2;
    return boxes.map((b, i) => ({
      ...b,
      x: padX + i * (w + gap),
      y,
      w,
      h,
      cx: padX + i * (w + gap) + w / 2,
      cy: y + h / 2,
    }));
  }

  private render(ctx: CanvasRenderingContext2D, W: number, H: number, phase: number): void {
    ctx.clearRect(0, 0, W, H);
    const layout = this.layout(W, H);
    const pinned = this.pinned();
    const plasmaA = '#7c5cff';
    const plasmaB = '#41d6ff';

    // arrows + flowing packets between consecutive boxes
    for (let i = 0; i < layout.length - 1; i++) {
      const a = layout[i];
      const b = layout[i + 1];
      const x0 = a.x + a.w;
      const x1 = b.x;
      const y = a.cy;

      ctx.strokeStyle = 'rgba(160,170,200,0.35)';
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      ctx.moveTo(x0, y);
      ctx.lineTo(x1 - 6, y);
      ctx.stroke();
      // arrowhead
      ctx.fillStyle = 'rgba(160,170,200,0.55)';
      ctx.beginPath();
      ctx.moveTo(x1 - 6, y - 4);
      ctx.lineTo(x1, y);
      ctx.lineTo(x1 - 6, y + 4);
      ctx.closePath();
      ctx.fill();

      // flowing packet (a small glowing dot riding the wire)
      const span = x1 - 6 - x0;
      const local = (phase + i * 0.18) % 1;
      const px = x0 + local * span;
      const grad = ctx.createRadialGradient(px, y, 0, px, y, 7);
      grad.addColorStop(0, plasmaB);
      grad.addColorStop(1, 'rgba(65,214,255,0)');
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(px, y, 7, 0, Math.PI * 2);
      ctx.fill();
    }

    // boxes
    for (const b of layout) {
      const isPinned = pinned === b.t;
      const dim = (b as { dim?: boolean }).dim === true && !isPinned;
      const r = 10;
      // panel
      ctx.beginPath();
      this.roundRect(ctx, b.x, b.y, b.w, b.h, r);
      ctx.fillStyle = isPinned ? 'rgba(124,92,255,0.16)' : 'rgba(18,22,34,0.92)';
      ctx.fill();
      ctx.lineWidth = isPinned ? 2 : 1;
      ctx.strokeStyle = isPinned
        ? plasmaA
        : dim
          ? 'rgba(120,130,160,0.35)'
          : 'rgba(120,130,160,0.55)';
      if (dim) ctx.setLineDash([4, 4]);
      ctx.stroke();
      ctx.setLineDash([]);

      // title
      ctx.fillStyle = dim ? 'rgba(200,205,220,0.55)' : '#eef0f6';
      ctx.font = '600 12px ui-monospace, SFMono-Regular, Menlo, monospace';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      this.fitText(ctx, b.t, b.cx, b.cy - 8, b.w - 12);
      // subtitle
      ctx.fillStyle = dim ? 'rgba(150,158,180,0.5)' : 'rgba(170,178,200,0.85)';
      ctx.font = '10px ui-monospace, SFMono-Regular, Menlo, monospace';
      this.fitText(ctx, b.s, b.cx, b.cy + 9, b.w - 10);
    }

    // flow label
    ctx.fillStyle = 'rgba(150,158,180,0.7)';
    ctx.font = '10px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    const tag = this.flow() === 'train'
      ? 'forward → loss → optimizer (only the backbone learns)'
      : 'checkpoint → reverse chain → image (no gradients)';
    ctx.fillText(tag, 6, 6);
  }

  private fitText(ctx: CanvasRenderingContext2D, text: string, x: number, y: number, max: number): void {
    let t = text;
    while (t.length > 3 && ctx.measureText(t).width > max) {
      t = t.slice(0, -2);
    }
    if (t !== text) t = t.slice(0, -1) + '…';
    ctx.fillText(t, x, y);
  }

  private roundRect(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number): void {
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }

  // ===================================================================
  // REAL code excerpts (verbatim from the repo, with true line ranges).
  // ===================================================================

  /** pytorch/diffusion/ddpm.py :81-103 */
  readonly snipPLosses = `def p_losses(
    self,
    model,
    x0: Tensor,
    t: Tensor,
    context: Optional[Tensor] = None,
    mask: Optional[Tensor] = None,
) -> Tensor:
    """L_simple at a *given* batch of timesteps \`\`t\`\`. ... """
    # 1. Sample noise eps ~ N(0, I).
    # 2. Forward-diffuse: x_t = q_sample(x0, t, eps).   (pure math)
    # 3. Ask the model to predict eps from (x_t, t, context).  (learned)
    # 4. Return MSE between the true eps and the predicted eps.
    noise = torch.randn_like(x0)
    x_t = self.schedule.q_sample(x0, t, noise)
    model_out = model(x_t, t, context, mask)
    target = noise if self.predict == "eps" else x0
    return torch.nn.functional.mse_loss(model_out, target)`;

  /** pytorch/train.py :321-337 */
  readonly snipDropout = `# --- to diffusion space (pixels, or VAE latent) ---
if vae is not None:
    with torch.no_grad():
        x0 = vae.encode(images).sample()  # [B, 4, 8, 8], already scaled
else:
    x0 = images                            # [B, 3, 32, 32]

# --- build the text context and apply CFG condition-dropout ---
context = mask = None
if text_encoder is not None:
    context, mask = labels_to_context(labels, text_encoder)
    context, mask = drop_context(
        context, mask, p=cfg.guidance_dropout, generator=gen
    )

# --- the diffusion training objective (random t, predict eps, MSE) ---
loss = diffusion.training_loss(backbone, x0, context=context, mask=mask)`;

  /** pytorch/diffusion/ddpm.py :152-183 */
  readonly snipPSample = `@torch.no_grad()
def p_sample(
    self,
    model,
    x_t: Tensor,
    t: Tensor,
    context: Optional[Tensor] = None,
    mask: Optional[Tensor] = None,
    guidance_scale: float = 1.0,
    uncond_context: Optional[Tensor] = None,
) -> Tensor:
    """One ancestral DDPM reverse step: sample x_{t-1} ~ p_theta(x_{t-1}|x_t). ... """
    sched = self.schedule
    eps = self._model_eps(model, x_t, t, context, mask, guidance_scale, uncond_context)
    x0_hat = sched.predict_x0_from_eps(x_t, t, eps)
    # Clamp the predicted x0 to a sane range; ...
    x0_hat = x0_hat.clamp(-3.0, 3.0)

    mean, _var, log_var = sched.posterior(x0_hat, x_t, t)
    noise = torch.randn_like(x_t)
    # No noise at the last step (t == 0): mask out the noise term per batch row.
    nonzero = (t != 0).float().reshape(-1, *((1,) * (x_t.dim() - 1)))
    return mean + nonzero * (0.5 * log_var).exp() * noise`;

  /** pytorch/diffusion/ddpm.py :266-288 */
  readonly snipDdim = `for idx, ti in enumerate(seq):
    t = torch.full((b,), ti, device=device, dtype=torch.long)
    eps = self._model_eps(
        model, x_t, t, context, mask, guidance_scale, uncond_context
    )
    x0_hat = sched.predict_x0_from_eps(x_t, t, eps).clamp(-3.0, 3.0)

    a_t = acp[ti]
    # The "previous" (earlier, less-noisy) timestep in our sub-sequence;
    # at the final step we target alphas_cumprod_prev = 1 (clean x0).
    t_prev = seq[idx + 1] if idx + 1 < len(seq) else -1
    a_prev = acp[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=device)

    # DDIM stochasticity term sigma (eq. 16). eta=0 -> sigma=0 -> ODE.
    sigma = (
        eta
        * torch.sqrt((1 - a_prev) / (1 - a_t))
        * torch.sqrt(1 - a_t / a_prev)
    )
    # Direction pointing to x_t, with the remaining (non-sigma) variance.
    dir_xt = torch.sqrt((1 - a_prev - sigma ** 2).clamp(min=0.0)) * eps
    noise = sigma * torch.randn_like(x_t) if eta > 0 else 0.0
    x_t = torch.sqrt(a_prev) * x0_hat + dir_xt + noise`;

  /** pytorch/sample.py :141-170 */
  readonly snipSample = `schedule = NoiseSchedule(timesteps=cfg.timesteps, kind="cosine").to(device)
diffusion = GaussianDiffusion(schedule, predict="eps")

context, mask, uncond = build_conditioning(
    cfg, text_encoder, prompt, label, n, device
)
shape = (n, cfg.in_channels, cfg.latent_size, cfg.latent_size)

# Guidance only matters when we actually have a condition to push toward.
g = guidance if context is not None else 1.0

if sampler == "ddim":
    x0 = diffusion.ddim_sample(
        backbone, shape, steps=steps, eta=0.0,
        context=context, mask=mask, guidance_scale=g, uncond_context=uncond,
        device=device,
    )
else:  # full ancestral DDPM (uses all cfg.timesteps steps)
    x0 = diffusion.sample(
        backbone, shape,
        context=context, mask=mask, guidance_scale=g, uncond_context=uncond,
        device=device, progress=True,
    )

# latent -> pixels if needed.
if vae is not None:
    x0 = vae.decode(x0)

# diffusion works in [-1, 1]; map to [0, 1] for display and clamp.
return ((x0 + 1.0) * 0.5).clamp(0.0, 1.0)`;

  /** pytorch/train.py :202-216 */
  readonly snipVae = `# --- optional VAE: decides what spatial size / channel count diffusion runs on ---
vae: Optional[VAE] = None
if cfg.latent:
    vae = VAE(in_channels=3, latent_channels=4).to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)  # treat the VAE as FROZEN (as in real latent diffusion)
    in_channels = vae.latent_channels                       # 4
    latent_size = cfg.image_size // vae.downsample_factor    # 32 / 4 = 8
else:
    in_channels = 3
    latent_size = cfg.image_size

cfg.in_channels = in_channels
cfg.latent_size = latent_size`;

  /** Quickstart commands — flags verified against train.py / sample.py CLIs. */
  readonly snipQuickstart = `# from the repo root:
pip install -r pytorch/requirements.txt
python -m pytorch.tests.smoke
python -m pytorch.train  --epochs 1 --dataset shapes --out runs/model.pt
python -m pytorch.sample --ckpt runs/model.pt --steps 50 --guidance 4
python -m pytorch.toy.toy_diffusion_2d --target moons`;

  /** pytorch/requirements.txt :6-10 */
  readonly snipReqs = `torch>=2.0          # the whole repo is PyTorch; CPU build is fine (no CUDA assumed)
numpy>=1.23         # toy 2-D data generation, misc array work
matplotlib>=3.6     # headless (Agg) plotting for sample grids and the toy scatter PNG
tqdm>=4.64          # optional progress bars during sampling (degrades gracefully if absent)
pillow>=9.0         # image I/O helper (PIL); used for optional image loading/saving`;
}
