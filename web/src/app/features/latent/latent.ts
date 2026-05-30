import {
  Component,
  ChangeDetectionStrategy,
  ElementRef,
  afterNextRender,
  inject,
  DestroyRef,
  viewChild,
  signal,
  computed,
  effect,
} from '@angular/core';
import { Chapter } from '../../shared/chapter';
import { CodeRef } from '../../shared/code-ref';
import { Math as MathTex } from '../../shared/math';
import { FirebaseService } from '../../core/firebase.service';

/**
 * Chapter 02 — /latent
 * Why Stable Diffusion denoises in a tiny latent grid.
 *
 * Two interactive canvas figures:
 *   FIG 1 — encode → z → decode, with a downsample factor (4 / 8) and latent
 *           channel control, plus live "numbers" counts and the compression ratio.
 *   FIG 2 — a latent viewed as small per-channel heatmaps, with resample and a
 *           two-latent interpolation slider (an analogy, since the real VAE is
 *           learned and non-linear).
 *
 * All quoted code is from pytorch/diffusion/vae.py.
 */
@Component({
  selector: 'app-latent',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [Chapter, CodeRef, MathTex],
  template: `
    <app-chapter slug="latent">
      <p class="lede">
        Pixels are an extravagant way to store a picture. A
        <strong>512×512</strong> RGB image is
        <span class="num">{{ pixelImageNumbers().toLocaleString() }}</span> raw numbers,
        and a diffusion model would have to push <em>all of them</em> through a giant
        network <em>at every one of the dozens of denoising steps</em>. Stable Diffusion's
        trick is to never touch pixels during diffusion at all. It first squeezes the image
        into a small <strong>latent grid</strong> with a learned compressor — a VAE — and
        runs the whole noisy-to-clean process in that tiny space. Only at the very end does
        it decode back to pixels.
      </p>

      <h2>The squeeze, made tangible</h2>
      <p>
        A VAE has two halves. The <strong>encoder</strong> <app-math expr="E" /> maps an
        image <app-math expr="x" /> to a much smaller spatial grid of numbers
        <app-math expr="z" />; the <strong>decoder</strong> <app-math expr="D" /> maps that
        grid back to an image <app-math expr="\\hat{x}" />. The shrink is purely
        <em>spatial</em>: a downsample factor <app-math expr="f" /> turns an
        <app-math expr="H\\times W" /> image into an
        <app-math expr="\\tfrac{H}{f}\\times\\tfrac{W}{f}" /> latent, so the area drops by
        <app-math expr="f^2" />. Real SD uses <app-math expr="f=8" /> and
        <app-math expr="4" /> latent channels.
      </p>

      <div class="fig">
        <div class="fig__controls">
          <div class="ctrl">
            <span class="ctrl__label">downsample factor <code>f</code></span>
            <div class="seg">
              @for (opt of fOptions; track opt) {
                <button
                  type="button"
                  class="seg__btn"
                  [class.seg__btn--on]="fFactor() === opt"
                  (click)="setFactor(opt)"
                >
                  {{ opt }}×
                </button>
              }
            </div>
          </div>
          <div class="ctrl">
            <span class="ctrl__label">latent channels <code>C</code></span>
            <div class="seg">
              @for (opt of cOptions; track opt) {
                <button
                  type="button"
                  class="seg__btn"
                  [class.seg__btn--on]="latentCh() === opt"
                  (click)="setChannels(opt)"
                >
                  {{ opt }}
                </button>
              }
            </div>
          </div>
          <button type="button" class="btn" (click)="resampleScene()">↻ new image</button>
        </div>

        <canvas #cv1 class="fig__canvas"></canvas>

        <div class="counts">
          <div class="count">
            <span class="count__k">pixels in</span>
            <span class="count__v">512 × 512 × 3</span>
            <span class="count__n">{{ pixelImageNumbers().toLocaleString() }}</span>
          </div>
          <div class="count count--arrow">→</div>
          <div class="count count--accent">
            <span class="count__k">latent z</span>
            <span class="count__v"
              >{{ latentSide() }} × {{ latentSide() }} × {{ latentCh() }}</span
            >
            <span class="count__n">{{ latentNumbers().toLocaleString() }}</span>
          </div>
          <div class="count count--ratio">
            <span class="count__k">smaller by</span>
            <span class="count__big gradient-text">≈{{ ratio() }}×</span>
          </div>
        </div>
        <p class="fig__cap">
          Figure 1 — <strong>encode → z → decode.</strong> The image is squeezed into a
          {{ latentSide() }}×{{ latentSide() }}×{{ latentCh() }} latent and expanded back.
          The block-pixelation is only an <em>analogy</em> for the spatial squeeze: the real
          VAE encoder is a learned convolutional network, not literal averaging.
        </p>
      </div>

      <div class="callout">
        <span class="callout__tag tag tag--accent">be honest</span>
        <p>
          The middle frame above looks like a coarse mosaic because that is the cleanest way
          to <em>show</em> "fewer, larger cells." But a VAE latent is not a downscaled
          thumbnail. Each latent cell is a learned <em>perceptual</em> code — a vector that
          the decoder knows how to expand into rich texture. That is why
          <app-math expr="f=8" /> can throw away 98% of the numbers and still reconstruct a
          crisp image: the decoder fills the detail back in.
        </p>
      </div>

      <h2>What the encoder actually outputs</h2>
      <p>
        It would be brittle to make the encoder emit a single fixed latent. Instead it emits
        a tiny <strong>probability distribution</strong> per cell — a diagonal Gaussian with
        a mean and a (log-)variance — and we <em>sample</em> from it. That is the
        "variational" in VAE. In code, the encoder produces
        <code>2·latent_channels</code> feature maps and splits them into
        <code>mean</code> and <code>logvar</code>:
      </p>

      <app-code-ref
        file="pytorch/diffusion/vae.py"
        lang="python"
        [code]="snipGaussian"
        caption="The per-cell posterior q(z|x). logvar (not std) is what the net emits — unconstrained, and exp() keeps the variance positive."
        [lines]="[86, 106]"
      />

      <p>
        Sampling uses the <strong>reparameterization trick</strong>:
        <app-math expr="z = \\mu + \\sigma \\odot \\varepsilon" /> with
        <app-math expr="\\varepsilon \\sim \\mathcal{N}(0, I)" />. Pushing the randomness into
        <app-math expr="\\varepsilon" /> keeps it off the path that carries gradients, so the
        whole thing stays trainable by backprop. At inference we usually skip the noise and
        just take the mode (the mean) for a stable, reproducible latent.
      </p>

      <h2>Why <em>variational</em>? The two-term objective</h2>
      <p>
        Training the VAE balances two pressures. A
        <strong>reconstruction</strong> term wants <app-math expr="D(E(x))" /> to look like
        <app-math expr="x" />. A <strong>KL</strong> term wants every per-image posterior
        <app-math expr="q(z\\mid x)=\\mathcal{N}(\\mu,\\sigma^2)" /> to stay close to a plain
        unit Gaussian <app-math expr="\\mathcal{N}(0,I)" />:
      </p>

      <app-math
        display
        expr="\\mathcal{L}_{\\text{VAE}} \\;=\\; \\underbrace{\\big\\lVert x - D(z)\\big\\rVert^2}_{\\text{reconstruction}} \\;+\\; \\beta\\,\\underbrace{D_{\\mathrm{KL}}\\!\\big(\\,\\mathcal{N}(\\mu,\\sigma^2)\\,\\Vert\\,\\mathcal{N}(0,I)\\,\\big)}_{\\text{keep the latent well-behaved}}"
      />

      <p>
        Without the KL term a plain autoencoder is free to carve out a spiky, irregular latent
        space — great for reconstruction, hopeless to put a smooth Gaussian diffusion prior on.
        The KL pulls the latent toward unit scale and unit variance so the diffusion model can
        treat <app-math expr="z" /> as "just some Gaussian-ish tensor." For a diagonal Gaussian
        the divergence has a closed form, summed over all latent dimensions:
      </p>

      <app-math
        display
        expr="D_{\\mathrm{KL}} \\;=\\; \\tfrac12 \\sum \\big(\\mu^2 + \\sigma^2 - 1 - \\log \\sigma^2\\big)"
      />

      <app-code-ref
        file="pytorch/diffusion/vae.py"
        lang="python"
        [code]="snipKl"
        caption="The closed-form KL, exactly the formula above. The weight β is kept tiny by the trainer — the VAE is a compressor first, a generator second."
        [lines]="[116, 127]"
      />

      <h2>Encode and decode live in "diffusion space"</h2>
      <p>
        One more subtlety. Even after the KL, the raw latent's standard deviation is not
        exactly one. Stable Diffusion <em>measured</em> it and bakes in a single rescale
        constant — the famous <app-math expr="0.18215" /> — so the tensor handed to diffusion
        has roughly unit variance. In this repo, <code>encode</code> folds that constant in and
        <code>decode</code> divides it back out, so the rest of the codebase never has to think
        about it:
      </p>

      <app-code-ref
        file="pytorch/diffusion/vae.py"
        lang="python"
        [code]="snipEncodeDecode"
        caption="encode / decode operate in diffusion space: encode multiplies the Gaussian by scale_factor, decode divides it back. They are exact inverses w.r.t. the constant."
        [lines]="[383, 408]"
      />

      <p>
        Scaling a Gaussian by a constant <app-math expr="s" /> sends
        <app-math expr="\\mu \\to s\\mu" /> and <app-math expr="\\sigma^2 \\to s^2\\sigma^2" />,
        i.e. <app-math expr="\\log\\sigma^2 \\to \\log\\sigma^2 + 2\\log s" /> — which is exactly
        the two lines that adjust <code>mean</code> and <code>logvar</code> in
        <code>encode</code>. Applying it to the distribution (not just to a sample) keeps
        <code>.sample()</code>, <code>.mode()</code> and <code>.kl()</code> mutually consistent.
      </p>

      <h2>A latent, channel by channel</h2>
      <p>
        So what does a latent actually look like? It is a small stack of grids — for SD,
        <strong>{{ latentSide() }}×{{ latentSide() }}</strong> with
        <strong>{{ latentCh() }}</strong> channels. Below each channel is its own heatmap.
        Hit <em>resample</em> to draw a fresh latent from the encoder's Gaussian (here, a toy
        smooth field standing in for a learned code), or drag the slider to
        <strong>interpolate</strong> between two latents and watch the decoded image morph.
      </p>

      <div class="fig">
        <div class="fig__controls">
          <button type="button" class="btn" (click)="resampleLatents()">↻ resample z₀, z₁</button>
          <div class="ctrl ctrl--grow">
            <span class="ctrl__label"
              >interpolate&nbsp; z = (1−t)·z₀ + t·z₁&nbsp;&nbsp;<span class="ctrl__t"
                >t = {{ tDisplay() }}</span
              ></span
            >
            <input
              type="range"
              min="0"
              max="1"
              step="0.01"
              [value]="tMix()"
              (input)="onMix($event)"
            />
          </div>
        </div>

        <div class="latent-row">
          <div class="latent-block">
            <canvas #cv2 class="fig__canvas fig__canvas--heat"></canvas>
            <span class="latent-block__cap">{{ latentCh() }} latent channels (heatmaps)</span>
          </div>
          <div class="latent-arrow">
            <span class="latent-arrow__d">decode</span>
            <span class="latent-arrow__a">→</span>
          </div>
          <div class="latent-block">
            <canvas #cv3 class="fig__canvas fig__canvas--decode"></canvas>
            <span class="latent-block__cap">decoded image (analogy)</span>
          </div>
        </div>
        <p class="fig__cap">
          Figure 2 — A latent is a compact, multi-channel code, not a picture. Each heatmap is
          one channel of <app-math expr="z" />; together they hold everything the decoder needs.
          The decode shown here is a smooth-field stand-in — the point is that a continuous walk
          in latent space produces a continuous walk in image space, which is exactly what lets
          diffusion sculpt images by moving smoothly through <app-math expr="z" />.
        </p>
      </div>

      <h2>The latent-diffusion idea, in one line</h2>
      <p>
        Put it together and the recipe is almost embarrassingly simple. Encode the image to a
        latent, scale it, run the entire forward-and-reverse diffusion process on that small
        tensor, and decode only the final result:
      </p>

      <app-math
        display
        expr="z_0 = s\\cdot E(x) \\quad\\xrightarrow{\\;\\text{diffuse \\& denoise in latent space}\\;}\\quad \\hat{z}_0 \\quad\\xrightarrow{\\;D\\;}\\quad \\hat{x} = D(\\hat{z}_0)"
      />

      <p>
        The denoiser is the same kind of network either way — a U-Net or a diffusion Transformer.
        The only thing that changed is the <em>size</em> of what it processes. At
        <app-math expr="f=8" />, a step costs roughly <span class="num">{{ ratio() }}×</span> less
        than working in pixels, which is precisely what made high-resolution diffusion cheap
        enough to train on ordinary hardware. Diffusion never sees a pixel; the VAE is the bridge
        on either end, trained once and then frozen.
      </p>

      <div class="takeaways panel">
        <h3>Takeaways</h3>
        <ul>
          <li>
            The VAE compresses <em>spatially</em> by a factor <app-math expr="f" />; area — and
            therefore diffusion cost — drops by <app-math expr="f^2" /> (times a small channel
            correction).
          </li>
          <li>
            The latent is a <strong>learned perceptual code</strong>, not a thumbnail. The
            decoder reconstructs detail the latent never stored explicitly.
          </li>
          <li>
            "Variational" = the encoder outputs a Gaussian; a tiny KL term keeps the latent
            smooth and unit-scaled so a diffusion prior fits cleanly.
          </li>
          <li>
            Diffusion runs entirely on <app-math expr="z = s\\cdot E(x)" />; only the final latent
            is decoded with <app-math expr="D" />.
          </li>
        </ul>
      </div>
    </app-chapter>
  `,
  styles: [
    `
      :host {
        display: block;
      }
      .lede {
        font-size: var(--step-1);
        color: var(--ink-1);
      }
      .lede strong {
        color: var(--ink-0);
      }
      .num {
        font-family: var(--font-mono);
        color: var(--plasma-b);
        font-weight: 600;
      }

      /* ---- figure shell ---- */
      .fig {
        margin: 1.8rem 0 2.2rem;
        padding: 1.1rem;
        border: 1px solid var(--line);
        border-radius: var(--radius);
        background: linear-gradient(180deg, rgba(255, 255, 255, 0.025), rgba(255, 255, 255, 0)),
          var(--bg-1);
        box-shadow: var(--shadow-1);
      }
      .fig__controls {
        display: flex;
        flex-wrap: wrap;
        align-items: flex-end;
        gap: 0.9rem 1.4rem;
        margin-bottom: 1rem;
      }
      .fig__canvas {
        display: block;
        width: 100%;
        height: 300px;
        border-radius: var(--radius-sm);
        background: var(--bg-0);
        border: 1px solid var(--line);
      }
      .fig__canvas--heat {
        height: 240px;
      }
      .fig__canvas--decode {
        height: 240px;
      }
      .fig__cap {
        margin: 0.85rem 0 0;
        color: var(--ink-2);
        font-size: 0.85rem;
        line-height: 1.55;
      }
      .fig__cap strong {
        color: var(--ink-1);
      }

      /* ---- controls ---- */
      .ctrl {
        display: flex;
        flex-direction: column;
        gap: 0.4rem;
      }
      .ctrl--grow {
        flex: 1 1 260px;
        min-width: 220px;
      }
      .ctrl__label {
        font-family: var(--font-mono);
        font-size: 0.72rem;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: var(--ink-2);
      }
      .ctrl__label code {
        color: var(--plasma-a);
      }
      .ctrl__t {
        color: var(--plasma-b);
        font-weight: 600;
        text-transform: none;
        letter-spacing: 0;
      }
      .seg {
        display: inline-flex;
        gap: 0.3rem;
        padding: 0.25rem;
        background: var(--bg-3);
        border: 1px solid var(--line);
        border-radius: 999px;
      }
      .seg__btn {
        font-family: var(--font-mono);
        font-size: 0.82rem;
        padding: 0.35em 0.9em;
        border-radius: 999px;
        border: 0;
        background: transparent;
        color: var(--ink-2);
        cursor: pointer;
        transition: color 0.15s var(--ease), background 0.15s var(--ease);
      }
      .seg__btn:hover {
        color: var(--ink-0);
      }
      .seg__btn--on {
        color: #0a0a12;
        background: var(--grad-plasma);
        font-weight: 600;
      }

      /* ---- count strip ---- */
      .counts {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 0.7rem;
        margin-top: 1rem;
      }
      .count {
        display: flex;
        flex-direction: column;
        gap: 0.15rem;
        padding: 0.65rem 0.9rem;
        border: 1px solid var(--line);
        border-radius: var(--radius-sm);
        background: var(--bg-2);
        min-width: 120px;
      }
      .count--accent {
        border-color: rgba(124, 92, 255, 0.45);
        background: var(--accent-soft);
      }
      .count--arrow,
      .count--ratio {
        background: transparent;
        border: 0;
        min-width: auto;
      }
      .count--arrow {
        font-size: 1.4rem;
        color: var(--ink-3);
        padding: 0;
      }
      .count__k {
        font-family: var(--font-mono);
        font-size: 0.64rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--ink-3);
      }
      .count__v {
        font-family: var(--font-mono);
        font-size: 0.82rem;
        color: var(--ink-1);
      }
      .count__n {
        font-family: var(--font-mono);
        font-size: 1.05rem;
        font-weight: 600;
        color: var(--ink-0);
      }
      .count--accent .count__n {
        color: #cdbcff;
      }
      .count__big {
        font-family: var(--font-display);
        font-size: 1.9rem;
        font-weight: 700;
        line-height: 1;
      }

      /* ---- callout ---- */
      .callout {
        display: flex;
        gap: 1rem;
        align-items: flex-start;
        margin: 1.6rem 0;
        padding: 1rem 1.2rem;
        border-left: 2px solid var(--plasma-a);
        border-radius: var(--radius-sm);
        background: rgba(124, 92, 255, 0.07);
      }
      .callout__tag {
        flex: none;
        margin-top: 0.15rem;
      }
      .callout p {
        margin: 0;
      }

      /* ---- figure 2 layout ---- */
      .latent-row {
        display: grid;
        grid-template-columns: 1fr auto 1fr;
        gap: 0.9rem;
        align-items: center;
      }
      .latent-block {
        display: flex;
        flex-direction: column;
        gap: 0.4rem;
      }
      .latent-block__cap {
        font-family: var(--font-mono);
        font-size: 0.68rem;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: var(--ink-3);
        text-align: center;
      }
      .latent-arrow {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 0.1rem;
        color: var(--ink-2);
      }
      .latent-arrow__d {
        font-family: var(--font-mono);
        font-size: 0.62rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--ink-3);
      }
      .latent-arrow__a {
        font-size: 1.5rem;
        color: var(--plasma-a);
      }

      /* ---- takeaways ---- */
      .takeaways {
        margin: 2.4rem 0 1rem;
        padding: 1.2rem 1.4rem;
      }
      .takeaways h3 {
        margin-top: 0;
      }
      .takeaways ul {
        margin: 0;
        padding-left: 1.1rem;
        display: grid;
        gap: 0.6rem;
      }
      .takeaways li {
        color: var(--ink-1);
      }
      .takeaways strong {
        color: var(--ink-0);
      }

      @media (max-width: 640px) {
        .latent-row {
          grid-template-columns: 1fr;
        }
        .latent-arrow__a {
          transform: rotate(90deg);
        }
      }
    `,
  ],
})
export class Latent {
  private readonly fb = inject(FirebaseService);
  private readonly destroyRef = inject(DestroyRef);

  private readonly cv1 = viewChild.required<ElementRef<HTMLCanvasElement>>('cv1');
  private readonly cv2 = viewChild.required<ElementRef<HTMLCanvasElement>>('cv2');
  private readonly cv3 = viewChild.required<ElementRef<HTMLCanvasElement>>('cv3');

  // ---- control options ----
  readonly fOptions = [4, 8] as const;
  readonly cOptions = [4, 8, 16] as const;

  // ============================================================ //
  // Real snippets from pytorch/diffusion/vae.py (line-accurate). //
  // ============================================================ //
  readonly snipGaussian = `def __init__(self, mean: Tensor, logvar: Tensor) -> None:
    self.mean = mean
    # Clamp logvar to a sane range so var = exp(logvar) never under/overflows.
    # ...
    self.logvar = torch.clamp(logvar, -30.0, 20.0)
    self.std = torch.exp(0.5 * self.logvar)
    self.var = torch.exp(self.logvar)

def sample(self, generator: torch.Generator | None = None) -> Tensor:
    """Reparameterized sample: z = mean + std * eps, eps ~ N(0, I).
    # ...
    """
    eps = torch.randn(
        self.mean.shape,
        generator=generator,
        device=self.mean.device,
        dtype=self.mean.dtype,
    )
    return self.mean + self.std * eps`;

  readonly snipKl = `def kl(self) -> Tensor:
    """KL( N(mean, var) || N(0, I) ), summed over latent dims, mean over batch.

    Closed form for a diagonal Gaussian vs the unit Gaussian:
        KL = 0.5 * sum( mean^2 + var - 1 - logvar )
    # ...
    """
    # Per-element KL contribution.
    per_elem = 0.5 * (self.mean.pow(2) + self.var - 1.0 - self.logvar)
    # Sum over channel + spatial dims, average over the batch.
    return per_elem.flatten(start_dim=1).sum(dim=1).mean()`;

  readonly snipEncodeDecode = `def encode(self, x: Tensor) -> DiagonalGaussian:
    """Image -> posterior q(z|x) in *diffusion space*.
    # ...
    """
    h = self.encoder(x)
    mean, logvar = torch.chunk(h, 2, dim=1)
    s = self.scale_factor
    mean = mean * s
    logvar = logvar + 2.0 * math.log(s)
    return DiagonalGaussian(mean, logvar)

def decode(self, z: Tensor) -> Tensor:
    """Latent (in diffusion space) -> reconstructed image.
    # ...
    """
    z = z / self.scale_factor
    return self.decoder(z)`;

  // ---- reactive state ----
  readonly fFactor = signal<number>(8);
  readonly latentCh = signal<number>(4);
  readonly sceneSeed = signal<number>(0.137);
  readonly tMix = signal<number>(0);
  private readonly latentSeedA = signal<number>(0.42);
  private readonly latentSeedB = signal<number>(0.91);

  // The reference image is a fixed 512×512×3 (RGB).
  private readonly imageSide = 512;
  readonly pixelImageNumbers = computed(() => this.imageSide * this.imageSide * 3);
  readonly latentSide = computed(() => Math.round(this.imageSide / this.fFactor()));
  readonly latentNumbers = computed(
    () => this.latentSide() * this.latentSide() * this.latentCh(),
  );
  readonly ratio = computed(() =>
    Math.round(this.pixelImageNumbers() / this.latentNumbers()),
  );
  readonly tDisplay = computed(() => this.tMix().toFixed(2));

  constructor() {
    afterNextRender(() => {
      this.drawFig1();
      this.startFig2Loop();
    });
    // Figure 1 is static-per-control: redraw when any input changes.
    effect(() => {
      // touch the signals so the effect re-runs on change
      this.fFactor();
      this.latentCh();
      this.sceneSeed();
      this.redrawFig1();
    });
  }

  // ---- control handlers ----
  setFactor(f: number): void {
    this.fFactor.set(f);
    this.fb.event('interact', { section: 'latent', control: 'downsample_factor', value: f });
  }

  setChannels(c: number): void {
    this.latentCh.set(c);
    this.fb.event('interact', { section: 'latent', control: 'latent_channels', value: c });
  }

  resampleScene(): void {
    this.sceneSeed.set(Math.random() * 1000 + 1);
    this.fb.event('interact', { section: 'latent', control: 'new_image' });
  }

  resampleLatents(): void {
    this.latentSeedA.set(Math.random() * 1000 + 1);
    this.latentSeedB.set(Math.random() * 1000 + 1);
    this.fb.event('interact', { section: 'latent', control: 'resample_latents' });
  }

  onMix(ev: Event): void {
    const v = Number((ev.target as HTMLInputElement).value);
    this.tMix.set(v);
  }

  // ============================================================ //
  // Deterministic pseudo-random helpers (so a "seed" is stable). //
  // ============================================================ //
  private hash(x: number, y: number, seed: number): number {
    const s = Math.sin(x * 127.1 + y * 311.7 + seed * 13.37) * 43758.5453;
    return s - Math.floor(s);
  }

  /** Smooth value-noise field in [0,1], used to fake an "image" / latent channel. */
  private field(u: number, v: number, seed: number, freq: number): number {
    const x = u * freq;
    const y = v * freq;
    const xi = Math.floor(x);
    const yi = Math.floor(y);
    const xf = x - xi;
    const yf = y - yi;
    const sx = xf * xf * (3 - 2 * xf);
    const sy = yf * yf * (3 - 2 * yf);
    const n00 = this.hash(xi, yi, seed);
    const n10 = this.hash(xi + 1, yi, seed);
    const n01 = this.hash(xi, yi + 1, seed);
    const n11 = this.hash(xi + 1, yi + 1, seed);
    const nx0 = n00 + sx * (n10 - n00);
    const nx1 = n01 + sx * (n11 - n01);
    return nx0 + sy * (nx1 - nx0);
  }

  /** A plasma-ish RGB for a scalar in [0,1]. */
  private heat(t: number): [number, number, number] {
    const c = Math.max(0, Math.min(1, t));
    // violet -> cyan -> magenta gradient (matches the site accent)
    const a = [124, 92, 255];
    const b = [65, 214, 255];
    const d = [255, 92, 138];
    let r: number, g: number, bl: number;
    if (c < 0.5) {
      const k = c / 0.5;
      r = a[0] + k * (b[0] - a[0]);
      g = a[1] + k * (b[1] - a[1]);
      bl = a[2] + k * (b[2] - a[2]);
    } else {
      const k = (c - 0.5) / 0.5;
      r = b[0] + k * (d[0] - b[0]);
      g = b[1] + k * (d[1] - b[1]);
      bl = b[2] + k * (d[2] - b[2]);
    }
    return [r, g, bl];
  }

  // ============================================================ //
  // FIGURE 1 — encode → z → decode                                //
  // ============================================================ //
  private fig1Ctx: CanvasRenderingContext2D | null = null;
  private fig1Resize: (() => void) | null = null;

  private drawFig1(): void {
    const canvas = this.cv1().nativeElement;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    this.fig1Ctx = ctx;
    const resize = () => {
      const dpr = Math.min(devicePixelRatio || 1, 2);
      const r = canvas.getBoundingClientRect();
      canvas.width = Math.floor(r.width * dpr);
      canvas.height = Math.floor(r.height * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      this.renderFig1();
    };
    this.fig1Resize = resize;
    resize();
    addEventListener('resize', resize);
    this.destroyRef.onDestroy(() => removeEventListener('resize', resize));
  }

  private redrawFig1(): void {
    if (this.fig1Ctx) this.renderFig1();
  }

  /** Render three panels: original image, coarse latent grid, decoded image. */
  private renderFig1(): void {
    const ctx = this.fig1Ctx;
    const canvas = this.cv1().nativeElement;
    if (!ctx) return;
    const dpr = Math.min(devicePixelRatio || 1, 2);
    const W = canvas.width / dpr;
    const H = canvas.height / dpr;
    ctx.clearRect(0, 0, W, H);

    const pad = 16;
    const labelH = 22;
    const gap = 14;
    const panelW = (W - pad * 2 - gap * 2) / 3;
    const top = pad + labelH;
    const panelH = Math.min(panelW, H - top - pad - 16);
    const size = panelH;
    const x0 = pad + (panelW - size) / 2;
    const x1 = pad + panelW + gap + (panelW - size) / 2;
    const x2 = pad + (panelW + gap) * 2 + (panelW - size) / 2;
    const seed = this.sceneSeed();

    // Number of latent "cells" we visualize (a small grid that grows with f).
    const cells = this.fFactor() === 4 ? 16 : 8;

    // --- panel 1: the "image" (a fine smooth field) ---
    this.renderField(ctx, x0, top, size, seed, 7, 1);
    this.frame(ctx, x0, top, size);
    this.label(ctx, x0, top, size, labelH, 'image  x  (512²×3)', '#c0c6d6');

    // --- panel 2: the latent grid (block-downsampled — analogy) ---
    this.renderField(ctx, x1, top, size, seed, 7, cells);
    // overlay a grid to emphasize "fewer, larger cells"
    ctx.save();
    ctx.strokeStyle = 'rgba(255,255,255,0.10)';
    ctx.lineWidth = 1;
    for (let i = 1; i < cells; i++) {
      const p = (size / cells) * i;
      ctx.beginPath();
      ctx.moveTo(x1 + p, top);
      ctx.lineTo(x1 + p, top + size);
      ctx.moveTo(x1, top + p);
      ctx.lineTo(x1 + size, top + p);
      ctx.stroke();
    }
    ctx.restore();
    this.frame(ctx, x1, top, size, 'rgba(124,92,255,0.55)');
    this.label(
      ctx,
      x1,
      top,
      size,
      labelH,
      `z  (${this.latentSide()}²×${this.latentCh()})`,
      '#cdbcff',
    );

    // --- panel 3: decoded image (the field again — decoder restores detail) ---
    this.renderField(ctx, x2, top, size, seed, 7, 1);
    this.frame(ctx, x2, top, size);
    this.label(ctx, x2, top, size, labelH, 'decoded  x̂  (512²×3)', '#c0c6d6');

    // --- arrows + operator labels between panels ---
    this.arrow(ctx, x0 + size, x1, top + size / 2, 'E', '#7c5cff');
    this.arrow(ctx, x1 + size, x2, top + size / 2, 'D', '#41d6ff');
  }

  /** Draw an RGB field into a square; `res` = number of cells per side (1 ⇒ fine). */
  private renderField(
    ctx: CanvasRenderingContext2D,
    x: number,
    y: number,
    size: number,
    seed: number,
    freq: number,
    res: number,
  ): void {
    if (res <= 1) {
      // fine field: draw per ~2px
      const step = Math.max(2, Math.floor(size / 96));
      for (let py = 0; py < size; py += step) {
        for (let px = 0; px < size; px += step) {
          const u = px / size;
          const v = py / size;
          const t = this.field(u, v, seed, freq);
          const hue = this.field(u, v, seed + 5, freq * 0.6);
          const [r, g, b] = this.heat((t * 0.6 + hue * 0.4));
          ctx.fillStyle = `rgb(${r | 0},${g | 0},${b | 0})`;
          ctx.fillRect(x + px, y + py, step, step);
        }
      }
    } else {
      // coarse: average the field over each cell, fill blocks
      const cell = size / res;
      for (let cy = 0; cy < res; cy++) {
        for (let cx = 0; cx < res; cx++) {
          const u = (cx + 0.5) / res;
          const v = (cy + 0.5) / res;
          const t = this.field(u, v, seed, freq);
          const hue = this.field(u, v, seed + 5, freq * 0.6);
          const [r, g, b] = this.heat(t * 0.6 + hue * 0.4);
          ctx.fillStyle = `rgb(${r | 0},${g | 0},${b | 0})`;
          ctx.fillRect(x + cx * cell, y + cy * cell, Math.ceil(cell), Math.ceil(cell));
        }
      }
    }
  }

  private frame(
    ctx: CanvasRenderingContext2D,
    x: number,
    y: number,
    size: number,
    stroke = 'rgba(255,255,255,0.14)',
  ): void {
    ctx.save();
    ctx.strokeStyle = stroke;
    ctx.lineWidth = 1;
    ctx.strokeRect(x + 0.5, y + 0.5, size - 1, size - 1);
    ctx.restore();
  }

  private label(
    ctx: CanvasRenderingContext2D,
    x: number,
    y: number,
    size: number,
    labelH: number,
    text: string,
    color: string,
  ): void {
    ctx.save();
    ctx.fillStyle = color;
    ctx.font = '11px "JetBrains Mono", monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'alphabetic';
    ctx.fillText(text, x + size / 2, y - 7);
    ctx.restore();
  }

  private arrow(
    ctx: CanvasRenderingContext2D,
    xFrom: number,
    xTo: number,
    y: number,
    glyph: string,
    color: string,
  ): void {
    const cx = (xFrom + xTo) / 2;
    ctx.save();
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(xFrom + 3, y);
    ctx.lineTo(xTo - 6, y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(xTo - 6, y - 4);
    ctx.lineTo(xTo - 1, y);
    ctx.lineTo(xTo - 6, y + 4);
    ctx.closePath();
    ctx.fill();
    ctx.font = 'italic 13px "JetBrains Mono", monospace';
    ctx.textAlign = 'center';
    ctx.fillText(glyph, cx, y - 8);
    ctx.restore();
  }

  // ============================================================ //
  // FIGURE 2 — latent heatmaps + decoded morph (RAF loop)         //
  // ============================================================ //
  private startFig2Loop(): void {
    const heatCanvas = this.cv2().nativeElement;
    const decCanvas = this.cv3().nativeElement;
    const hctx = heatCanvas.getContext('2d');
    const dctx = decCanvas.getContext('2d');
    if (!hctx || !dctx) return;

    const resize = () => {
      const dpr = Math.min(devicePixelRatio || 1, 2);
      for (const [c, x] of [
        [heatCanvas, hctx],
        [decCanvas, dctx],
      ] as const) {
        const r = c.getBoundingClientRect();
        c.width = Math.floor(r.width * dpr);
        c.height = Math.floor(r.height * dpr);
        x.setTransform(dpr, 0, 0, dpr, 0, 0);
      }
    };
    resize();
    addEventListener('resize', resize);

    let raf = 0;
    const loop = () => {
      this.renderHeatmaps(hctx, heatCanvas);
      this.renderDecoded(dctx, decCanvas);
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);

    this.destroyRef.onDestroy(() => {
      cancelAnimationFrame(raf);
      removeEventListener('resize', resize);
    });
  }

  /** Mixed scalar latent value at (cell u,v) for channel ch, interpolating z0→z1. */
  private latentValue(u: number, v: number, ch: number): number {
    const t = this.tMix();
    const a = this.field(u, v, this.latentSeedA() + ch * 17.3, 4 + ch);
    const b = this.field(u, v, this.latentSeedB() + ch * 17.3, 4 + ch);
    return a + t * (b - a);
  }

  private renderHeatmaps(ctx: CanvasRenderingContext2D, canvas: HTMLCanvasElement): void {
    const dpr = Math.min(devicePixelRatio || 1, 2);
    const W = canvas.width / dpr;
    const H = canvas.height / dpr;
    ctx.clearRect(0, 0, W, H);

    const ch = this.latentCh();
    // lay channels out in a grid (e.g. 4 -> 2x2, 8 -> 4x2, 16 -> 4x4)
    const cols = ch <= 4 ? 2 : 4;
    const rows = Math.ceil(ch / cols);
    const gap = 10;
    const pad = 8;
    const tileW = (W - pad * 2 - gap * (cols - 1)) / cols;
    const tileH = (H - pad * 2 - gap * (rows - 1)) / rows;
    const grid = 8; // latent is 8x8 per channel (SD convention)

    for (let i = 0; i < ch; i++) {
      const col = i % cols;
      const row = Math.floor(i / cols);
      const tx = pad + col * (tileW + gap);
      const ty = pad + row * (tileH + gap);
      const cellW = tileW / grid;
      const cellH = tileH / grid;
      for (let gy = 0; gy < grid; gy++) {
        for (let gx = 0; gx < grid; gx++) {
          const u = (gx + 0.5) / grid;
          const v = (gy + 0.5) / grid;
          const val = this.latentValue(u, v, i);
          const [r, g, b] = this.heat(val);
          ctx.fillStyle = `rgb(${r | 0},${g | 0},${b | 0})`;
          ctx.fillRect(tx + gx * cellW, ty + gy * cellH, Math.ceil(cellW), Math.ceil(cellH));
        }
      }
      ctx.save();
      ctx.strokeStyle = 'rgba(255,255,255,0.12)';
      ctx.lineWidth = 1;
      ctx.strokeRect(tx + 0.5, ty + 0.5, tileW - 1, tileH - 1);
      ctx.fillStyle = 'rgba(244,246,251,0.75)';
      ctx.font = '9px "JetBrains Mono", monospace';
      ctx.textAlign = 'left';
      ctx.fillText(`ch ${i}`, tx + 3, ty + 11);
      ctx.restore();
    }
  }

  private renderDecoded(ctx: CanvasRenderingContext2D, canvas: HTMLCanvasElement): void {
    const dpr = Math.min(devicePixelRatio || 1, 2);
    const W = canvas.width / dpr;
    const H = canvas.height / dpr;
    ctx.clearRect(0, 0, W, H);
    const size = Math.min(W, H) - 16;
    const x = (W - size) / 2;
    const y = (H - size) / 2;

    // "Decode": combine the first few latent channels into a smooth RGB field.
    const step = Math.max(2, Math.floor(size / 90));
    for (let py = 0; py < size; py += step) {
      for (let px = 0; px < size; px += step) {
        const u = px / size;
        const v = py / size;
        const c0 = this.latentValue(u, v, 0);
        const c1 = this.latentValue(u, v, 1);
        const c2 = this.latentValue(u, v, Math.min(2, this.latentCh() - 1));
        const t = (c0 * 0.5 + c1 * 0.3 + c2 * 0.2);
        const [r, g, b] = this.heat(t);
        ctx.fillStyle = `rgb(${r | 0},${g | 0},${b | 0})`;
        ctx.fillRect(x + px, y + py, step, step);
      }
    }
    ctx.save();
    ctx.strokeStyle = 'rgba(65,214,255,0.45)';
    ctx.lineWidth = 1;
    ctx.strokeRect(x + 0.5, y + 0.5, size - 1, size - 1);
    ctx.restore();
  }
}
