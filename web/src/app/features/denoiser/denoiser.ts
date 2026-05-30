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

type Arch = 'unet' | 'dit';

@Component({
  selector: 'app-denoiser',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [Chapter, CodeRef, MathTex],
  template: `
    <app-chapter slug="denoiser">
      <p class="lede">
        The forward process is pure arithmetic — no learning. Everything the model knows lives in a single network
        that runs <em>inside the reverse step</em>: feed it a noisy latent <app-math expr="x_t" />, the current
        timestep <app-math expr="t" />, and (optionally) the text prompt, and it predicts the noise
        <app-math expr="\\epsilon" /> that was mixed in. Subtract a sliver of that prediction, repeat a few dozen
        times, and static turns into a picture. This chapter is about that network — how it was a <strong>U-Net</strong>,
        why it is becoming a <strong>diffusion Transformer</strong>, and the one idea that connects them.
      </p>

      <div class="contract panel">
        <span class="tag tag--accent">the contract</span>
        <app-math display
          expr="\\epsilon_\\theta(x_t,\\,t,\\,c) \\;\\approx\\; \\epsilon \\quad\\text{where}\\quad x_t=\\sqrt{\\bar\\alpha_t}\\,x_0+\\sqrt{1-\\bar\\alpha_t}\\,\\epsilon" />
        <p class="contract__note">
          U-Net and DiT are interchangeable: both obey the same <code>forward(x, t, context, mask) → eps</code>
          signature in this repo. They differ only in <em>how</em> they compute it.
        </p>
      </div>

      <!-- ============================ FIGURE 1 ============================ -->
      <h2>The two shapes of a denoiser</h2>
      <p>
        A <strong>U-Net</strong> is convolutional. It pushes the latent <em>down</em> through a few resolutions to see
        global structure, then back <em>up</em>, re-injecting the lost detail through <strong>skip connections</strong>.
        A <strong>DiT</strong> throws the U-shape away: it chops the latent into patches, treats them as a flat
        sequence of tokens, and runs a plain Transformer stack — no downsampling, no skips, no convolutions in the
        trunk. Toggle below and watch a pulse of data travel through each.
      </p>

      <figure class="fig">
        <div class="fig__controls">
          <div class="seg" role="tablist" aria-label="architecture">
            <button
              type="button"
              class="seg__btn"
              [class.seg__btn--on]="arch() === 'unet'"
              (click)="setArch('unet')"
            >U-Net</button>
            <button
              type="button"
              class="seg__btn"
              [class.seg__btn--on]="arch() === 'dit'"
              (click)="setArch('dit')"
            >DiT</button>
          </div>
          <label class="chk">
            <input type="checkbox" [checked]="conditioned()" (change)="toggleCond($event)" />
            text conditioning (cross-attention)
          </label>
          <button type="button" class="btn" (click)="pulse()">▶ send a pulse</button>
        </div>
        <div class="fig__stage">
          <canvas #cv class="fig__canvas"></canvas>
        </div>
        <figcaption class="fig__cap">
          {{ archCaption() }}
        </figcaption>
      </figure>

      <p>
        Notice what stays the same across the toggle: a <span class="ink-time">time embedding</span> is injected at
        <em>every</em> processing block, and (when conditioning is on) a <span class="ink-cross">cross-attention</span>
        path lets the prompt steer the latent. The U-Net injects time inside conv <code>ResBlock</code>s and inserts
        <code>SpatialTransformer</code> blocks only at the deeper, cheaper resolutions; the DiT folds time into
        <strong>adaLN-Zero</strong> modulation and gives every block its own cross-attention.
      </p>

      <h3>U-Net: the residual block that hears the clock</h3>
      <p>
        Inside a U-Net, the workhorse is the <code>ResBlock</code>: two <code>(GroupNorm → SiLU → Conv)</code> stacks
        with the projected time vector <strong>added between them</strong>. That addition is FiLM-lite — it shifts the
        activations based on how noisy the input is, which the network needs because denoising at
        <app-math expr="t=10" /> (almost clean) is a completely different job from <app-math expr="t=900" />
        (almost pure static).
      </p>

      <app-code-ref
        file="pytorch/diffusion/unet.py"
        lang="python"
        [code]="snipResBlock"
        [lines]="[80, 85]"
        caption="ResBlock.forward — the time embedding is broadcast over H×W and added between the two conv stacks (FiLM)."
      />

      <p>
        The full <code>UNet.forward</code> is the encoder → bottleneck → decoder dance, with each decoder stage
        concatenating the matching skip tensor it stashed on the way down. That <code>torch.cat</code> is the arc you
        see drawn in the figure above.
      </p>

      <app-code-ref
        file="pytorch/diffusion/unet.py"
        lang="python"
        [code]="snipUnetForward"
        [lines]="[269, 288]"
        caption="UNet.forward — push down (saving skips), pass the bottleneck, then pop and concatenate each skip on the way up."
      />

      <!-- ============================ FIGURE 2 ============================ -->
      <h2>Patchify: turning a grid into a sequence</h2>
      <p>
        A Transformer eats a 1-D sequence of tokens, but a latent is a 2-D grid
        <app-math expr="[C,\\,H,\\,W]" />. <strong>Patchify</strong> bridges them: cut the grid into a
        <app-math expr="(H/p)\\times(W/p)" /> array of <app-math expr="p\\times p" /> squares, flatten each square
        into one vector, and line them up. Each patch becomes <em>exactly one token</em>. Slide the patch size and
        watch the token count — and the cost of attention — change.
      </p>

      <figure class="fig">
        <div class="fig__controls">
          <div class="patch-ctl">
            <span class="patch-ctl__label">patch size p</span>
            <div class="seg seg--sm">
              @for (opt of patchOpts; track opt) {
                <button
                  type="button"
                  class="seg__btn"
                  [class.seg__btn--on]="patch() === opt"
                  (click)="setPatch(opt)"
                >{{ opt }}</button>
              }
            </div>
          </div>
          <div class="patch-stats">
            <span class="stat"><b>{{ grid() }}×{{ grid() }}</b> grid</span>
            <span class="stat"><b>{{ tokenCount() }}</b> tokens</span>
            <span class="stat"><b>{{ tokenDim() }}</b> dims / token</span>
            <span class="stat stat--cost"><b>{{ attnCost() }}</b> attention pairs</span>
          </div>
        </div>
        <div class="fig__stage fig__stage--patch">
          <canvas #pcv class="fig__canvas"></canvas>
        </div>
        <figcaption class="fig__cap">
          A 16×16 latent split into {{ tokenCount() }} patches of {{ patch() }}×{{ patch() }}, then unrolled into a
          token sequence (row-major). Halving p quadruples the tokens — and roughly quadruples-squared the attention.
        </figcaption>
      </figure>

      <p>
        The arithmetic is unforgiving: attention is <app-math expr="O(N^2)" /> in the token count
        <app-math expr="N=(H/p)^2" />. Drop from <app-math expr="p=4" /> to <app-math expr="p=2" /> and
        <app-math expr="N" /> goes 4× while the attention matrix goes 16×. Smaller patches mean finer detail and a
        far bigger bill — which is exactly why DiTs run in a <em>compressed VAE latent</em>, not in pixels. Here is the
        real conversion, channels-fastest:
      </p>

      <app-code-ref
        file="pytorch/diffusion/dit.py"
        lang="python"
        [code]="snipPatchify"
        [lines]="[64, 72]"
        caption="patchify — reshape + permute folds each p×p patch (and all its channels) into one token vector of length C·p·p."
      />

      <!-- ============================ TIME EMBEDDING ============================ -->
      <h2>How a single integer becomes a steering signal</h2>
      <p>
        The timestep <app-math expr="t" /> is just an integer, but the network needs it as a smooth vector so that
        nearby steps look nearby. The trick is the same sinusoidal featurization Transformers use for position — only
        here the "position" is the diffusion step. A fixed bank of geometrically-spaced frequencies turns
        <app-math expr="t" /> into <app-math expr="[\\sin(\\omega_k t),\\,\\cos(\\omega_k t)]" />; a small learnable MLP
        then projects it into whatever space the blocks expect.
      </p>

      <app-code-ref
        file="pytorch/diffusion/cross_attention.py"
        lang="python"
        [code]="snipTimestep"
        [lines]="[56, 65]"
        caption="SinusoidalPosEmb.forward — t is embedded at many frequencies; nearby timesteps get nearby vectors. (TimestepEmbedding wraps this in a learnable MLP.)"
      />

      <p>
        Once you have that vector, the two architectures inject it differently — and this is the single biggest
        difference between them:
      </p>
      <ul class="compare">
        <li>
          <span class="tag">U-Net</span> add the projected time vector inside each <code>ResBlock</code> (FiLM). Text
          enters separately, via <code>SpatialTransformer</code> cross-attention blocks bolted onto a few resolutions.
        </li>
        <li>
          <span class="tag tag--accent">DiT</span> turn time (plus a pooled text vector) into six modulation signals
          per block — <em>shift, scale, gate</em> for the attention sub-layer and again for the MLP — and apply them as
          <strong>adaLN-Zero</strong>. One clean mechanism replaces both the FiLM-in-convs and the bolt-on attention.
        </li>
      </ul>

      <!-- ============================ DIT BLOCK ============================ -->
      <h2>adaLN-Zero: why the DiT trains so calmly</h2>
      <p>
        In a DiT block the LayerNorms have <em>no</em> learnable affine — the conditioning provides the scale and shift
        instead. Each sub-layer computes
        <app-math expr="x \\,{+}\\, g\\cdot \\mathrm{sublayer}\\big((1{+}s)\\,\\mathrm{LN}(x)+h\\big)" />, where the gate
        <app-math expr="g" />, scale <app-math expr="s" /> and shift <app-math expr="h" /> all come from
        <app-math expr="t" />. The <strong>"-Zero"</strong> is the punchline: the linear that produces those signals is
        zero-initialized, so every block starts as the identity (<app-math expr="g=0" />) and the whole network's first
        prediction is <app-math expr="\\epsilon = 0" /> — a calm, stable place to begin training.
      </p>

      <app-code-ref
        file="pytorch/diffusion/dit.py"
        lang="python"
        [code]="snipDitBlock"
        [lines]="[144, 161]"
        caption="DiTBlock.forward — six (or seven) modulation signals from the timestep gate the attention, optional cross-attention, and MLP sub-layers."
      />

      <!-- ============================ CROSS ATTENTION ============================ -->
      <h2>Cross-attention: where the prompt actually touches the pixels</h2>
      <p>
        In both backbones, text conditioning is literally one module: <code>CrossAttention</code>. Each image token
        forms a <strong>query</strong>; the prompt's word tokens supply the <strong>keys and values</strong>. Every
        image position asks "which words are relevant to me?" and pulls in their values — a continuous steering signal
        applied at <em>every step</em>, not something generated alongside the image.
      </p>

      <app-math display
        expr="\\mathrm{Attn}(Q,K,V)=\\mathrm{softmax}\\!\\left(\\frac{Q K^{\\top}}{\\sqrt{d}}\\right)V" />

      <p>
        Pass <code>context = None</code> and the very same module degrades to <strong>self-attention</strong>
        (<app-math expr="K,V" /> come from the image itself), which is how distant regions of a picture stay
        consistent. That dual use is why one class powers both jobs:
      </p>

      <app-code-ref
        file="pytorch/diffusion/cross_attention.py"
        lang="python"
        [code]="snipCrossAttn"
        [lines]="[138, 156]"
        caption="CrossAttention.forward — with context it is text→image cross-attention; without it the same code is self-attention (ctx := x)."
      />

      <h2>So why is everyone switching to DiT?</h2>
      <p>
        The U-Net's convolutional U-shape is a strong inductive bias for images and is wonderfully sample-efficient at
        small scale — it is why SD 1.x, 2.x and SDXL are U-Nets. But a Transformer of patches scales more
        <em>predictably</em> with compute and data: add blocks, widen the hidden dim, feed more tokens, and quality
        climbs along clean curves. That is the recipe behind Stable Diffusion 3's MM-DiT, PixArt, and Sora-style
        video models, where the latent becomes a <em>spacetime</em> grid of patches and the same machinery denoises a
        whole clip at once. Same contract — predict <app-math expr="\\epsilon" /> — radically more scalable shape.
      </p>

      <p class="footnote">
        One caveat worth keeping straight: this whole family <em>denoises a latent</em>. It is not predicting the next
        frame or the next pixel — that "next-token" mental model belongs to the autoregressive family, a different
        chapter entirely.
      </p>
    </app-chapter>
  `,
  styles: [`
    :host { display: block; }
    .lede { font-size: var(--step-1); color: var(--ink-1); }
    .lede em { color: var(--ink-0); font-style: normal; border-bottom: 1px dashed var(--line-strong); }

    .contract { padding: 1.2rem 1.4rem; margin: 1.8rem 0 0.4rem; }
    .contract .tag { margin-bottom: 0.4rem; display: inline-block; }
    .contract__note { color: var(--ink-2); font-size: 0.86rem; margin: 0.2rem 0 0; }

    .ink-time { color: var(--warn); font-weight: 600; }
    .ink-cross { color: var(--plasma-c); font-weight: 600; }

    /* ---- figures ---- */
    .fig {
      margin: 1.8rem 0 0.6rem;
      border: 1px solid var(--line);
      border-radius: var(--radius);
      background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0)), var(--bg-2);
      overflow: hidden;
    }
    .fig__controls {
      display: flex; flex-wrap: wrap; align-items: center; gap: 0.8rem 1.1rem;
      padding: 0.85rem 1rem; border-bottom: 1px solid var(--line);
      background: rgba(255,255,255,0.015);
    }
    .fig__stage { position: relative; width: 100%; height: 340px; background: #07090f; }
    .fig__stage--patch { height: 360px; }
    .fig__canvas { display: block; width: 100%; height: 100%; }
    .fig__cap {
      padding: 0.65rem 1rem; border-top: 1px solid var(--line);
      color: var(--ink-2); font-size: 0.84rem; line-height: 1.5;
    }

    /* segmented toggle */
    .seg { display: inline-flex; border: 1px solid var(--line-strong); border-radius: 999px; padding: 3px; background: var(--bg-3); }
    .seg--sm { padding: 2px; }
    .seg__btn {
      font-family: var(--font-mono); font-size: 0.78rem; cursor: pointer;
      border: 0; background: transparent; color: var(--ink-2);
      padding: 0.3em 0.95em; border-radius: 999px; transition: color .15s var(--ease), background .2s var(--ease);
    }
    .seg--sm .seg__btn { padding: 0.28em 0.7em; min-width: 2.1em; }
    .seg__btn:hover { color: var(--ink-0); }
    .seg__btn--on { color: #fff; background: var(--grad-plasma); box-shadow: var(--shadow-1); }

    .chk { display: inline-flex; align-items: center; gap: 0.5em; font-family: var(--font-mono); font-size: 0.76rem; color: var(--ink-2); cursor: pointer; }
    .chk input { accent-color: var(--plasma-a); width: 15px; height: 15px; cursor: pointer; }

    .btn { cursor: pointer; }

    .patch-ctl { display: inline-flex; align-items: center; gap: 0.6rem; }
    .patch-ctl__label { font-family: var(--font-mono); font-size: 0.74rem; color: var(--ink-2); text-transform: uppercase; letter-spacing: 0.1em; }
    .patch-stats { display: inline-flex; flex-wrap: wrap; gap: 0.45rem 0.6rem; margin-left: auto; }
    .stat {
      font-family: var(--font-mono); font-size: 0.72rem; color: var(--ink-2);
      border: 1px solid var(--line); border-radius: 999px; padding: 0.22em 0.7em; background: var(--bg-3);
    }
    .stat b { color: var(--ink-0); }
    .stat--cost { border-color: rgba(255,92,138,0.4); }
    .stat--cost b { color: var(--plasma-c); }

    /* prose extras */
    .compare { list-style: none; padding: 0; margin: 1.2rem 0; display: grid; gap: 0.7rem; }
    .compare li {
      border: 1px solid var(--line); border-radius: var(--radius-sm);
      padding: 0.8rem 1rem; background: var(--bg-2); color: var(--ink-1); font-size: 0.92rem;
    }
    .compare .tag { margin-right: 0.5rem; }
    .footnote { color: var(--ink-2); font-size: 0.9rem; border-left: 2px solid var(--line-strong); padding-left: 0.9rem; }

    @media (max-width: 560px) {
      .fig__controls { gap: 0.6rem; }
      .patch-stats { margin-left: 0; }
    }
  `],
})
export class Denoiser {
  private readonly fb = inject(FirebaseService);
  private readonly destroyRef = inject(DestroyRef);
  private readonly cv = viewChild.required<ElementRef<HTMLCanvasElement>>('cv');
  private readonly pcv = viewChild.required<ElementRef<HTMLCanvasElement>>('pcv');

  // ---- Figure 1 state ----
  readonly arch = signal<Arch>('unet');
  readonly conditioned = signal<boolean>(true);
  private pulseStart = 0; // performance.now() of last manual pulse; 0 => free-running

  readonly archCaption = computed(() =>
    this.arch() === 'unet'
      ? 'U-Net: the latent flows down two resolutions (saving skip arcs), through a self-attention bottleneck, then back up — concatenating each skip. Time is added in every ResBlock; cross-attention sits at the deeper levels.'
      : 'DiT: the latent is patchified into tokens, run through a stack of identical Transformer blocks (each modulated by the timestep via adaLN-Zero), then unpatchified back to a latent. No skips, no downsampling.',
  );

  // ---- Figure 2 state ----
  readonly latentSide = 16; // a 16×16 toy latent
  readonly channels = 4; // SD-style 4-channel VAE latent
  readonly patchOpts = [1, 2, 4] as const;
  readonly patch = signal<number>(2);

  readonly grid = computed(() => this.latentSide / this.patch());
  readonly tokenCount = computed(() => this.grid() * this.grid());
  readonly tokenDim = computed(() => this.channels * this.patch() * this.patch());
  readonly attnCost = computed(() => {
    const n = this.tokenCount();
    return (n * n).toLocaleString();
  });

  // ====================== code snippets (verbatim) ======================
  readonly snipResBlock = `def forward(self, x: Tensor, t_emb: Tensor) -> Tensor:
    h = self.in_layers(x)
    # add time embedding: [B, out_ch] -> [B, out_ch, 1, 1]
    h = h + self.time_proj(t_emb)[:, :, None, None]
    h = self.out_layers(h)
    return h + self.skip(x)`;

  readonly snipUnetForward = `# --- encoder ---
h = self.input_conv(x)
skips = [h]
for block in self.down_blocks:
    h = self._run_block(block, h, t_emb, context, mask)
    skips.append(h)

# --- bottleneck ---
h = self.mid_block1(h, t_emb)
h = self.mid_attn(h)
if self.mid_cross is not None:
    h = self.mid_cross(h, context=context, mask=mask)
h = self.mid_block2(h, t_emb)

# --- decoder (concatenate the matching skip before each ResBlock) ---
for block in self.up_blocks:
    first = block[0]
    if isinstance(first, ResBlock):
        h = torch.cat([h, skips.pop()], dim=1)  # skip connection
    h = self._run_block(block, h, t_emb, context, mask)`;

  readonly snipPatchify = `b, c, h, w = x.shape
if h % p != 0 or w % p != 0:
    raise ValueError(f"image size ({h},{w}) not divisible by patch_size {p}")
gh, gw = h // p, w // p
# [B,C,gh,p,gw,p] -> [B,gh,gw,C,p,p] -> [B, gh*gw, C*p*p]
x = x.reshape(b, c, gh, p, gw, p)
x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
tokens = x.reshape(b, gh * gw, c * p * p)
return tokens, (gh, gw)`;

  readonly snipTimestep = `device = t.device
half = self.dim // 2
# frequencies geometrically spaced between 1 and 1/10000 (DDPM/Transformer std).
# emb_freqs[k] = exp(-k * log(10000) / (half-1))
emb = math.log(10000.0) / max(half - 1, 1)
emb = torch.exp(torch.arange(half, device=device, dtype=torch.float32) * -emb)
# outer product: [B,1] * [1,half] -> [B, half]
emb = t.float()[:, None] * emb[None, :]
# interleave sin/cos -> [B, dim]
return torch.cat([emb.sin(), emb.cos()], dim=-1)`;

  readonly snipDitBlock = `signals = self.ada(cond)
if self.has_cross:
    shift1, scale1, gate1, shift2, scale2, gate2, gate_c = signals.chunk(7, dim=-1)
else:
    shift1, scale1, gate1, shift2, scale2, gate2 = signals.chunk(6, dim=-1)

# --- self-attention sub-layer (gated residual, adaLN-modulated input) ---
x = x + gate1.unsqueeze(1) * self.attn(_modulate(self.norm1(x), shift1, scale1))

# --- optional cross-attention to text tokens ---
if self.has_cross and context is not None:
    x = x + gate_c.unsqueeze(1) * self.cross_attn(
        self.norm_cross(x), context=context, mask=mask
    )

# --- MLP sub-layer ---
x = x + gate2.unsqueeze(1) * self.mlp(_modulate(self.norm2(x), shift2, scale2))
return x`;

  readonly snipCrossAttn = `# If no context, this is self-attention: tokens attend among themselves.
ctx = context if context is not None else x

q = self._split_heads(self.to_q(x))      # [B, H, N, d]
k = self._split_heads(self.to_k(ctx))    # [B, H, L, d]
v = self._split_heads(self.to_v(ctx))    # [B, H, L, d]

attn_mask = None
if mask is not None and context is not None:
    keep = mask[:, None, None, :].to(q.dtype)          # 1.0 real, 0.0 pad
    attn_mask = (1.0 - keep) * torch.finfo(q.dtype).min

# softmax(QK^T/sqrt(d) + attn_mask) @ V
out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)  # [B,H,N,d]`;

  constructor() {
    afterNextRender(() => {
      this.drawArch();
      this.drawPatch();
    });
    // redraw the patch figure whenever the patch size changes
    effect(() => {
      this.patch();
      this.redrawPatch?.();
    });
  }

  // ---------------------------- handlers ----------------------------
  setArch(a: Arch): void {
    this.arch.set(a);
    this.pulse();
    this.fb.event('interact', { section: 'denoiser', control: 'arch', value: a });
  }
  toggleCond(e: Event): void {
    this.conditioned.set((e.target as HTMLInputElement).checked);
    this.fb.event('interact', { section: 'denoiser', control: 'conditioning', value: this.conditioned() });
  }
  pulse(): void {
    this.pulseStart = (typeof performance !== 'undefined' ? performance.now() : Date.now());
    this.fb.event('interact', { section: 'denoiser', control: 'pulse', value: this.arch() });
  }
  setPatch(p: number): void {
    this.patch.set(p);
    this.fb.event('interact', { section: 'denoiser', control: 'patch', value: p });
  }

  private redrawPatch: (() => void) | null = null;

  // ======================================================================
  // FIGURE 1 — animated architecture diagram with a traveling pulse
  // ======================================================================
  private drawArch(): void {
    const canvas = this.cv().nativeElement;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    let W = 0, H = 0;
    const resize = () => {
      const dpr = Math.min(devicePixelRatio || 1, 2);
      const r = canvas.getBoundingClientRect();
      W = r.width; H = r.height;
      canvas.width = Math.floor(r.width * dpr);
      canvas.height = Math.floor(r.height * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    resize();
    addEventListener('resize', resize);

    const PLASMA_A = '#7c5cff', PLASMA_B = '#41d6ff', WARN = '#ffcf5c', CROSS = '#ff5c8a';
    const INK2 = '#8a92a6', LINE = 'rgba(255,255,255,0.10)';

    // Build the layout for the current architecture. Returns nodes along a path
    // [0..1] so we can place a pulse by arc-length, plus skip arcs for the U-Net.
    interface Node { x: number; y: number; w: number; h: number; label: string; tone: 'a' | 'b' | 'mid'; }
    interface Skip { from: number; to: number; }

    const buildUnet = () => {
      const nodes: Node[] = [];
      const cx = W / 2;
      const colGap = Math.min(120, (W - 120) / 4);
      const topY = 64, botY = H - 64;
      const bw = 86, bh = 30;
      // encoder columns (left, descending), bottleneck (center), decoder (right, ascending)
      const xs = [cx - colGap * 2, cx - colGap, cx, cx + colGap, cx + colGap * 2];
      const downYs = [topY, (topY + botY) / 2 - 14];
      // encoder
      nodes.push({ x: xs[0], y: downYs[0], w: bw, h: bh, label: 'enc ↓', tone: 'a' });
      nodes.push({ x: xs[1], y: downYs[1], w: bw, h: bh, label: 'enc ↓', tone: 'a' });
      // bottleneck
      nodes.push({ x: xs[2], y: botY, w: bw + 12, h: bh, label: 'bottleneck', tone: 'mid' });
      // decoder
      nodes.push({ x: xs[3], y: downYs[1], w: bw, h: bh, label: 'dec ↑', tone: 'b' });
      nodes.push({ x: xs[4], y: downYs[0], w: bw, h: bh, label: 'dec ↑', tone: 'b' });
      const skips: Skip[] = [{ from: 0, to: 4 }, { from: 1, to: 3 }];
      return { nodes, skips };
    };

    const buildDit = () => {
      const nodes: Node[] = [];
      const n = 5; // patch-embed + 3 DiT blocks + unpatchify (visually)
      const labels = ['patchify', 'DiT ×', 'DiT ×', 'DiT ×', 'unpatch'];
      const tones: Array<'a' | 'b' | 'mid'> = ['a', 'mid', 'mid', 'mid', 'b'];
      const bw = 74, bh = 34;
      const usable = W - 100;
      const step = usable / (n - 1);
      const y = H / 2;
      for (let i = 0; i < n; i++) {
        nodes.push({ x: 50 + step * i, y, w: bw, h: bh, label: labels[i], tone: tones[i] });
      }
      return { nodes, skips: [] as Skip[] };
    };

    const roundRect = (x: number, y: number, w: number, h: number, r: number) => {
      ctx.beginPath();
      ctx.moveTo(x - w / 2 + r, y - h / 2);
      ctx.arcTo(x + w / 2, y - h / 2, x + w / 2, y + h / 2, r);
      ctx.arcTo(x + w / 2, y + h / 2, x - w / 2, y + h / 2, r);
      ctx.arcTo(x - w / 2, y + h / 2, x - w / 2, y - h / 2, r);
      ctx.arcTo(x - w / 2, y - h / 2, x + w / 2, y - h / 2, r);
      ctx.closePath();
    };

    let raf = 0;
    const loop = () => {
      ctx.clearRect(0, 0, W, H);
      const isUnet = this.arch() === 'unet';
      const { nodes, skips } = isUnet ? buildUnet() : buildDit();
      const cond = this.conditioned();
      const now = (typeof performance !== 'undefined' ? performance.now() : Date.now());

      // ---- main path (poly-line through node centers) ----
      ctx.strokeStyle = LINE;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(nodes[0].x, nodes[0].y);
      for (let i = 1; i < nodes.length; i++) ctx.lineTo(nodes[i].x, nodes[i].y);
      ctx.stroke();

      // arrow caps between nodes
      ctx.fillStyle = INK2;
      for (let i = 0; i < nodes.length - 1; i++) {
        const a = nodes[i], b = nodes[i + 1];
        const mx = (a.x + b.x) / 2, my = (a.y + b.y) / 2;
        const ang = Math.atan2(b.y - a.y, b.x - a.x);
        ctx.save();
        ctx.translate(mx, my);
        ctx.rotate(ang);
        ctx.beginPath();
        ctx.moveTo(5, 0); ctx.lineTo(-3, 3.5); ctx.lineTo(-3, -3.5);
        ctx.closePath(); ctx.fill();
        ctx.restore();
      }

      // ---- skip arcs (U-Net only) ----
      for (const s of skips) {
        const a = nodes[s.from], b = nodes[s.to];
        const midX = (a.x + b.x) / 2;
        const lift = Math.min(a.y, b.y) - 42;
        ctx.strokeStyle = 'rgba(124,92,255,0.5)';
        ctx.lineWidth = 1.6;
        ctx.setLineDash([5, 4]);
        ctx.beginPath();
        ctx.moveTo(a.x, a.y - a.h / 2);
        ctx.quadraticCurveTo(midX, lift, b.x, b.y - b.h / 2);
        ctx.stroke();
        ctx.setLineDash([]);
        // little "+cat" label at the apex
        ctx.fillStyle = 'rgba(124,92,255,0.85)';
        ctx.font = '600 10px ui-monospace, monospace';
        ctx.textAlign = 'center';
        ctx.fillText('skip ⌢ cat', midX, lift - 4);
      }

      // ---- nodes ----
      for (const nd of nodes) {
        const grad = ctx.createLinearGradient(nd.x - nd.w / 2, 0, nd.x + nd.w / 2, 0);
        if (nd.tone === 'a') { grad.addColorStop(0, 'rgba(124,92,255,0.25)'); grad.addColorStop(1, 'rgba(124,92,255,0.10)'); }
        else if (nd.tone === 'b') { grad.addColorStop(0, 'rgba(65,214,255,0.10)'); grad.addColorStop(1, 'rgba(65,214,255,0.25)'); }
        else { grad.addColorStop(0, 'rgba(255,92,138,0.18)'); grad.addColorStop(1, 'rgba(124,92,255,0.18)'); }
        roundRect(nd.x, nd.y, nd.w, nd.h, 8);
        ctx.fillStyle = grad;
        ctx.fill();
        ctx.strokeStyle = 'rgba(255,255,255,0.18)';
        ctx.lineWidth = 1;
        ctx.stroke();
        ctx.fillStyle = '#eef1f8';
        ctx.font = '600 11px ui-monospace, monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(nd.label, nd.x, nd.y);

        // time-embedding tick on EVERY node (warn-colored), below the box
        ctx.fillStyle = WARN;
        ctx.font = '600 9px ui-monospace, monospace';
        ctx.fillText('+t', nd.x, nd.y + nd.h / 2 + 9);

        // cross-attention marker on conditioned blocks (deeper U-Net levels / DiT blocks)
        const showCross = cond && (isUnet ? nd.tone !== 'a' : nd.tone === 'mid');
        if (showCross) {
          ctx.fillStyle = CROSS;
          ctx.fillText('×attn', nd.x, nd.y - nd.h / 2 - 8);
        }
      }

      // ---- traveling pulse along the main path ----
      // total path length:
      const segLens: number[] = [];
      let total = 0;
      for (let i = 0; i < nodes.length - 1; i++) {
        const d = Math.hypot(nodes[i + 1].x - nodes[i].x, nodes[i + 1].y - nodes[i].y);
        segLens.push(d); total += d;
      }
      const period = 2600; // ms for a full traversal
      const base = this.pulseStart || now;
      const phase = ((now - base) % period) / period; // 0..1
      let target = phase * total;
      let px = nodes[0].x, py = nodes[0].y;
      for (let i = 0; i < segLens.length; i++) {
        if (target <= segLens[i]) {
          const f = segLens[i] === 0 ? 0 : target / segLens[i];
          px = nodes[i].x + (nodes[i + 1].x - nodes[i].x) * f;
          py = nodes[i].y + (nodes[i + 1].y - nodes[i].y) * f;
          break;
        }
        target -= segLens[i];
      }
      // glow
      const glow = ctx.createRadialGradient(px, py, 0, px, py, 16);
      glow.addColorStop(0, 'rgba(255,255,255,0.95)');
      glow.addColorStop(0.4, PLASMA_B);
      glow.addColorStop(1, 'rgba(65,214,255,0)');
      ctx.beginPath();
      ctx.arc(px, py, 16, 0, Math.PI * 2);
      ctx.fillStyle = glow;
      ctx.fill();
      ctx.beginPath();
      ctx.arc(px, py, 4, 0, Math.PI * 2);
      ctx.fillStyle = '#ffffff';
      ctx.fill();

      // ---- legend ----
      ctx.textAlign = 'left';
      ctx.textBaseline = 'alphabetic';
      ctx.font = '600 10px ui-monospace, monospace';
      ctx.fillStyle = WARN; ctx.fillText('+t  time embedding (every block)', 14, H - 26);
      if (cond) { ctx.fillStyle = CROSS; ctx.fillText('×attn  cross-attention (prompt)', 14, H - 12); }
      ctx.fillStyle = PLASMA_A;
      ctx.textAlign = 'right';
      ctx.fillText(isUnet ? 'noisy x_t  →  predicted ε' : 'latent  →  tokens  →  latent', W - 14, H - 12);

      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);

    this.destroyRef.onDestroy(() => {
      cancelAnimationFrame(raf);
      removeEventListener('resize', resize);
    });
  }

  // ======================================================================
  // FIGURE 2 — interactive patchify: grid → patches → token sequence
  // ======================================================================
  private drawPatch(): void {
    const canvas = this.pcv().nativeElement;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    let W = 0, H = 0;
    const resize = () => {
      const dpr = Math.min(devicePixelRatio || 1, 2);
      const r = canvas.getBoundingClientRect();
      W = r.width; H = r.height;
      canvas.width = Math.floor(r.width * dpr);
      canvas.height = Math.floor(r.height * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      render();
    };

    // stable per-cell hue so patches keep a consistent texture as p changes
    const side = this.latentSide;
    const cellHue: number[] = [];
    for (let i = 0; i < side * side; i++) {
      // pseudo-random but deterministic
      const v = Math.sin(i * 12.9898) * 43758.5453;
      cellHue.push(v - Math.floor(v));
    }

    const patchColor = (gx: number, gy: number, gridN: number) => {
      const t = (gx + gy * gridN) / Math.max(gridN * gridN - 1, 1);
      // interpolate along the plasma ramp violet -> cyan -> magenta
      const stops = [[124, 92, 255], [65, 214, 255], [255, 92, 138]];
      const seg = t * 2;
      const i = Math.min(1, Math.floor(seg));
      const f = seg - i;
      const a = stops[i], b = stops[i + 1];
      const r = Math.round(a[0] + (b[0] - a[0]) * f);
      const g = Math.round(a[1] + (b[1] - a[1]) * f);
      const bl = Math.round(a[2] + (b[2] - a[2]) * f);
      return `rgb(${r},${g},${bl})`;
    };

    const render = () => {
      if (W === 0) return;
      ctx.clearRect(0, 0, W, H);
      const p = this.patch();
      const gridN = side / p;

      // ---- left: the latent grid, partitioned into p×p patches ----
      const pad = 18;
      const gridSize = Math.min(H - pad * 2, W * 0.42);
      const gx0 = pad;
      const gy0 = (H - gridSize) / 2;
      const cell = gridSize / side;

      // fine latent cells (faint texture)
      for (let y = 0; y < side; y++) {
        for (let x = 0; x < side; x++) {
          const h = cellHue[y * side + x];
          ctx.fillStyle = `hsla(${220 + h * 60}, 30%, ${22 + h * 14}%, 1)`;
          ctx.fillRect(gx0 + x * cell, gy0 + y * cell, cell, cell);
        }
      }
      // patch boundaries + tint
      for (let py = 0; py < gridN; py++) {
        for (let px = 0; px < gridN; px++) {
          const x = gx0 + px * p * cell;
          const y = gy0 + py * p * cell;
          const s = p * cell;
          ctx.fillStyle = patchColor(px, py, gridN);
          ctx.globalAlpha = 0.22;
          ctx.fillRect(x, y, s, s);
          ctx.globalAlpha = 1;
          ctx.strokeStyle = 'rgba(255,255,255,0.5)';
          ctx.lineWidth = 1;
          ctx.strokeRect(x + 0.5, y + 0.5, s - 1, s - 1);
        }
      }
      ctx.strokeStyle = 'rgba(255,255,255,0.25)';
      ctx.lineWidth = 1.5;
      ctx.strokeRect(gx0, gy0, gridSize, gridSize);

      // label
      ctx.fillStyle = '#8a92a6';
      ctx.font = '600 11px ui-monospace, monospace';
      ctx.textAlign = 'center';
      ctx.fillText(`${side}×${side} latent`, gx0 + gridSize / 2, gy0 - 6);

      // ---- arrow ----
      const arrowX = gx0 + gridSize + 26;
      const midY = H / 2;
      ctx.strokeStyle = 'rgba(255,255,255,0.3)';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(arrowX - 8, midY);
      ctx.lineTo(arrowX + 18, midY);
      ctx.stroke();
      ctx.fillStyle = 'rgba(255,255,255,0.45)';
      ctx.beginPath();
      ctx.moveTo(arrowX + 18, midY); ctx.lineTo(arrowX + 10, midY - 4); ctx.lineTo(arrowX + 10, midY + 4);
      ctx.closePath(); ctx.fill();
      ctx.fillStyle = '#8a92a6';
      ctx.font = '600 9px ui-monospace, monospace';
      ctx.fillText('patchify', arrowX + 5, midY - 10);

      // ---- right: unrolled token sequence (row-major) ----
      const seqX0 = arrowX + 34;
      const seqW = W - seqX0 - pad;
      const n = gridN * gridN;
      // choose a token box size that fits a wrapped grid in the available area
      const cols = Math.max(1, Math.ceil(Math.sqrt(n * (seqW / Math.max(gridSize, 1)))));
      const realCols = Math.min(cols, n);
      const tw = Math.min(34, (seqW - (realCols - 1) * 4) / realCols);
      const th = Math.min(tw, 16);
      const gapx = 4, gapy = 5;
      const rows = Math.ceil(n / realCols);
      const totalH = rows * th + (rows - 1) * gapy;
      const seqY0 = Math.max(pad + 14, (H - totalH) / 2);

      ctx.fillStyle = '#8a92a6';
      ctx.textAlign = 'left';
      ctx.font = '600 11px ui-monospace, monospace';
      ctx.fillText(`${n} tokens · ${this.channels}·${p}·${p} = ${this.tokenDim()} dims each`, seqX0, seqY0 - 8);

      for (let i = 0; i < n; i++) {
        const r = Math.floor(i / realCols);
        const c = i % realCols;
        const gxp = i % gridN;
        const gyp = Math.floor(i / gridN);
        const x = seqX0 + c * (tw + gapx);
        const y = seqY0 + r * (th + gapy);
        ctx.fillStyle = patchColor(gxp, gyp, gridN);
        ctx.globalAlpha = 0.85;
        ctx.fillRect(x, y, tw, th);
        ctx.globalAlpha = 1;
        ctx.strokeStyle = 'rgba(255,255,255,0.22)';
        ctx.lineWidth = 0.75;
        ctx.strokeRect(x + 0.5, y + 0.5, tw - 1, th - 1);
        if (tw > 20 && i < 100) {
          ctx.fillStyle = 'rgba(0,0,0,0.55)';
          ctx.font = '600 8px ui-monospace, monospace';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText(String(i), x + tw / 2, y + th / 2 + 0.5);
          ctx.textBaseline = 'alphabetic';
        }
      }
    };

    this.redrawPatch = render;
    resize();
    addEventListener('resize', resize);
    this.destroyRef.onDestroy(() => {
      removeEventListener('resize', resize);
      this.redrawPatch = null;
    });
  }
}
