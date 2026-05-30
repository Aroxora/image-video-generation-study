import {
  Component, ChangeDetectionStrategy, ElementRef, afterNextRender, inject, DestroyRef,
  viewChild, signal, computed, effect,
} from '@angular/core';
import { Chapter } from '../../shared/chapter';
import { CodeRef } from '../../shared/code-ref';
import { Math as MathTex } from '../../shared/math';
import { FirebaseService } from '../../core/firebase.service';

/**
 * Chapter 07 — The other family: tokens, interleaved like an LLM.
 *
 * Two interactive figures, both faithful to pytorch/autoregressive/*:
 *   1. VQ tokenization — an image is encoded to a continuous feature grid, each
 *      cell snapped to its NEAREST codebook vector (a tiny live nearest-neighbour
 *      search you can drive), yielding a grid of integer code ids. Mirrors
 *      VectorQuantizer.forward / VQVAE.encode (vqvae.py).
 *   2. Interleaved generation (marquee) — one horizontal sequence
 *      [BOS] · text · [BOI] · image codes · [EOI] that fills in token-by-token in
 *      raster order. Step / play advances next-token prediction; a softmax bar shows
 *      the candidate distribution for the current cell. Mirrors
 *      TokenTransformer.generate + autoregressive_generate (transformer.py, sample.py).
 */
@Component({
  selector: 'app-autoregressive',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [Chapter, CodeRef, MathTex],
  template: `
<app-chapter slug="autoregressive">

  <p class="lede">
    Everything else on this site denoises. This family does something completely
    different: it turns a picture into a <strong>sentence of integers</strong> and
    then models images the way GPT models text — <em>predict the next token, given
    all the previous ones</em>. The mental shortcut you keep hearing — "it generates
    the next frame" — is wrong for diffusion, but it is <em>literally true</em> here.
    DALL·E&nbsp;1, VQGAN, Parti, MUSE, and the video token models (MagViT, VideoPoet)
    all live in this room.
  </p>

  <p>
    Two pieces make it work. First a <strong>VQ-VAE</strong> that translates pixels
    into a small grid of discrete codes (and back). Then a plain decoder-only
    <strong>Transformer</strong> that autoregresses over one stream in which text
    tokens and image codes share a single vocabulary. We'll build the intuition for
    each, live.
  </p>

  <h2>Step one: make the image discrete</h2>

  <p>
    A Transformer predicts over a finite vocabulary — it needs <em>tokens</em>, not
    real-valued pixels. The VQ-VAE supplies them. An encoder compresses the image to a
    continuous feature grid <app-math expr="z_e \\in \\mathbb{R}^{H'\\times W'\\times D}" />;
    then a <strong>vector quantizer</strong> replaces each cell with the nearest of
    <app-math expr="K" /> learned <em>codebook</em> vectors. The index of that nearest
    code — an integer in <app-math expr="[0, K)" /> — is the token.
  </p>

  <app-math display
    expr="q(z_e)_{ij} = \\arg\\min_{k\\in[0,K)} \\; \\lVert z_e^{(ij)} - e_k \\rVert_2^2,
          \\qquad e_k \\in \\mathbb{R}^{D}\\ \\text{(the codebook).}" />

  <p>
    The figure below is exactly that argmin. Each cell of the grid holds a continuous
    vector (drawn as a small 2-D point); we snap it to whichever codebook entry is
    closest, and the cell lights up with that code's id and color. Drag the
    <strong>encoder feature scatter</strong> or nudge the codebook — the assignments
    (and therefore the integer token grid on the right) update live.
  </p>

  <!-- FIGURE 1 — VQ tokenization -->
  <figure class="fig">
    <div class="fig__stage fig__stage--split">
      <div class="vq">
        <span class="vq__title">encoder features <span class="mono">z<sub>e</sub></span> → nearest code</span>
        <canvas #vq class="fig__canvas vq__canvas"
          (pointerdown)="vqDown($event)" (pointermove)="vqMove($event)"
          (pointerup)="vqUp()" (pointerleave)="vqUp()"></canvas>
      </div>
      <div class="vq">
        <span class="vq__title">integer code grid <span class="mono">indices [{{ GRID }}×{{ GRID }}]</span></span>
        <canvas #grid class="fig__canvas vq__canvas"></canvas>
      </div>
    </div>
    <div class="ctl">
      <div class="ctl__row ctl__row--btns">
        <span class="ctl__group">codebook size <span class="mono">K = {{ codeCount() }}</span></span>
        <button type="button" class="btn" (click)="setK(8)" [class.btn--primary]="codeCount() === 8">8</button>
        <button type="button" class="btn" (click)="setK(16)" [class.btn--primary]="codeCount() === 16">16</button>
        <button type="button" class="btn" (click)="setK(32)" [class.btn--primary]="codeCount() === 32">32</button>
        <button type="button" class="btn" (click)="reseedFeatures()">↻ re-encode</button>
      </div>
      <div class="ctl__readout">
        <span class="chip"><span class="chip__k">cells</span><span class="chip__v">{{ GRID * GRID }}</span></span>
        <span class="chip"><span class="chip__k">distinct codes used</span><span class="chip__v">{{ distinctCodes() }}</span></span>
        <span class="chip chip--accent"><span class="chip__k">grid → "sentence"</span><span class="chip__v">{{ GRID * GRID }} tokens</span></span>
      </div>
    </div>
    <figcaption class="fig__cap">
      Vector quantization, live. <strong>Left:</strong> each encoder cell (dot) is
      snapped to its nearest codebook vector (ring) in <app-math expr="\\ell_2" /> — drag
      a dot or a ring to watch assignments flip. <strong>Right:</strong> the resulting
      grid of integer code ids. Flatten it row-major and you have the image half of the
      sequence the Transformer predicts.
    </figcaption>
  </figure>

  <p>
    The quantizer is the only genuinely new idea here, and it has one wrinkle:
    <code>argmin</code> has no gradient, so the encoder couldn't learn through it.
    The fix is the <strong>straight-through estimator</strong> — forward-pass the snapped
    vector, but copy the gradient straight back as if nothing was snapped, via
    <code>z_q = z_e + (z_q - z_e).detach()</code>. The whole forward pass, including the
    no-outer-product L2 distance, is short:
  </p>

  <app-code-ref
    file="pytorch/autoregressive/vqvae.py"
    lang="python"
    [code]="snipQuantize"
    [lines]="[133, 160]"
    caption="VectorQuantizer.forward — the L2 nearest-code search, the two VQ-loss terms, and the straight-through trick that keeps the encoder trainable." />

  <p>
    After training the VQ-VAE is <strong>frozen</strong>. From then on we only touch the
    <code>indices</code>: <code>encode</code> runs the encoder and returns the
    <app-math expr="H'\\times W'" /> integer grid, and <code>decode_indices</code> looks the
    codes back up and runs the decoder to recover pixels. Those two methods are the entire
    bridge between the language model and the image:
  </p>

  <app-code-ref
    file="pytorch/autoregressive/vqvae.py"
    lang="python"
    [code]="snipEncode"
    [lines]="[310, 329]"
    caption="VQVAE.encode / decode_indices — image → integer code grid, and codes → pixels. The Transformer never sees a pixel." />

  <h2>Step two: one sequence, text and image interleaved</h2>

  <p>
    Now the payoff. The autoregressive models do <em>not</em> keep a separate "text
    branch" and "image branch" — they flatten everything into a single sequence over one
    <strong>shared vocabulary</strong> and run a plain GPT over it. Special tokens mark the
    seams:
  </p>

  <app-math display
    expr="\\underbrace{[\\text{BOS}]}_{\\text{start}}\\;\\,
          \\underbrace{t_0\\,t_1\\,\\cdots\\,t_k}_{\\text{text prompt}}\\;\\,
          \\underbrace{[\\text{BOI}]}_{\\text{image begins}}\\;\\,
          \\underbrace{c_0\\,c_1\\,\\cdots\\,c_{HW-1}}_{\\text{VQ codes, row-major}}\\;\\,
          \\underbrace{[\\text{EOI}]}_{\\text{image ends}}" />

  <p>
    Because attention is <strong>causal</strong> — every position sees only earlier ones —
    predicting <app-math expr="c_0" /> conditions on the whole prompt; predicting
    <app-math expr="c_1" /> conditions on the prompt <em>and</em> <app-math expr="c_0" />;
    and so on. "Make an image from text" is just "<em>continue the sentence after
    <app-math expr="[\\text{BOI}]" /></em>." The marquee below is that one sequence. Press
    <strong>step</strong> (or <strong>play</strong>) and the model samples one image code
    per click, in raster order, filling the grid. The bars show the (illustrative) softmax
    over candidate codes for the cell about to be drawn.
  </p>

  <!-- FIGURE 2 — interleaved generation marquee -->
  <figure class="fig">
    <div class="fig__stage fig__stage--seq">
      <div class="seq" role="img" [attr.aria-label]="'Interleaved token sequence, ' + filled() + ' of ' + nCodes + ' image codes generated'">
        <span class="seq__tok seq__tok--special">[BOS]</span>
        @for (t of promptTokens(); track $index) {
          <span class="seq__tok seq__tok--text">{{ t }}</span>
        }
        <span class="seq__tok seq__tok--special">[BOI]</span>
        @for (cell of cells(); track cell.i) {
          <span class="seq__tok seq__tok--code"
            [class.seq__tok--pending]="cell.state === 'pending'"
            [class.seq__tok--cursor]="cell.state === 'cursor'"
            [style.--code-color]="cell.color">
            {{ cell.state === 'pending' ? '·' : cell.code }}
          </span>
        }
        @if (done()) { <span class="seq__tok seq__tok--special">[EOI]</span> }
      </div>
    </div>

    <div class="fig__stage fig__stage--canvasrow">
      <div class="seqcv">
        <span class="vq__title">image so far <span class="mono">(codes → pixels)</span></span>
        <canvas #out class="fig__canvas seqcv__canvas"></canvas>
      </div>
      <div class="seqcv seqcv--soft">
        <span class="vq__title">
          next-token softmax @if (!done()) { <span class="mono">p(c<sub>{{ filled() }}</sub> | context)</span> }
          @else { <span class="mono">— sequence complete —</span> }
        </span>
        <canvas #soft class="fig__canvas seqcv__canvas"></canvas>
      </div>
    </div>

    <div class="ctl">
      <div class="ctl__row">
        <label class="ctl__label" for="prompt">prompt (its char tokens become <span class="mono">t<sub>0..k</sub></span>)</label>
        <input id="prompt" class="ctl__text" type="text" maxlength="22"
          [value]="prompt()" (input)="onPrompt($event)" placeholder="a small red house" />
      </div>
      <div class="ctl__row">
        <label class="ctl__label" for="temp">
          temperature <span class="mono">τ = {{ temperature().toFixed(2) }}</span>
          <span class="ctl__hint">(low = sharp/greedy · high = diverse)</span>
        </label>
        <input id="temp" class="ctl__range" type="range" min="0.2" max="1.6" step="0.05"
          [value]="temperature()" (input)="onTemp($event)" />
      </div>
      <div class="ctl__row ctl__row--btns">
        <button type="button" class="btn btn--primary" (click)="step()" [disabled]="done()">▸ step</button>
        <button type="button" class="btn" (click)="togglePlay()" [disabled]="done()">
          {{ playing() ? '❚❚ pause' : '▶ play' }}
        </button>
        <button type="button" class="btn" (click)="resetSeq()">↺ reset</button>
        <span class="ctl__group"><span class="mono">{{ filled() }}</span> / {{ nCodes }} codes</span>
      </div>
    </div>
    <figcaption class="fig__cap">
      The one interleaved sequence. <strong>[BOS]</strong>, the prompt's text tokens,
      <strong>[BOI]</strong>, then the image codes filled token-by-token in raster order.
      The cursor cell shows where next-token prediction is pointing; its softmax bar is the
      distribution the model samples from (illustrative — random weights here). This is the
      moment where text and image <em>truly</em> share one stream.
    </figcaption>
  </figure>

  <p>
    The sampling rule for each new token is the ordinary GPT loop: run the model on the
    current sequence, take the <em>last</em> position's logits (that's the prediction for the
    next token), temperature-scale, optionally top-<app-math expr="k" /> filter, draw from the
    softmax, and append. The slider above is the <app-math expr="\\tau" /> in
    <app-math expr="\\text{softmax}(\\mathrm{logits}/\\tau)" /> — small <app-math expr="\\tau" />
    sharpens toward the argmax (more coherent, less varied), large <app-math expr="\\tau" />
    flattens it (more diverse, more noise). It is the same code text GPTs use:
  </p>

  <app-code-ref
    file="pytorch/autoregressive/transformer.py"
    lang="python"
    [code]="snipGenerate"
    [lines]="[223, 240]"
    caption="TokenTransformer.generate — the causal next-token loop. Last-position logits → temperature → top-k → softmax → sample → append. Once per token, for HW tokens." />

  <p>
    The Transformer itself is deliberately modality-blind: it only sees integer ids in
    <app-math expr="[0, \\text{vocab})" />. The <em>caller</em> assembles the layout. Here is the
    generation driver — it builds <code>[BOS] text [BOI]</code>, asks <code>generate</code> for
    exactly <app-math expr="H\\cdot W" /> image-code tokens, strips the vocab offset, reshapes
    row-major into the code grid, and decodes to pixels — i.e. everything the marquee above is
    doing, one cell at a time:
  </p>

  <app-code-ref
    file="pytorch/autoregressive/sample.py"
    lang="python"
    [code]="snipAutoregressive"
    [lines]="[149, 162]"
    caption="autoregressive_generate — prefix [BOS]text[BOI], sample H·W codes, strip the offset, reshape row-major, decode. The whole text → tokens → pixels pipeline." />

  <h2>Why a shared vocabulary is the whole trick</h2>

  <p>
    Notice how little special-casing there is. Text ids occupy <app-math expr="[0, n_{\\text{text}})" />;
    three special tokens (<code>BOS</code>, <code>BOI</code>, <code>EOI</code>) sit just above;
    the <app-math expr="K" /> image codes occupy the top of the range at a fixed
    <code>code_offset</code>. One embedding table, one output head, one causal loss over the
    concatenation. That uniformity is exactly why this family scales to <strong>video</strong>
    almost for free: you don't add a new architecture, you just keep appending more frames' code
    grids to the same stream. "Next token == next patch == next frame" is one sequence, one
    objective.
  </p>

  <p>
    The downside is honest: token-by-token sampling is sequential (no all-at-once refinement),
    and historically these models trailed diffusion on raw image fidelity — quantization throws
    away detail, and one bad early token can derail the rest of the grid. But the gap has been
    closing fast, and the most interesting frontier is <strong>hybrid</strong>: autoregress
    <em>across</em> chunks (frames, or coarse latents) for long-range coherence, then run a small
    <em>diffusion</em> model <em>within</em> each chunk to fill in the high-frequency detail —
    borrowing the strength of both families. That is the through-line of this whole site: two
    very different machineries, increasingly meeting in the middle.
  </p>

  <p class="contrast">
    <strong>One sentence to take with you:</strong> diffusion refines the
    <em>entire</em> image at every step and iterates over <em>noise level</em>; the
    autoregressive family commits to <em>one token at a time</em> and iterates over
    <em>position</em>. Same goal, opposite order of operations.
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
      padding: 1.4rem; background:
        radial-gradient(120% 120% at 50% 0%, rgba(124,92,255,0.07), transparent 60%), var(--bg-0);
    }
    .fig__stage--split { display: grid; grid-template-columns: 1fr 1fr; gap: 1.1rem; align-items: start; }
    .fig__stage--seq { padding: 1.2rem 1.1rem; }
    .fig__stage--canvasrow {
      display: grid; grid-template-columns: minmax(0,0.85fr) minmax(0,1.15fr); gap: 1.1rem;
      border-top: 1px solid var(--line);
    }
    .fig__canvas { display: block; width: 100%; image-rendering: pixelated; }

    .vq { display: grid; gap: 0.5rem; }
    .vq__title { font-family: var(--font-mono); font-size: 0.72rem; color: var(--ink-3); letter-spacing: 0.04em; }
    .vq__title .mono { color: var(--ink-1); }
    .vq__canvas { aspect-ratio: 1 / 1; border-radius: var(--radius-sm); border: 1px solid var(--line-strong); touch-action: none; cursor: grab; }
    .vq__canvas:active { cursor: grabbing; }

    .seqcv { display: grid; gap: 0.5rem; }
    .seqcv__canvas { aspect-ratio: 1 / 1; border-radius: var(--radius-sm); border: 1px solid var(--line-strong); }
    .seqcv--soft .seqcv__canvas { aspect-ratio: 16 / 11; image-rendering: auto; }

    /* the interleaved-sequence marquee */
    .seq {
      display: flex; flex-wrap: wrap; gap: 4px; align-items: center;
      font-family: var(--font-mono); font-size: 0.72rem; line-height: 1;
    }
    .seq__tok {
      display: inline-flex; align-items: center; justify-content: center;
      min-width: 1.7em; padding: 0.36em 0.45em; border-radius: 6px;
      border: 1px solid var(--line); background: var(--bg-2); color: var(--ink-1);
    }
    .seq__tok--special { border-color: rgba(255,207,92,0.5); color: var(--warn); background: rgba(255,207,92,0.08); font-weight: 600; }
    .seq__tok--text { border-color: rgba(65,214,255,0.4); color: var(--plasma-b); background: rgba(65,214,255,0.07); }
    .seq__tok--code {
      color: #0a0c12; font-weight: 600;
      background: var(--code-color, var(--bg-2)); border-color: transparent;
    }
    .seq__tok--pending { background: var(--bg-2); border: 1px dashed var(--line-strong); color: var(--ink-3); font-weight: 400; }
    .seq__tok--cursor {
      background: var(--bg-2); border: 1px solid var(--plasma-a); color: var(--plasma-a); font-weight: 700;
      box-shadow: 0 0 0 2px rgba(124,92,255,0.25); animation: pulse 1.1s var(--ease) infinite;
    }
    @keyframes pulse { 0%,100% { box-shadow: 0 0 0 2px rgba(124,92,255,0.25); } 50% { box-shadow: 0 0 0 4px rgba(124,92,255,0.12); } }

    .ctl { padding: 1rem 1.1rem; border-top: 1px solid var(--line); display: grid; gap: 0.75rem; }
    .ctl__row { display: grid; gap: 0.5rem; }
    .ctl__row--btns { display: flex; flex-wrap: wrap; align-items: center; gap: 0.5rem; }
    .ctl__label { font-family: var(--font-mono); font-size: 0.8rem; color: var(--ink-2); }
    .ctl__hint { color: var(--ink-3); font-size: 0.92em; }
    .ctl__group { font-family: var(--font-mono); font-size: 0.78rem; color: var(--ink-3); }
    .mono { font-family: var(--font-mono); color: var(--ink-0); }

    .ctl__text {
      font-family: var(--font-mono); font-size: 0.85rem; color: var(--ink-0);
      background: var(--bg-3); border: 1px solid var(--line-strong);
      border-radius: var(--radius-sm); padding: 0.5rem 0.7rem; outline: none;
    }
    .ctl__text:focus { border-color: var(--plasma-a); }

    .ctl__range { -webkit-appearance: none; appearance: none; width: 100%; height: 6px;
      border-radius: 999px; background: linear-gradient(90deg, var(--plasma-a), var(--plasma-b));
      outline: none; cursor: pointer; }
    .ctl__range::-webkit-slider-thumb { -webkit-appearance: none; appearance: none;
      width: 18px; height: 18px; border-radius: 50%; background: #fff;
      border: 2px solid var(--plasma-a); box-shadow: 0 2px 8px rgba(0,0,0,0.5); cursor: pointer; }
    .ctl__range::-moz-range-thumb { width: 18px; height: 18px; border-radius: 50%; background: #fff;
      border: 2px solid var(--plasma-a); cursor: pointer; }

    .btn[disabled] { opacity: 0.4; cursor: not-allowed; }

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
    .chip--accent .chip__v { color: #cdbcff; }

    .fig__cap {
      padding: 0.75rem 1.1rem; border-top: 1px solid var(--line);
      font-size: 0.84rem; color: var(--ink-2); background: rgba(255,255,255,0.012);
    }
    .fig__cap strong { color: var(--ink-1); }

    .contrast {
      margin-top: 2rem; padding: 1rem 1.2rem;
      border-left: 3px solid var(--plasma-a); border-radius: var(--radius-sm);
      background: var(--accent-soft); color: var(--ink-1);
    }
    .contrast strong { color: #cdbcff; }

    @media (max-width: 640px) {
      .fig__stage--split, .fig__stage--canvasrow { grid-template-columns: 1fr; }
    }
  `],
})
export class Autoregressive {
  private readonly fb = inject(FirebaseService);
  private readonly destroyRef = inject(DestroyRef);

  // ====================================================================
  // shared codebook palette — K colors so a code id maps to a stable hue
  // ====================================================================
  readonly GRID = 8;                 // H' = W' code grid (figure 1 + figure 2)
  readonly nCodes = this.GRID * this.GRID;
  readonly codeCount = signal(16);   // K, the codebook size (figure 1 control)

  private codeColor(id: number, k: number): string {
    const hue = (id * 360) / Math.max(1, k);
    return `hsl(${hue.toFixed(0)}, 72%, 62%)`;
  }

  // ====================================================================
  // FIGURE 1 — vector quantization (live nearest-neighbour)
  // ====================================================================
  private readonly vqCv = viewChild.required<ElementRef<HTMLCanvasElement>>('vq');
  private readonly gridCv = viewChild.required<ElementRef<HTMLCanvasElement>>('grid');

  // encoder features: one 2-D point per grid cell (a 2-D stand-in for z_e ∈ R^D)
  private feats = signal<Float32Array>(this.makeFeatures(this.GRID, 7));
  // codebook: K 2-D points (e_k); we drag these too
  private book = signal<Float32Array>(this.makeBook(this.codeCount(), 3));

  // assignment: nearest code index per cell — the integer token grid
  readonly assign = computed<Int16Array>(() => {
    const f = this.feats(); const b = this.book(); const k = this.codeCount();
    const n = this.GRID * this.GRID;
    const out = new Int16Array(n);
    for (let c = 0; c < n; c++) {
      const fx = f[c * 2], fy = f[c * 2 + 1];
      let best = 0, bestD = Infinity;
      for (let j = 0; j < k; j++) {
        const dx = fx - b[j * 2], dy = fy - b[j * 2 + 1];
        const d = dx * dx + dy * dy;            // squared L2 — same metric as VectorQuantizer
        if (d < bestD) { bestD = d; best = j; }
      }
      out[c] = best;
    }
    return out;
  });
  readonly distinctCodes = computed(() => new Set(Array.from(this.assign())).size);

  // ---- feature / codebook factories -------------------------------------
  private makeFeatures(n: number, seed: number): Float32Array {
    const rng = this.rng(seed);
    const out = new Float32Array(n * n * 2);
    // cluster features into a few blobs so quantization looks meaningful
    const blobs = 4;
    const cx: number[] = [], cy: number[] = [];
    for (let i = 0; i < blobs; i++) { cx.push(0.2 + 0.6 * rng()); cy.push(0.2 + 0.6 * rng()); }
    for (let i = 0; i < n * n; i++) {
      const bI = i % blobs;
      const g = () => { const u1 = Math.max(1e-9, rng()), u2 = rng(); return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2); };
      out[i * 2] = Math.min(0.97, Math.max(0.03, cx[bI] + g() * 0.09));
      out[i * 2 + 1] = Math.min(0.97, Math.max(0.03, cy[bI] + g() * 0.09));
    }
    return out;
  }
  private makeBook(k: number, seed: number): Float32Array {
    const rng = this.rng(seed);
    const out = new Float32Array(k * 2);
    for (let j = 0; j < k; j++) { out[j * 2] = 0.08 + 0.84 * rng(); out[j * 2 + 1] = 0.08 + 0.84 * rng(); }
    return out;
  }
  private rng(seed: number): () => number {
    let st = (seed * 2654435761) >>> 0;
    return () => { st ^= st << 13; st ^= st >>> 17; st ^= st << 5; st >>>= 0; return st / 4294967296; };
  }

  // ---- drag interaction on the feature scatter --------------------------
  private dragKind: 'feat' | 'book' | null = null;
  private dragIdx = -1;
  vqDown(ev: PointerEvent): void {
    const p = this.canvasPoint(this.vqCv().nativeElement, ev);
    // hit-test codebook rings first (they're bigger targets), then features
    const b = this.book(), f = this.feats();
    let best: { kind: 'feat' | 'book'; idx: number; d: number } | null = null;
    for (let j = 0; j < this.codeCount(); j++) {
      const d = (p.x - b[j * 2]) ** 2 + (p.y - b[j * 2 + 1]) ** 2;
      if (!best || d < best.d) best = { kind: 'book', idx: j, d };
    }
    for (let c = 0; c < this.GRID * this.GRID; c++) {
      const d = (p.x - f[c * 2]) ** 2 + (p.y - f[c * 2 + 1]) ** 2;
      if (!best || d < best.d) best = { kind: 'feat', idx: c, d };
    }
    if (best && best.d < 0.0025) { this.dragKind = best.kind; this.dragIdx = best.idx; }
  }
  vqMove(ev: PointerEvent): void {
    if (!this.dragKind) return;
    const p = this.canvasPoint(this.vqCv().nativeElement, ev);
    const x = Math.min(0.98, Math.max(0.02, p.x)), y = Math.min(0.98, Math.max(0.02, p.y));
    if (this.dragKind === 'feat') {
      const f = this.feats().slice(); f[this.dragIdx * 2] = x; f[this.dragIdx * 2 + 1] = y; this.feats.set(f);
    } else {
      const b = this.book().slice(); b[this.dragIdx * 2] = x; b[this.dragIdx * 2 + 1] = y; this.book.set(b);
    }
  }
  vqUp(): void {
    if (this.dragKind) this.fb.event('interact', { section: 'autoregressive', control: 'vq-drag', kind: this.dragKind });
    this.dragKind = null; this.dragIdx = -1;
  }
  private canvasPoint(cv: HTMLCanvasElement, ev: PointerEvent): { x: number; y: number } {
    const r = cv.getBoundingClientRect();
    return { x: (ev.clientX - r.left) / r.width, y: (ev.clientY - r.top) / r.height };
  }

  setK(k: number): void {
    this.codeCount.set(k);
    this.book.set(this.makeBook(k, 3));
    this.fb.event('interact', { section: 'autoregressive', control: 'codebook-size', value: k });
  }
  reseedFeatures(): void {
    this.featSeed++;
    this.feats.set(this.makeFeatures(this.GRID, this.featSeed));
    this.fb.event('interact', { section: 'autoregressive', control: 're-encode' });
  }
  private featSeed = 7;

  // ---- figure-1 drawing -------------------------------------------------
  private drawScatter(): void {
    const cv = this.vqCv().nativeElement; const ctx = cv.getContext('2d'); if (!ctx) return;
    const dpr = Math.min(devicePixelRatio || 1, 2); const r = cv.getBoundingClientRect();
    const W = Math.max(1, Math.floor(r.width * dpr)), H = Math.max(1, Math.floor(r.height * dpr));
    if (cv.width !== W || cv.height !== H) { cv.width = W; cv.height = H; }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const w = r.width, h = r.height;
    ctx.clearRect(0, 0, w, h);

    const f = this.feats(), b = this.book(), a = this.assign(), k = this.codeCount();
    const X = (u: number) => u * w, Y = (v: number) => v * h;

    // assignment lines: each feature to its chosen code
    ctx.lineWidth = 1;
    for (let c = 0; c < this.GRID * this.GRID; c++) {
      const j = a[c];
      ctx.strokeStyle = 'rgba(255,255,255,0.10)';
      ctx.beginPath(); ctx.moveTo(X(f[c * 2]), Y(f[c * 2 + 1])); ctx.lineTo(X(b[j * 2]), Y(b[j * 2 + 1])); ctx.stroke();
    }
    // feature dots, tinted by their code
    for (let c = 0; c < this.GRID * this.GRID; c++) {
      ctx.fillStyle = this.codeColor(a[c], k);
      ctx.beginPath(); ctx.arc(X(f[c * 2]), Y(f[c * 2 + 1]), 3.2, 0, Math.PI * 2); ctx.fill();
    }
    // codebook rings (the K entries) with their id
    ctx.font = '600 10px ui-monospace, monospace'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    for (let j = 0; j < k; j++) {
      const cx = X(b[j * 2]), cy = Y(b[j * 2 + 1]);
      ctx.strokeStyle = this.codeColor(j, k); ctx.lineWidth = 2.4;
      ctx.beginPath(); ctx.arc(cx, cy, 8.5, 0, Math.PI * 2); ctx.stroke();
      ctx.fillStyle = '#05070c'; ctx.beginPath(); ctx.arc(cx, cy, 6.5, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = this.codeColor(j, k); ctx.fillText(String(j), cx, cy + 0.5);
    }
  }

  private drawCodeGrid(): void {
    const cv = this.gridCv().nativeElement; const ctx = cv.getContext('2d'); if (!ctx) return;
    const dpr = Math.min(devicePixelRatio || 1, 2); const r = cv.getBoundingClientRect();
    const W = Math.max(1, Math.floor(r.width * dpr)), H = Math.max(1, Math.floor(r.height * dpr));
    if (cv.width !== W || cv.height !== H) { cv.width = W; cv.height = H; }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const w = r.width, h = r.height; ctx.clearRect(0, 0, w, h);

    const a = this.assign(), n = this.GRID, k = this.codeCount();
    const cw = w / n, ch = h / n;
    ctx.font = `600 ${Math.min(cw, ch) * 0.36}px ui-monospace, monospace`;
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    for (let yy = 0; yy < n; yy++) {
      for (let xx = 0; xx < n; xx++) {
        const id = a[yy * n + xx];
        ctx.fillStyle = this.codeColor(id, k);
        ctx.fillRect(xx * cw + 1, yy * ch + 1, cw - 2, ch - 2);
        ctx.fillStyle = 'rgba(5,7,12,0.85)';
        ctx.fillText(String(id), xx * cw + cw / 2, yy * ch + ch / 2 + 0.5);
      }
    }
  }

  // ====================================================================
  // FIGURE 2 — interleaved generation marquee
  // ====================================================================
  private readonly outCv = viewChild.required<ElementRef<HTMLCanvasElement>>('out');
  private readonly softCv = viewChild.required<ElementRef<HTMLCanvasElement>>('soft');

  readonly prompt = signal('a small red house');
  readonly temperature = signal(0.9);
  readonly playing = signal(false);

  // char-level "tokens" of the prompt (mirrors the CharTokenizer the demo uses)
  readonly promptTokens = computed(() =>
    Array.from(this.prompt().slice(0, 22)).map((ch) => (ch === ' ' ? '␣' : ch)),
  );

  // the generated image codes: -1 = not yet generated
  private codes = signal<Int16Array>(new Int16Array(this.nCodes).fill(-1));
  readonly filled = computed(() => { const c = this.codes(); let n = 0; for (let i = 0; i < c.length; i++) if (c[i] >= 0) n++; return n; });
  readonly done = computed(() => this.filled() >= this.nCodes);

  // per-cell view model for the marquee
  readonly cells = computed(() => {
    const c = this.codes(); const filled = this.filled(); const k = this.figK;
    const out: { i: number; code: number; color: string; state: 'done' | 'cursor' | 'pending' }[] = [];
    for (let i = 0; i < this.nCodes; i++) {
      const code = c[i];
      const state = code >= 0 ? 'done' : i === filled ? 'cursor' : 'pending';
      out.push({ i, code, color: code >= 0 ? this.codeColor(code, k) : 'transparent', state });
    }
    return out;
  });

  // an illustrative logit field: a per-cell base bias + prompt/temperature flavour
  private readonly figK = 12;            // codebook size used by the marquee demo
  private logitSeed = 11;
  // current softmax over candidates for the cursor cell (for the bar chart)
  private softProbs = signal<Float32Array>(new Float32Array(this.figK));

  /** Illustrative next-token logits for cell `i`, biased by neighbours + prompt. */
  private logitsFor(i: number): Float32Array {
    const k = this.figK;
    const out = new Float32Array(k);
    // prompt hash → a few preferred codes, so the prompt visibly steers the codes
    let ph = 2166136261 >>> 0;
    const p = this.prompt();
    for (let s = 0; s < p.length; s++) { ph ^= p.charCodeAt(s); ph = Math.imul(ph, 16777619) >>> 0; }
    const c = this.codes();
    const row = Math.floor(i / this.GRID), col = i % this.GRID;
    // neighbour bias: prefer codes near the already-placed left / up neighbours (local coherence)
    const left = col > 0 ? c[i - 1] : -1;
    const up = row > 0 ? c[i - this.GRID] : -1;
    const rng = this.rng(this.logitSeed + i * 131);
    for (let j = 0; j < k; j++) {
      let v = (rng() - 0.5) * 1.4;                     // base randomness
      v += Math.cos((j + 1) * (ph % 17) * 0.21) * 1.1; // prompt-dependent preference
      if (left >= 0) v += Math.exp(-((j - left) ** 2) / 3) * 1.6;
      if (up >= 0) v += Math.exp(-((j - up) ** 2) / 3) * 1.6;
      out[j] = v;
    }
    return out;
  }

  private softmax(logits: Float32Array, tau: number): Float32Array {
    const k = logits.length; const out = new Float32Array(k);
    let max = -Infinity; for (let j = 0; j < k; j++) { const v = logits[j] / Math.max(0.05, tau); out[j] = v; if (v > max) max = v; }
    let sum = 0; for (let j = 0; j < k; j++) { out[j] = Math.exp(out[j] - max); sum += out[j]; }
    for (let j = 0; j < k; j++) out[j] /= sum;
    return out;
  }
  private sampleFrom(probs: Float32Array, u: number): number {
    let acc = 0; for (let j = 0; j < probs.length; j++) { acc += probs[j]; if (u <= acc) return j; }
    return probs.length - 1;
  }

  /** Recompute the softmax preview for the current cursor cell (no sampling). */
  private refreshSoftPreview(): void {
    if (this.done()) { this.softProbs.set(new Float32Array(this.figK)); return; }
    const probs = this.softmax(this.logitsFor(this.filled()), this.temperature());
    this.softProbs.set(probs);
  }

  step(): void {
    if (this.done()) return;
    const i = this.filled();
    const probs = this.softmax(this.logitsFor(i), this.temperature());
    const u = this.rng(this.logitSeed + i * 977 + Math.floor(this.temperature() * 100))();
    const chosen = this.sampleFrom(probs, u);
    const c = this.codes().slice(); c[i] = chosen; this.codes.set(c);
    this.fb.event('interact', { section: 'autoregressive', control: 'step', cell: i, code: chosen });
  }
  togglePlay(): void {
    this.playing.update((p) => !p);
    this.fb.event('interact', { section: 'autoregressive', control: 'play', value: this.playing() });
  }
  resetSeq(): void {
    this.playing.set(false);
    this.logitSeed++;
    this.codes.set(new Int16Array(this.nCodes).fill(-1));
    this.fb.event('interact', { section: 'autoregressive', control: 'reset' });
  }
  onPrompt(ev: Event): void {
    this.prompt.set((ev.target as HTMLInputElement).value);
    this.fb.event('interact', { section: 'autoregressive', control: 'prompt' });
  }
  onTemp(ev: Event): void {
    this.temperature.set(+(ev.target as HTMLInputElement).value);
  }

  private startPlayLoop(): void {
    let raf = 0; let last = performance.now(); let acc = 0; const stepEvery = 140;
    const loop = (now: number) => {
      const dt = now - last; last = now;
      if (this.playing()) {
        acc += dt;
        while (acc >= stepEvery) { acc -= stepEvery; if (this.done()) { this.playing.set(false); break; } this.step(); }
      } else { acc = 0; }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    this.destroyRef.onDestroy(() => cancelAnimationFrame(raf));
  }

  // ---- figure-2 drawing -------------------------------------------------
  private drawOut(): void {
    const cv = this.outCv().nativeElement; const ctx = cv.getContext('2d'); if (!ctx) return;
    const n = this.GRID; if (cv.width !== n) { cv.width = n; cv.height = n; }
    const c = this.codes(); const k = this.figK;
    const img = ctx.createImageData(n, n); const d = img.data;
    for (let i = 0; i < n * n; i++) {
      const code = c[i]; const p = i * 4;
      if (code < 0) { d[p] = 14; d[p + 1] = 17; d[p + 2] = 26; d[p + 3] = 255; continue; }
      const [r, g, b] = this.hslToRgb((code * 360) / k, 0.72, 0.62);
      d[p] = r; d[p + 1] = g; d[p + 2] = b; d[p + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }

  private drawSoft(): void {
    const cv = this.softCv().nativeElement; const ctx = cv.getContext('2d'); if (!ctx) return;
    const dpr = Math.min(devicePixelRatio || 1, 2); const r = cv.getBoundingClientRect();
    const W = Math.max(1, Math.floor(r.width * dpr)), H = Math.max(1, Math.floor(r.height * dpr));
    if (cv.width !== W || cv.height !== H) { cv.width = W; cv.height = H; }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const w = r.width, h = r.height; ctx.clearRect(0, 0, w, h);

    const probs = this.softProbs(); const k = this.figK;
    if (this.done()) {
      ctx.fillStyle = 'rgba(255,255,255,0.3)'; ctx.font = '12px ui-monospace, monospace';
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText('[EOI] — image complete', w / 2, h / 2);
      return;
    }
    const padL = 8, padR = 8, padT = 10, padB = 20;
    const plotW = w - padL - padR, plotH = h - padT - padB;
    const bw = plotW / k;
    let pmax = 0; for (let j = 0; j < k; j++) pmax = Math.max(pmax, probs[j]);
    pmax = Math.max(pmax, 1e-4);
    ctx.font = '9px ui-monospace, monospace'; ctx.textAlign = 'center';
    for (let j = 0; j < k; j++) {
      const bh = (probs[j] / pmax) * plotH;
      const x = padL + j * bw, y = padT + plotH - bh;
      ctx.fillStyle = this.codeColor(j, k);
      ctx.globalAlpha = 0.35 + 0.65 * (probs[j] / pmax);
      ctx.fillRect(x + 1.5, y, bw - 3, bh);
      ctx.globalAlpha = 1;
      ctx.fillStyle = 'rgba(255,255,255,0.45)';
      ctx.fillText(String(j), x + bw / 2, h - 7);
    }
    // axis baseline
    ctx.strokeStyle = 'rgba(255,255,255,0.12)'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(padL, padT + plotH); ctx.lineTo(w - padR, padT + plotH); ctx.stroke();
  }

  private hslToRgb(hDeg: number, s: number, l: number): [number, number, number] {
    const h = hDeg / 360;
    const f = (n: number) => {
      const a = s * Math.min(l, 1 - l);
      const j = (n + h * 12) % 12;
      return l - a * Math.max(-1, Math.min(j - 3, 9 - j, 1));
    };
    return [Math.round(f(0) * 255), Math.round(f(8) * 255), Math.round(f(4) * 255)];
  }

  // ====================================================================
  // lifecycle: draw on render + react to signals
  // ====================================================================
  constructor() {
    afterNextRender(() => {
      this.drawScatter(); this.drawCodeGrid();
      this.refreshSoftPreview();
      this.drawOut(); this.drawSoft();
      this.startPlayLoop();
      addEventListener('resize', this.onResize);
      this.destroyRef.onDestroy(() => removeEventListener('resize', this.onResize));
    });

    // figure 1 reacts to features / codebook / K
    effect(() => { this.feats(); this.book(); this.codeCount(); this.drawScatter(); this.drawCodeGrid(); });
    // figure 2: image redraws as codes fill in
    effect(() => { this.codes(); this.drawOut(); });
    // softmax preview tracks the cursor cell, prompt and temperature
    effect(() => { this.codes(); this.prompt(); this.temperature(); this.refreshSoftPreview(); });
    effect(() => { this.softProbs(); this.drawSoft(); });
  }

  private readonly onResize = () => {
    this.drawScatter(); this.drawCodeGrid(); this.drawSoft();
  };

  // ====================================================================
  // code snippets quoted verbatim from the repo
  // ====================================================================
  readonly snipQuantize = `# Squared L2 distance from every cell to every code, without forming the
# full outer product: ||a-b||^2 = ||a||^2 - 2 a.b + ||b||^2.
codebook = self.codebook.weight  # [K, D]
dist = (
    z_flat.pow(2).sum(dim=1, keepdim=True)              # [N, 1]
    - 2.0 * z_flat @ codebook.t()                       # [N, K]
    + codebook.pow(2).sum(dim=1)[None, :]               # [1, K]
)
# Nearest code per cell -> the integer tokens.
flat_indices = dist.argmin(dim=1)                       # [N]
indices = flat_indices.view(b, h, w)                    # [B, H, W]
z_q = self.codebook(flat_indices).view(b, h, w, d).permute(0, 3, 1, 2).contiguous()
# --- VQ loss (codebook + beta * commitment); .detach() is the stop-gradient
codebook_loss = F.mse_loss(z_q, z_e.detach())
commitment_loss = F.mse_loss(z_e, z_q.detach())
vq_loss = codebook_loss + self.beta * commitment_loss
# --- straight-through estimator: forward z_q (snapped), backward d z_q/d z_e == 1
z_q = z_e + (z_q - z_e).detach()`;

  readonly snipEncode = `@torch.no_grad()
def encode(self, x: Tensor) -> Tensor:
    """Image -> integer code grid [B, H', W'] (no gradients; inference helper).

    This is what you feed (flattened row-major) into the Transformer as the
    image half of the [BOS]text[BOI]codes[EOI] sequence.
    """
    z_e = self.encoder(x)
    _, indices, _ = self.quantizer(z_e)
    return indices

def decode_indices(self, indices: Tensor) -> Tensor:
    """Integer code grid [B, H', W'] -> reconstructed image [B, C, H, W]."""
    z_q = self.quantizer.embed_indices(indices)
    return self.decoder(z_q)`;

  readonly snipGenerate = `for _ in range(max_new):
    # Crop to the last max_len tokens so positions stay in range.
    idx_cond = idx if idx.size(1) <= self.max_len else idx[:, -self.max_len :]
    logits = self(idx_cond)[:, -1, :]  # [B, vocab] -- next-token logits

    if temperature != 1.0:
        logits = logits / max(temperature, 1e-8)

    if top_k is not None:
        k = min(top_k, logits.size(-1))
        kth = torch.topk(logits, k, dim=-1).values[:, -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))

    probs = F.softmax(logits, dim=-1)              # [B, vocab]
    next_token = torch.multinomial(probs, num_samples=1)  # [B, 1]
    idx = torch.cat([idx, next_token], dim=1)
return idx`;

  readonly snipAutoregressive = `# Prefix: [BOS] <text> [BOI]. Decoding continues right after [BOI].
seq = build_prompt_sequence(prompt_ids, vocab)        # [B, T+2]
prefix_len = seq.size(1)

# Sample exactly H*W image-code tokens (one per code-grid cell, row-major).
seq = transformer.generate(seq, max_new=n_codes, temperature=temperature, top_k=top_k)

# Slice out just the generated image tokens, strip the offset -> code ids [0, K).
image_tokens = seq[:, prefix_len : prefix_len + n_codes]   # [B, H*W]
codes = vocab.token_to_code(image_tokens)                  # [B, H*W] in [0, K)
code_grid = codes.view(-1, h, w).contiguous()              # [B, H, W] row-major

# Codes -> pixels via the VQ-VAE codebook + decoder.
image = vqvae.decode_indices(code_grid)                    # [B, C, H, W]`;
}
