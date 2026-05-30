import {
  Component, ChangeDetectionStrategy, ElementRef, afterNextRender, inject, DestroyRef,
  viewChild, signal, computed, effect,
} from '@angular/core';
import { Chapter } from '../../shared/chapter';
import { CodeRef } from '../../shared/code-ref';
import { Math as MathTex } from '../../shared/math';
import { FirebaseService } from '../../core/firebase.service';

/**
 * Chapter /text — "Cross-attention: how the prompt steers every step".
 *
 * Marquee interaction (FIGURE 1): a live N×M attention heatmap. Rows are the
 * 8×8 grid of image patches, columns are the prompt's tokens. The attention is a
 * DETERMINISTIC illustration (token salience + a string-seeded pseudo-random
 * field, softmaxed) — responsive and stable, clearly labelled as a mock of the
 * mechanism, not a trained model. Hovering a patch highlights the tokens it most
 * attends to; hovering a token highlights the patches that pull it in.
 *
 * FIGURE 2: the data path prompt → tokenizer → text encoder → [L×d] → injected
 * at every denoising step via cross-attention, with Attention(Q,K,V).
 */
@Component({
  selector: 'app-text',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [Chapter, CodeRef, MathTex],
  template: `
<app-chapter slug="text">

  <p class="lede">
    A diffusion model paints from static. Left alone it would resolve <em>some</em>
    plausible image — but not <em>your</em> image. The prompt is the hand on the wheel:
    at <strong>every</strong> denoising step, each patch of the half-formed picture
    gets to read the words and pull itself toward the ones that describe it. That
    "reading" is <strong>cross-attention</strong>, and it is the single mechanism by
    which language becomes pixels.
  </p>

  <h2>The prompt is a steering signal, not a co-passenger</h2>

  <p>
    A common misconception is that the model generates the image and the caption
    side by side. It does not. The text is encoded <em>once</em>, up front, into a
    sequence of per-token vectors. Those vectors are then held fixed and re-injected
    into the denoiser on <em>every</em> step — step 1 of 50, step 50 of 50, all of
    them. Nothing about the text is "generated"; it is a constant field the image is
    continually nudged against. Change one word and you change the steering applied
    a thousand times over the course of sampling.
  </p>

  <h2>Figure 1 — Watch a prompt attend to image patches</h2>

  <p>
    Edit the prompt below. We tokenize it on the client and build an
    <strong>attention map</strong>: a grid of image patches (rows) against your
    tokens (columns). A bright cell means "this patch is paying a lot of attention to
    this word". Hover a patch to see which words it reads; hover a token chip to see
    which patches reach for it.
  </p>

  <div class="panel fig">
    <div class="fig__controls">
      <label class="fld">
        <span class="fld__label">prompt</span>
        <input
          class="fld__input"
          type="text"
          [value]="prompt()"
          (input)="onPrompt($event)"
          spellcheck="false"
          autocomplete="off"
          aria-label="prompt" />
      </label>
      <div class="fig__presets">
        @for (p of presets; track p) {
          <button type="button" class="chip-btn" (click)="usePreset(p)">{{ p }}</button>
        }
      </div>
    </div>

    <div class="tokens">
      @for (t of tokens(); track t.i) {
        <button
          type="button"
          class="tok"
          [class.tok--hot]="hoverTok() === t.i || (hoverPatch() >= 0 && topTokenFor(hoverPatch()) === t.i)"
          [style.--w]="colWeight(t.i)"
          (pointerenter)="setHoverTok(t.i)"
          (pointerleave)="setHoverTok(-1)">
          <span class="tok__id">{{ t.i }}</span>{{ t.text }}
        </button>
      }
      @if (tokens().length === 0) {
        <span class="tok tok--empty">(empty prompt → null context)</span>
      }
    </div>

    <div class="heat">
      <canvas #cv class="heat__cv"
        (pointermove)="onCanvasMove($event)"
        (pointerleave)="onCanvasLeave()"></canvas>
      <div class="heat__yaxis"><span>image patches&nbsp;→</span></div>
    </div>

    <p class="fig__readout">
      @if (hoverPatch() >= 0) {
        Patch <strong>#{{ hoverPatch() }}</strong> ({{ patchRC(hoverPatch()) }}) attends most to
        <strong class="hot">“{{ tokenText(topTokenFor(hoverPatch())) }}”</strong>
        ({{ topTokenPct(hoverPatch()) }}% of its attention).
      } @else if (hoverTok() >= 0) {
        Token <strong class="hot">“{{ tokenText(hoverTok()) }}”</strong> is pulled in hardest by
        <strong>{{ patchesForTok(hoverTok()) }}</strong> patches — the regions that "mean" that word.
      } @else {
        Hover a patch (canvas) or a token chip. Brighter = more attention.
        Columns: the {{ tokens().length }} tokens of your prompt. Rows: an 8×8 grid of image patches.
      }
    </p>
    <figcaption class="cap">
      Illustrative cross-attention — a deterministic mock (token salience + a
      string-seeded field, softmaxed per patch). It shows the <em>shape</em> of the
      mechanism, not the output of a trained model.
    </figcaption>
  </div>

  <h3>What the model actually computes here</h3>

  <p>
    Each image patch becomes a query <app-math expr="q" />. Each text token becomes a
    key <app-math expr="k" /> and a value <app-math expr="v" />. The patch scores every
    token by similarity, softmaxes those scores into weights that sum to one, and reads
    out a weighted blend of the token values:
  </p>

  <app-math display expr="\\mathrm{Attention}(Q,K,V)=\\mathrm{softmax}\\!\\left(\\frac{QK^{\\top}}{\\sqrt{d}}\\right)V" />

  <p>
    The crucial wiring: in <strong>cross-attention</strong>, <app-math expr="Q" /> comes
    from the <em>image</em> tokens while <app-math expr="K" /> and <app-math expr="V" />
    come from the <em>text</em> embeddings. (When <app-math expr="K,V" /> also come from
    the image, the very same module is plain self-attention — that is how one class
    serves both jobs in the code below.) The <app-math expr="1/\\sqrt{d}" /> factor keeps
    the dot products from blowing up before the softmax; the padding mask sends fake
    "filler" tokens to <app-math expr="-\\infty" /> so they get zero weight.
  </p>

  <app-code-ref
    file="pytorch/diffusion/cross_attention.py"
    lang="python"
    [code]="snipCross"
    caption="CrossAttention.forward — Q from the image x, K/V from the text context. One module does both self- and cross-attention."
    [lines]="[132, 161]" />

  <p>
    Notice there is no special "text path" in the network. Conditioning is just
    attention with the keys and values swapped out for the prompt. Drop the
    <code>context</code> argument and the model runs unconditionally — which, with a
    second pass, is exactly the trick that powers classifier-free guidance in the next
    chapter.
  </p>

  <h2>Figure 2 — From characters to a steering field</h2>

  <p>
    Before any attention can happen, the prompt has to become numbers. The data path is
    short and every stage matters:
  </p>

  <div class="panel path">
    @for (s of pathStages(); track s.id) {
      <div class="path__stage" [class.path__stage--active]="pathHot() === s.id"
        (pointerenter)="pathHot.set(s.id)" (pointerleave)="pathHot.set('')">
        <div class="path__kind">{{ s.kind }}</div>
        <div class="path__name">{{ s.name }}</div>
        <div class="path__shape">{{ s.shape }}</div>
      </div>
      @if (!$last) { <div class="path__arrow" aria-hidden="true">→</div> }
    }
  </div>
  <p class="fig__readout">
    @if (pathHot()) { {{ stageNote(pathHot()) }} }
    @else { Hover a stage. The last two boxes repeat <strong>once per denoising step</strong> — the embeddings are computed once, reused everywhere. }
  </p>
  <figcaption class="cap">
    The conditioning pipeline. Tokenizer + encoder run <strong>once</strong>; the
    cross-attention read runs <strong>at every step</strong> of sampling.
  </figcaption>

  <h3>Step 1 — tokenize</h3>

  <p>
    A tokenizer chops the string into discrete units and maps each to an integer id.
    Real systems use subword vocabularies (CLIP's BPE, T5's SentencePiece); this repo
    uses a dependency-free <strong>character</strong> tokenizer so the whole thing runs
    on a laptop. The interface is identical either way: <em>text → ids + a mask</em>
    marking which positions are real versus padding.
  </p>

  <app-code-ref
    file="pytorch/diffusion/text_encoder.py"
    lang="python"
    [code]="snipTok"
    caption="CharTokenizer.encode — prepend BOS, truncate to the window, right-pad. mask[b,i]=True marks a REAL token."
    [lines]="[79, 103]" />

  <h3>Step 2 — encode into per-token vectors</h3>

  <p>
    The ids run through a small Transformer (in production: a big, <em>frozen</em>
    CLIP or T5 encoder, pretrained on text and never trained alongside the diffusion
    model). The output is one <app-math expr="d" />-dimensional vector
    <strong>per token</strong> — not a single pooled sentence vector. Keeping each
    token separate is exactly what lets the image attend to "red" and "fox" and "snow"
    independently.
  </p>

  <app-code-ref
    file="pytorch/diffusion/text_encoder.py"
    lang="python"
    [code]="snipEnc"
    caption="TinyTextEncoder.encode_text — raw strings → (emb [B, L, d], mask). This (emb, mask) pair is the 'context' the backbone cross-attends to."
    [lines]="[225, 237]" />

  <p>
    That <app-math expr="[L \\times d]" /> matrix — <app-math expr="L" /> tokens, each a
    <app-math expr="d" />-vector — <em>is</em> the steering signal. The diffusion code
    only depends on this output contract, which is why a frozen CLIP or T5 can be
    swapped in as a drop-in replacement without touching the denoiser.
  </p>

  <h2>Why "next-frame prediction" is the wrong mental model here</h2>

  <p>
    It is tempting to imagine the model writing the image out left-to-right, predicting
    the next piece from the prompt and what it has drawn so far. That picture belongs to
    the <em>autoregressive</em> family (a later chapter), not to diffusion. A diffusion
    model holds the <em>entire</em> image at once and refines all of it together; the
    prompt never gets "consumed" — it stays a steering field, applied in full on every
    pass. Cross-attention is how that field touches every patch, every step, all the way
    from static to picture.
  </p>

</app-chapter>
  `,
  styles: [`
    :host { display: block; }
    .lede { font-size: var(--step-1); color: var(--ink-1); }
    .lede em { color: var(--ink-0); font-style: italic; }

    code {
      font-family: var(--font-mono); font-size: 0.86em;
      background: var(--bg-3); border: 1px solid var(--line);
      padding: 0.05em 0.4em; border-radius: 6px; color: #cdbcff;
    }

    .fig { padding: 1.1rem 1.1rem 0.9rem; margin: 1.4rem 0; }
    .fig__controls { display: grid; gap: 0.75rem; margin-bottom: 1rem; }
    .fld { display: grid; gap: 0.35rem; }
    .fld__label {
      font-family: var(--font-mono); font-size: 0.68rem; letter-spacing: 0.14em;
      text-transform: uppercase; color: var(--ink-3);
    }
    .fld__input {
      width: 100%; box-sizing: border-box;
      font-family: var(--font-mono); font-size: 0.95rem; color: var(--ink-0);
      background: var(--bg-0); border: 1px solid var(--line-strong);
      border-radius: var(--radius-sm); padding: 0.65rem 0.8rem; outline: none;
      transition: border-color .15s, box-shadow .15s;
    }
    .fld__input:focus { border-color: var(--plasma-a); box-shadow: 0 0 0 3px rgba(124,92,255,0.18); }

    .fig__presets { display: flex; flex-wrap: wrap; gap: 0.4rem; }
    .chip-btn {
      font-family: var(--font-mono); font-size: 0.72rem; cursor: pointer;
      color: var(--ink-2); background: var(--bg-2); border: 1px solid var(--line);
      border-radius: 999px; padding: 0.25em 0.7em; transition: color .15s, border-color .15s;
    }
    .chip-btn:hover { color: #fff; border-color: rgba(124,92,255,0.6); }

    .tokens { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-bottom: 0.95rem; }
    .tok {
      position: relative; cursor: default;
      font-family: var(--font-mono); font-size: 0.84rem; color: var(--ink-0);
      background:
        linear-gradient(rgba(124,92,255,calc(0.10 + var(--w,0) * 0.55)), rgba(124,92,255,calc(0.10 + var(--w,0) * 0.55))),
        var(--bg-2);
      border: 1px solid var(--line-strong);
      border-radius: var(--radius-sm); padding: 0.3em 0.55em 0.3em 0.5em;
      display: inline-flex; align-items: center; gap: 0.4em;
      transition: border-color .12s, transform .12s, box-shadow .12s;
    }
    .tok__id {
      font-size: 0.62rem; color: var(--ink-3);
      background: var(--bg-0); border-radius: 5px; padding: 0.05em 0.3em;
    }
    .tok--hot {
      border-color: var(--plasma-b); color: #fff; transform: translateY(-1px);
      box-shadow: 0 0 0 2px rgba(65,214,255,0.28);
    }
    .tok--empty { color: var(--ink-3); font-style: italic; background: var(--bg-2); }

    .heat { position: relative; }
    .heat__cv {
      display: block; width: 100%; height: 340px;
      border: 1px solid var(--line); border-radius: var(--radius-sm);
      background: var(--bg-0); cursor: crosshair; touch-action: none;
    }
    .heat__yaxis {
      position: absolute; left: -4px; top: 0; bottom: 0; display: flex; align-items: center;
      pointer-events: none;
    }
    .heat__yaxis span {
      font-family: var(--font-mono); font-size: 0.62rem; color: var(--ink-3);
      transform: rotate(180deg); writing-mode: vertical-rl; letter-spacing: 0.1em;
    }

    .fig__readout {
      font-size: 0.9rem; color: var(--ink-1); min-height: 2.6em;
      margin: 0.85rem 0 0.2rem; line-height: 1.5;
    }
    .fig__readout .hot { color: var(--plasma-b); }
    .hot { color: var(--plasma-b); }

    .cap {
      font-size: 0.8rem; color: var(--ink-2); line-height: 1.5;
      border-top: 1px solid var(--line); padding-top: 0.7rem; margin-top: 0.5rem;
    }
    .cap em { color: var(--ink-1); }

    .path {
      display: flex; flex-wrap: wrap; align-items: stretch; gap: 0.5rem;
      padding: 1rem; margin: 1.4rem 0 0.4rem;
    }
    .path__stage {
      flex: 1 1 110px; min-width: 110px;
      border: 1px solid var(--line-strong); border-radius: var(--radius-sm);
      background: var(--bg-2); padding: 0.7rem 0.65rem; cursor: default;
      transition: border-color .14s, transform .14s, background .14s;
    }
    .path__stage--active {
      border-color: var(--plasma-a); transform: translateY(-2px);
      background: var(--accent-soft);
    }
    .path__kind {
      font-family: var(--font-mono); font-size: 0.6rem; letter-spacing: 0.12em;
      text-transform: uppercase; color: var(--ink-3); margin-bottom: 0.3rem;
    }
    .path__name { font-family: var(--font-display); font-weight: 600; color: var(--ink-0); font-size: 0.92rem; }
    .path__shape { font-family: var(--font-mono); font-size: 0.72rem; color: var(--plasma-b); margin-top: 0.3rem; }
    .path__arrow { display: flex; align-items: center; color: var(--ink-3); font-size: 1.1rem; flex: none; }

    @media (max-width: 640px) {
      .path { flex-direction: column; }
      .path__arrow { transform: rotate(90deg); align-self: center; }
      .heat__cv { height: 280px; }
    }
  `],
})
export class TextConditioning {
  private readonly cv = viewChild.required<ElementRef<HTMLCanvasElement>>('cv');
  private readonly destroyRef = inject(DestroyRef);
  private readonly fb = inject(FirebaseService);

  // ---- controls --------------------------------------------------------------
  readonly prompt = signal('a red fox in deep snow');
  readonly presets = ['a red fox in deep snow', 'neon city at night, rain', 'a calm green lake, mountains'];

  readonly hoverPatch = signal(-1);   // image-patch index (0..GRID*GRID-1), -1 = none
  readonly hoverTok = signal(-1);     // token index, -1 = none

  private readonly GRID = 8;          // 8×8 = 64 image patches

  // ---- tokenization (client-side, illustrative) ------------------------------
  /** Split on whitespace; lightly subword-split long words so chips look like real tokens. */
  readonly tokens = computed(() => {
    const raw = this.prompt().toLowerCase().replace(/[^a-z0-9\s,]/g, ' ');
    const words = raw.split(/[\s,]+/).filter((w) => w.length > 0);
    const out: { i: number; text: string }[] = [];
    for (const w of words) {
      if (w.length <= 6) {
        out.push({ i: out.length, text: w });
      } else {
        // crude subword split so "mountains" → "mountai" + "ns"
        for (let p = 0; p < w.length; p += 6) {
          out.push({ i: out.length, text: w.slice(p, p + 6) });
        }
      }
      if (out.length >= 12) break; // keep the heatmap readable
    }
    return out;
  });

  /** Per-token base salience: content words carry more weight than tiny function words. */
  private readonly salience = computed(() =>
    this.tokens().map((t) => {
      const stop = new Set(['a', 'an', 'the', 'in', 'on', 'of', 'at', 'and', 'to', 'with']);
      const base = stop.has(t.text) ? 0.25 : 1.0;
      // longer / rarer-looking tokens get a small bump
      return base + Math.min(t.text.length, 8) * 0.05;
    }),
  );

  /**
   * Deterministic attention matrix A[patch][token], softmaxed across tokens per patch.
   * Built from token salience × a string-seeded pseudo-random affinity field so it is
   * stable for a given prompt yet varies meaningfully across patches and words.
   */
  private readonly attn = computed<number[][]>(() => {
    const toks = this.tokens();
    const sal = this.salience();
    const G = this.GRID;
    const P = G * G;
    const seed = this.hashString(this.prompt());
    const A: number[][] = [];
    if (toks.length === 0) return A;

    for (let p = 0; p < P; p++) {
      const row: number[] = new Array(toks.length);
      const px = (p % G) / (G - 1);          // 0..1
      const py = Math.floor(p / G) / (G - 1); // 0..1
      let max = -Infinity;
      for (let t = 0; t < toks.length; t++) {
        // each token "owns" a smooth gaussian blob whose centre is seeded by (token, prompt)
        const cx = this.frac(seed + t * 91.7 + 13.1);
        const cy = this.frac(seed + t * 57.3 + 7.9);
        const spread = 0.18 + this.frac(seed + t * 31.7) * 0.22;
        const dx = px - cx, dy = py - cy;
        const blob = Math.exp(-(dx * dx + dy * dy) / (2 * spread * spread));
        // a little high-freq texture so patches differ even within a blob
        const tex = 0.15 * this.frac(seed + p * 0.137 + t * 3.71);
        const logit = (Math.log(sal[t] + 1e-6) + blob * 2.4 + tex) * 1.4;
        row[t] = logit;
        if (logit > max) max = logit;
      }
      // softmax across tokens
      let sum = 0;
      for (let t = 0; t < toks.length; t++) { row[t] = Math.exp(row[t] - max); sum += row[t]; }
      for (let t = 0; t < toks.length; t++) row[t] /= sum;
      A.push(row);
    }
    return A;
  });

  /** Column weight (mean attention a token receives across all patches) → chip tint 0..1. */
  colWeight(tokIdx: number): number {
    const A = this.attn();
    if (A.length === 0) return 0;
    let s = 0;
    for (let p = 0; p < A.length; p++) s += A[p][tokIdx] ?? 0;
    const mean = s / A.length;
    // normalize against uniform so a single dominant token reads bright
    const n = this.tokens().length || 1;
    return Math.min(1, mean * n * 0.65);
  }

  // ---- readout helpers (template) --------------------------------------------
  tokenText(i: number): string { return this.tokens()[i]?.text ?? ''; }

  topTokenFor(patch: number): number {
    const A = this.attn();
    const row = A[patch];
    if (!row) return -1;
    let best = 0;
    for (let t = 1; t < row.length; t++) if (row[t] > row[best]) best = t;
    return best;
  }

  topTokenPct(patch: number): number {
    const A = this.attn();
    const t = this.topTokenFor(patch);
    if (t < 0 || !A[patch]) return 0;
    return Math.round(A[patch][t] * 100);
  }

  patchesForTok(tokIdx: number): number {
    const A = this.attn();
    let c = 0;
    for (let p = 0; p < A.length; p++) if (this.topTokenFor(p) === tokIdx) c++;
    return c;
  }

  patchRC(patch: number): string {
    const G = this.GRID;
    return `row ${Math.floor(patch / G)}, col ${patch % G}`;
  }

  // ---- input handlers --------------------------------------------------------
  onPrompt(ev: Event): void {
    this.prompt.set((ev.target as HTMLInputElement).value);
    this.hoverPatch.set(-1);
    this.fb.event('interact', { section: 'text', control: 'prompt' });
  }

  usePreset(p: string): void {
    this.prompt.set(p);
    this.hoverPatch.set(-1);
    this.fb.event('interact', { section: 'text', control: 'preset' });
  }

  setHoverTok(i: number): void {
    this.hoverTok.set(i);
    if (i >= 0) this.hoverPatch.set(-1);
  }

  onCanvasMove(ev: PointerEvent): void {
    const canvas = this.cv().nativeElement;
    const r = canvas.getBoundingClientRect();
    const G = this.GRID;
    const col = Math.floor(((ev.clientX - r.left) / r.width) * this.tokens().length || 0);
    // map x→token column, y→patch row across the FULL height
    const tokN = this.tokens().length;
    if (tokN === 0) { this.hoverPatch.set(-1); return; }
    const fy = (ev.clientY - r.top) / r.height; // 0..1 over the 64 rows
    const patchRowsTotal = G * G;
    const patch = Math.max(0, Math.min(patchRowsTotal - 1, Math.floor(fy * patchRowsTotal)));
    this.hoverPatch.set(patch);
    this.hoverTok.set(-1);
    void col;
  }

  onCanvasLeave(): void { this.hoverPatch.set(-1); }

  // ---- data-path figure ------------------------------------------------------
  readonly pathHot = signal('');
  readonly pathStages = signal([
    { id: 'prompt', kind: 'string', name: 'prompt', shape: '"a red fox…"' },
    { id: 'tok', kind: 'CharTokenizer', name: 'token ids', shape: '[B, L]' },
    { id: 'enc', kind: 'CLIP / T5', name: 'text encoder', shape: '[B, L, d]' },
    { id: 'ctx', kind: 'context', name: 'K, V cache', shape: '[B, L, d]' },
    { id: 'xattn', kind: 'every step', name: 'cross-attention', shape: 'Q·Kᵀ → V' },
  ]);

  stageNote(id: string): string {
    switch (id) {
      case 'prompt': return 'Raw text. Nothing is learned yet — just the words you typed.';
      case 'tok': return 'Tokenizer → integer ids + a padding mask (True = real token). Runs once.';
      case 'enc': return 'A frozen encoder turns ids into one vector PER token. Pretrained separately, never trained with the diffusion model.';
      case 'ctx': return 'The [L×d] embeddings become the keys & values the image will attend to. Computed once, then cached.';
      case 'xattn': return 'Image patches (queries) read the text via softmax(QKᵀ/√d)·V. This repeats on EVERY denoising step.';
      default: return '';
    }
  }

  // ---- deterministic helpers -------------------------------------------------
  private hashString(s: string): number {
    let h = 2166136261;
    for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); }
    return (h >>> 0) / 4294967295;
  }
  /** fractional part of a sine-scrambled value → pseudo-random in [0,1). */
  private frac(x: number): number {
    const v = Math.sin(x * 127.1 + 311.7) * 43758.5453;
    return v - Math.floor(v);
  }

  // ---- canvas ----------------------------------------------------------------
  constructor() {
    afterNextRender(() => this.run());
    // redraw whenever the attention matrix or hover state changes
    effect(() => { this.attn(); this.hoverPatch(); this.hoverTok(); this.requestDraw(); });
  }

  private ctx: CanvasRenderingContext2D | null = null;
  private dirty = true;
  private requestDraw(): void { this.dirty = true; }

  private run(): void {
    const canvas = this.cv().nativeElement;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    this.ctx = ctx;

    const resize = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const r = canvas.getBoundingClientRect();
      canvas.width = Math.max(1, Math.floor(r.width * dpr));
      canvas.height = Math.max(1, Math.floor(r.height * dpr));
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      this.dirty = true;
    };
    resize();
    window.addEventListener('resize', resize);

    let raf = 0;
    const loop = () => {
      if (this.dirty) { this.dirty = false; this.paint(); }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);

    this.destroyRef.onDestroy(() => {
      cancelAnimationFrame(raf);
      window.removeEventListener('resize', resize);
    });
  }

  /** Heat-color ramp: low = bg-blue → mid plasma-a → high plasma-b. */
  private heatColor(v: number): string {
    // v in [0,1]; mix three stops
    const stops = [
      [10, 14, 23],     // near bg-0
      [124, 92, 255],   // plasma-a
      [65, 214, 255],   // plasma-b
    ];
    const t = Math.max(0, Math.min(1, v));
    const seg = t < 0.5 ? 0 : 1;
    const lt = t < 0.5 ? t / 0.5 : (t - 0.5) / 0.5;
    const a = stops[seg], b = stops[seg + 1];
    const r = Math.round(a[0] + (b[0] - a[0]) * lt);
    const g = Math.round(a[1] + (b[1] - a[1]) * lt);
    const bl = Math.round(a[2] + (b[2] - a[2]) * lt);
    return `rgb(${r},${g},${bl})`;
  }

  private paint(): void {
    const ctx = this.ctx; if (!ctx) return;
    const canvas = this.cv().nativeElement;
    const r = canvas.getBoundingClientRect();
    const w = r.width, h = r.height;
    ctx.clearRect(0, 0, w, h);

    const A = this.attn();
    const toks = this.tokens();
    const G = this.GRID;
    const rows = G * G; // 64 image patches as rows
    const cols = toks.length;

    if (cols === 0) {
      ctx.fillStyle = 'rgba(255,255,255,0.35)';
      ctx.font = '13px ui-monospace, monospace';
      ctx.textAlign = 'center';
      ctx.fillText('empty prompt → null context (no steering)', w / 2, h / 2);
      return;
    }

    const padL = 8, padR = 8, padT = 6, padB = 6;
    const cellW = (w - padL - padR) / cols;
    const cellH = (h - padT - padB) / rows;
    const hp = this.hoverPatch();
    const ht = this.hoverTok();

    // find a per-frame max for normalization (visual contrast)
    let vmax = 0;
    for (let p = 0; p < rows; p++) for (let t = 0; t < cols; t++) if (A[p][t] > vmax) vmax = A[p][t];
    const norm = vmax > 0 ? 1 / vmax : 1;

    for (let p = 0; p < rows; p++) {
      const rowHot = hp === p;
      for (let t = 0; t < cols; t++) {
        const colHot = ht === t;
        let v = A[p][t] * norm;
        ctx.fillStyle = this.heatColor(v);
        // dim cells outside the hovered row/column to focus attention
        let alpha = 1;
        if (hp >= 0) alpha = rowHot ? 1 : 0.32;
        else if (ht >= 0) alpha = colHot ? 1 : 0.32;
        ctx.globalAlpha = alpha;
        const x = padL + t * cellW;
        const y = padT + p * cellH;
        ctx.fillRect(x, y, Math.ceil(cellW) - 0.5, Math.ceil(cellH) - 0.5);
      }
    }
    ctx.globalAlpha = 1;

    // outline hovered row (a single image patch) across all tokens
    if (hp >= 0) {
      const y = padT + hp * cellH;
      ctx.strokeStyle = '#41d6ff';
      ctx.lineWidth = 1.5;
      ctx.strokeRect(padL, y, w - padL - padR, cellH);
      // mark the strongest token cell
      const tt = this.topTokenFor(hp);
      if (tt >= 0) {
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 2;
        ctx.strokeRect(padL + tt * cellW, y, cellW, cellH);
      }
    }
    // outline hovered column (a token) across all patches
    if (ht >= 0) {
      const x = padL + ht * cellW;
      ctx.strokeStyle = '#41d6ff';
      ctx.lineWidth = 1.5;
      ctx.strokeRect(x, padT, cellW, h - padT - padB);
    }
  }

  // ---- real code snippets (faithful excerpts of the repo) --------------------
  readonly snipTok = `def encode(self, texts: List[str]) -> Tuple[Tensor, Tensor]:
    """Encode a list of strings -> \`\`(ids [B, L] long, mask [B, L] bool)\`\`.
    # ...
    """
    B = len(texts)
    L = self.max_len
    ids = torch.full((B, L), self.pad_id, dtype=torch.long)
    mask = torch.zeros((B, L), dtype=torch.bool)

    for b, text in enumerate(texts):
        # BOS marks "start of prompt"; the model can use it as a summary slot.
        toks = [self.bos_id]
        for ch in text:
            # ...
            toks.append(self._stoi.get(ch, self.pad_id))
        toks = toks[:L]  # truncate over-long prompts to the window
        n = len(toks)
        ids[b, :n] = torch.tensor(toks, dtype=torch.long)
        mask[b, :n] = True
    return ids, mask`;

  readonly snipEnc = `@torch.no_grad()
def encode_text(self, texts: List[str]) -> Tuple[Tensor, Tensor]:
    """Convenience: raw strings -> \`\`(emb [B, L, dim], mask [B, L] bool)\`\`.

    Runs the internal tokenizer then the encoder, on the module's own device.
    This is the method samplers / training scripts call to turn a prompt list
    into \`\`context\`\`.
    """
    device = self.pos_emb.device
    ids, mask = self.tokenizer.encode(texts)
    ids, mask = ids.to(device), mask.to(device)
    emb = self.forward(ids, mask)
    return emb, mask`;

  readonly snipCross = `def forward(
    self,
    x: Tensor,
    context: Optional[Tensor] = None,
    mask: Optional[Tensor] = None,
) -> Tensor:
    # If no context, this is self-attention: tokens attend among themselves.
    ctx = context if context is not None else x

    q = self._split_heads(self.to_q(x))      # [B, H, N, d]
    k = self._split_heads(self.to_k(ctx))    # [B, H, L, d]
    v = self._split_heads(self.to_v(ctx))    # [B, H, L, d]

    # Build an additive attention-bias from the key-padding mask, if given.
    # mask is True for REAL tokens; we must MASK OUT the False (pad) positions.
    attn_mask = None
    if mask is not None and context is not None:
        keep = mask[:, None, None, :].to(q.dtype)          # 1.0 real, 0.0 pad
        attn_mask = (1.0 - keep) * torch.finfo(q.dtype).min

    # Prefer PyTorch's fused scaled-dot-product attention (flash/mem-efficient on
    # GPU, correct on CPU). It applies softmax(QK^T/sqrt(d) + attn_mask) @ V.
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)  # [B,H,N,d]

    # merge heads -> [B, N, heads*dim_head] -> project back to query_dim
    b, h, n, d = out.shape
    out = out.transpose(1, 2).reshape(b, n, h * d)
    return self.to_out(out)`;
}
