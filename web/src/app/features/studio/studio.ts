import { Component, ChangeDetectionStrategy, signal, inject } from '@angular/core';
import { RouterLink } from '@angular/router';
import { FirebaseService } from '../../core/firebase.service';

/**
 * Live image Studio — a thin client for an SDXL endpoint that *you* run.
 *
 * It ships no model and no keys. It POSTs your prompt to whatever endpoint you
 * point it at (default: http://127.0.0.1:8000, i.e. the SSH tunnel to a Lambda
 * box started with `python -m lambda_lab.run start serve-sdxl`). With no box
 * running it is inert — a random visitor cannot generate anything. Generation
 * happens entirely on your GPU; SDXL applies no prompt filter, so lawful use is
 * your responsibility.
 */
@Component({
  selector: 'app-studio',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [RouterLink],
  template: `
<div class="studio">
  <header class="s-head">
    <a class="brandlink" routerLink="/">← Erosolar</a>
    <div>
      <h1>Erosolar Image Studio</h1>
      <p class="sub">Self-hosted SDXL — text-to-image and image-to-image. A front-end for an endpoint <em>you</em> run; nothing is generated on this site.</p>
    </div>
  </header>

  <div class="banner">
    <strong>Self-hosted &amp; unfiltered.</strong> Images are produced on your own GPU, which applies no
    prompt classifier. You are responsible for lawful use — no sexual content involving minors and no
    non-consensual imagery of real people. Everything else legal is yours to make.
  </div>

  <section class="panel connect">
    <label class="lbl">your SDXL endpoint</label>
    <div class="row">
      <input class="in in--grow" type="text" [value]="endpoint()" (input)="onEndpoint($event)" placeholder="http://127.0.0.1:8000" />
      <button class="btn" (click)="testConnection()">test connection</button>
    </div>
    <p class="status" [class.ok]="status() === 'ok'" [class.bad]="status() === 'bad'">{{ statusMsg() || 'not connected yet' }}</p>
  </section>

  <section class="grid">
    <div class="panel controls">
      <label class="lbl">prompt</label>
      <textarea class="in ta" rows="3" [value]="prompt()" (input)="onPrompt($event)" placeholder="a weathered lighthouse at dusk, volumetric fog, cinematic"></textarea>

      <label class="lbl">negative prompt <span class="muted">(optional)</span></label>
      <input class="in" type="text" [value]="negative()" (input)="onNeg($event)" placeholder="blurry, low quality, watermark" />

      <label class="lbl">init image <span class="muted">(optional — upload a picture to transform it)</span></label>
      <div class="row">
        <input class="in in--grow" type="file" accept="image/*" (change)="onFile($event)" />
        @if (initImage()) { <button type="button" class="btn" (click)="clearImage()">remove</button> }
      </div>
      @if (initImage()) {
        <div class="initwrap">
          <img class="initprev" [src]="initImage()" alt="upload preview" />
          <div class="sl initstr">
            <span class="lbl">change strength <b>{{ strength().toFixed(2) }}</b> <span class="muted">(higher = less like the upload)</span></span>
            <input type="range" min="0.2" max="0.95" step="0.05" [value]="strength()" (input)="onStrength($event)" />
          </div>
        </div>
      }

      <div class="sliders">
        <div class="sl">
          <span class="lbl">steps <b>{{ steps() }}</b></span>
          <input type="range" min="8" max="50" step="1" [value]="steps()" (input)="onSteps($event)" />
        </div>
        <div class="sl">
          <span class="lbl">guidance <b>{{ guidance().toFixed(1) }}</b></span>
          <input type="range" min="1" max="12" step="0.5" [value]="guidance()" (input)="onGuidance($event)" />
        </div>
        <div class="sl">
          <span class="lbl">size <b>{{ size() }}px</b></span>
          <input type="range" min="512" max="1024" step="128" [value]="size()" (input)="onSize($event)" />
        </div>
        <div class="sl">
          <span class="lbl">seed <span class="muted">(blank = random)</span></span>
          <input class="in" type="text" [value]="seed()" (input)="onSeed($event)" placeholder="random" />
        </div>
      </div>

      <button class="btn btn--primary gen" (click)="generate()" [disabled]="loading()">
        {{ loading() ? (initImage() ? 'transforming…' : 'generating…') : (initImage() ? 'transform ▸' : 'generate ▸') }}
      </button>
    </div>

    <div class="panel out">
      @if (image()) {
        <img class="result" [src]="image()" alt="generated image" />
        <a class="btn dl" [href]="image()" download="gen-lab-sdxl.png">download PNG</a>
      } @else {
        <div class="placeholder">
          <span class="ph-mark">◧</span>
          <p>Your image appears here. Start your box, open the tunnel, hit <em>generate</em>.</p>
        </div>
      }
    </div>
  </section>

  <section class="panel howto">
    <h2>Start your own box (≈ $1.29/hr, $10 cap)</h2>
    <pre class="code"><code>export LAMBDA_API_KEY=...            # cloud.lambda.ai/api-keys
python -m lambda_lab.run start serve-sdxl --budget 10

# it prints a tunnel command — run it in another terminal:
ssh -L 8000:localhost:8000 ubuntu@&lt;instance-ip&gt;

# then this page can reach http://127.0.0.1:8000 . When done:
python -m lambda_lab.run teardown sdxl-&lt;id&gt;</code></pre>
    <p class="note">
      The endpoint binds to localhost on the box and is reached only through your tunnel — it is never
      exposed publicly. A serving GPU bills until you tear it down, so kill it when you're finished.
      If the browser blocks the call, it's usually the tunnel being down or a mixed-content/CORS block
      (the server sends permissive CORS headers; make sure the tunnel is up).
    </p>
  </section>
</div>
  `,
  styles: [`
    :host { display: block; }
    .studio { max-width: 1000px; margin: 0 auto; padding: clamp(1.2rem, 4vw, 3rem) 1rem 5rem; }
    .s-head { display: flex; align-items: flex-start; gap: 1.2rem; margin-bottom: 1.4rem; }
    .brandlink { font-family: var(--font-mono); font-size: 0.8rem; color: var(--ink-2); text-decoration: none; padding-top: 0.4rem; white-space: nowrap; }
    .brandlink:hover { color: #fff; }
    .s-head h1 { font-size: var(--step-3); margin: 0; }
    .sub { color: var(--ink-2); margin: 0.2rem 0 0; }
    .sub em { color: var(--ink-0); font-style: italic; }

    .banner { border-left: 2px solid #ff5c8a; background: rgba(255,92,138,0.08); border-radius: var(--radius-sm); padding: 0.8rem 1rem; color: var(--ink-1); font-size: 0.86rem; margin-bottom: 1.4rem; }
    .banner strong { color: #ff9bb5; }

    .panel { border: 1px solid var(--line); border-radius: var(--radius); background: var(--bg-1); box-shadow: var(--shadow-1); padding: 1.1rem; }
    .connect { margin-bottom: 1.2rem; }
    .lbl { font-family: var(--font-mono); font-size: 0.7rem; letter-spacing: 0.08em; text-transform: uppercase; color: var(--ink-3); display: block; margin: 0.2rem 0 0.4rem; }
    .lbl b { color: var(--plasma-b); font-weight: 600; }
    .muted { color: var(--ink-3); text-transform: none; letter-spacing: 0; }
    .row { display: flex; gap: 0.6rem; }
    .in { font-family: var(--font-mono); font-size: 0.84rem; background: var(--bg-3); color: var(--ink-0); border: 1px solid var(--line-strong); border-radius: 8px; padding: 0.5rem 0.7rem; width: 100%; }
    .in--grow { flex: 1; }
    .ta { resize: vertical; line-height: 1.5; }
    .status { font-family: var(--font-mono); font-size: 0.78rem; color: var(--ink-3); margin: 0.6rem 0 0; }
    .status.ok { color: var(--plasma-b); }
    .status.bad { color: #ff9bb5; }

    .grid { display: grid; gap: 1.2rem; margin-bottom: 1.2rem; }
    @media (min-width: 760px) { .grid { grid-template-columns: 1fr 1fr; } }
    .controls { display: flex; flex-direction: column; }
    .sliders { display: grid; grid-template-columns: 1fr 1fr; gap: 0.8rem 1rem; margin: 1rem 0; }
    .sl { display: flex; flex-direction: column; gap: 0.3rem; }
    input[type=range] { width: 100%; accent-color: var(--plasma-a); }
    input[type=file].in { padding: 0.4rem 0.5rem; cursor: pointer; font-family: var(--font-mono); font-size: 0.78rem; }
    .initwrap { display: flex; gap: 0.8rem; align-items: flex-start; margin: 0.1rem 0 0.3rem; }
    .initprev { width: 92px; height: 92px; object-fit: cover; border-radius: 8px; border: 1px solid var(--line-strong); flex: none; }
    .initstr { flex: 1; min-width: 0; }
    .btn { font-family: var(--font-display); font-size: 0.85rem; padding: 0.5em 1em; border-radius: var(--radius-sm); border: 1px solid var(--line-strong); background: var(--bg-2); color: var(--ink-1); cursor: pointer; transition: border-color .15s, color .15s; }
    .btn:hover:not(:disabled) { color: #fff; border-color: rgba(124,92,255,0.6); }
    .btn:disabled { opacity: 0.5; cursor: progress; }
    .btn--primary { background: var(--grad-plasma); color: #0a0a12; border-color: transparent; font-weight: 600; }
    .gen { margin-top: auto; }

    .out { display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 0.8rem; min-height: 320px; }
    .result { width: 100%; max-width: 512px; border-radius: var(--radius-sm); border: 1px solid var(--line); }
    .dl { text-decoration: none; }
    .placeholder { text-align: center; color: var(--ink-3); }
    .ph-mark { font-size: 3rem; color: var(--line-strong); display: block; }
    .placeholder p { max-width: 30ch; margin: 0.6rem auto 0; font-size: 0.86rem; }
    .placeholder em { color: var(--ink-1); font-style: normal; }

    .howto h2 { font-size: var(--step-1); margin: 0 0 0.8rem; }
    .code { margin: 0; padding: 0.9rem 1rem; background: #0c0f17; border: 1px solid var(--line); border-radius: var(--radius-sm); overflow-x: auto; }
    .code code { font-family: var(--font-mono); font-size: 0.78rem; color: #cdd3e6; line-height: 1.7; white-space: pre; }
    .note { color: var(--ink-2); font-size: 0.82rem; margin: 0.8rem 0 0; line-height: 1.55; }
  `],
})
export class Studio {
  private readonly fb = inject(FirebaseService);

  readonly endpoint = signal('http://127.0.0.1:8000');
  readonly prompt = signal('');
  readonly negative = signal('');
  readonly steps = signal(30);
  readonly guidance = signal(6);
  readonly size = signal(1024);
  readonly seed = signal<string>('');

  readonly initImage = signal<string | null>(null); // uploaded image (data URL) for img2img
  readonly strength = signal(0.6);

  readonly loading = signal(false);
  readonly status = signal<'idle' | 'ok' | 'bad'>('idle');
  readonly statusMsg = signal('');
  readonly image = signal<string | null>(null);

  private base(): string { return this.endpoint().trim().replace(/\/+$/, ''); }

  onEndpoint(e: Event) { this.endpoint.set((e.target as HTMLInputElement).value); }
  onPrompt(e: Event) { this.prompt.set((e.target as HTMLTextAreaElement).value); }
  onNeg(e: Event) { this.negative.set((e.target as HTMLInputElement).value); }
  onSteps(e: Event) { this.steps.set(Number((e.target as HTMLInputElement).value)); }
  onGuidance(e: Event) { this.guidance.set(Number((e.target as HTMLInputElement).value)); }
  onSize(e: Event) { this.size.set(Number((e.target as HTMLInputElement).value)); }
  onSeed(e: Event) { this.seed.set((e.target as HTMLInputElement).value.replace(/[^0-9]/g, '')); }
  onStrength(e: Event) { this.strength.set(Number((e.target as HTMLInputElement).value)); }
  clearImage() { this.initImage.set(null); }
  onFile(e: Event) {
    const f = (e.target as HTMLInputElement).files?.[0];
    if (!f) return;
    const reader = new FileReader();
    reader.onload = () => this.initImage.set(reader.result as string);
    reader.readAsDataURL(f);
  }

  async testConnection(): Promise<void> {
    this.status.set('idle');
    this.statusMsg.set('checking…');
    try {
      const r = await fetch(this.base() + '/health');
      if (!r.ok) throw new Error('HTTP ' + r.status);
      const j = await r.json();
      this.status.set(j.ready ? 'ok' : 'bad');
      this.statusMsg.set(j.ready ? `connected · ${j.model} on ${j.device}` : 'reachable — model still loading, retry shortly');
    } catch {
      this.status.set('bad');
      this.statusMsg.set('cannot reach endpoint — is your box up and the SSH tunnel open?');
    }
  }

  async generate(): Promise<void> {
    const p = this.prompt().trim();
    if (!p) { this.status.set('bad'); this.statusMsg.set('enter a prompt first'); return; }
    this.loading.set(true);
    this.statusMsg.set('generating…');
    this.fb.event('interact', { section: 'studio', control: 'generate' });
    try {
      const r = await fetch(this.base() + '/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: p,
          negative: this.negative() || undefined,
          steps: this.steps(),
          guidance: this.guidance(),
          width: this.size(),
          height: this.size(),
          seed: this.seed() === '' ? undefined : Number(this.seed()),
          init_image: this.initImage() || undefined,
          strength: this.strength(),
        }),
      });
      const j = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(j.error || 'HTTP ' + r.status);
      this.image.set(j.image);
      this.status.set('ok');
      this.statusMsg.set('done');
    } catch (e) {
      this.status.set('bad');
      this.statusMsg.set('failed: ' + ((e as Error)?.message || 'connection error — check the tunnel'));
    } finally {
      this.loading.set(false);
    }
  }
}
