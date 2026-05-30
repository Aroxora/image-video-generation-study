import { Component, ChangeDetectionStrategy, input, computed, signal, inject } from '@angular/core';
import { DomSanitizer, type SafeHtml } from '@angular/platform-browser';
import { ghBlob } from '../core/repo';
import { highlight } from './hljs';

/**
 * A code panel that ties an on-page explanation to the real implementation.
 * Shows a syntax-highlighted snippet, the repo path, a copy button, and a
 * deep link to the exact file (and optional line range) on GitHub.
 *
 *   <app-code-ref
 *     file="pytorch/diffusion/schedule.py"
 *     lang="python"
 *     caption="The forward process is pure math — no network involved."
 *     [code]="snippet" />
 */
@Component({
  selector: 'app-code-ref',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <figure class="code">
      <header class="code__bar">
        <span class="code__dot" aria-hidden="true"></span>
        <a class="code__file" [href]="href()" target="_blank" rel="noopener">
          {{ file() }}@if (lineLabel()) {<span class="code__lines">{{ lineLabel() }}</span>}
        </a>
        <span class="code__lang">{{ lang() }}</span>
        <button type="button" class="code__copy" (click)="copy()" [attr.aria-label]="'Copy ' + file()">
          {{ copied() ? 'copied ✓' : 'copy' }}
        </button>
      </header>
      <pre class="code__pre"><code class="hljs" [innerHTML]="rendered()"></code></pre>
      @if (caption()) {<figcaption class="code__cap">{{ caption() }}</figcaption>}
    </figure>
  `,
  styles: [`
    .code {
      margin: 1.4rem 0;
      border: 1px solid var(--line);
      border-radius: var(--radius-sm);
      background: #0c0f17;
      overflow: hidden;
      box-shadow: var(--shadow-1);
    }
    .code__bar {
      display: flex; align-items: center; gap: 0.7rem;
      padding: 0.55rem 0.85rem;
      background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0));
      border-bottom: 1px solid var(--line);
      font-family: var(--font-mono); font-size: 0.78rem;
    }
    .code__dot { width: 9px; height: 9px; border-radius: 50%; background: var(--grad-plasma); flex: none; }
    .code__file { color: var(--ink-1); text-decoration: none; }
    .code__file:hover { color: #fff; }
    .code__lines { color: var(--ink-3); margin-left: 0.2rem; }
    .code__lang { margin-left: auto; color: var(--ink-3); text-transform: uppercase; letter-spacing: 0.12em; font-size: 0.66rem; }
    .code__copy {
      font-family: var(--font-mono); font-size: 0.7rem; cursor: pointer;
      color: var(--ink-2); background: var(--bg-3); border: 1px solid var(--line-strong);
      border-radius: 7px; padding: 0.2rem 0.55rem; transition: color .15s, border-color .15s;
    }
    .code__copy:hover { color: #fff; border-color: rgba(124,92,255,0.6); }
    .code__pre { margin: 0; padding: 1rem 1.1rem; overflow-x: auto; font-size: 0.82rem; line-height: 1.6; }
    .code__cap {
      padding: 0.6rem 1rem; border-top: 1px solid var(--line);
      color: var(--ink-2); font-size: 0.84rem; background: rgba(255,255,255,0.015);
    }
    code.hljs { background: transparent; }
  `],
})
export class CodeRef {
  readonly file = input.required<string>();
  readonly code = input.required<string>();
  readonly lang = input<'python' | 'typescript' | 'bash' | 'json'>('python');
  readonly caption = input<string>('');
  readonly lines = input<readonly [number, number] | null>(null);

  private readonly sanitizer = inject(DomSanitizer);
  readonly copied = signal(false);

  readonly href = computed(() => ghBlob(this.file(), this.lines() ?? undefined));
  readonly lineLabel = computed(() => {
    const l = this.lines();
    return l ? `:${l[0]}-${l[1]}` : '';
  });
  readonly rendered = computed<SafeHtml>(() =>
    this.sanitizer.bypassSecurityTrustHtml(highlight(this.code().replace(/\s+$/, ''), this.lang())),
  );

  copy(): void {
    const text = this.code();
    const nav = typeof navigator !== 'undefined' ? navigator : undefined;
    if (nav?.clipboard) {
      nav.clipboard.writeText(text).then(
        () => {
          this.copied.set(true);
          setTimeout(() => this.copied.set(false), 1400);
        },
        () => void 0,
      );
    }
  }
}
