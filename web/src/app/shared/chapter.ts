import { Component, ChangeDetectionStrategy, input, computed } from '@angular/core';
import { RouterLink } from '@angular/router';
import { SECTIONS, sectionBySlug } from '../core/sections';
import { ghBlob } from '../core/repo';

/**
 * Page chrome shared by every chapter: a numbered header that names the idea
 * and links straight to the PyTorch files that implement it, the projected
 * body, and a previous/next pager derived from the SECTIONS table.
 *
 *   <app-chapter slug="diffusion"> ...content... </app-chapter>
 */
@Component({
  selector: 'app-chapter',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [RouterLink],
  template: `
    <article class="chapter">
      <header class="chapter__head">
        <div class="chapter__no">{{ meta()?.no }}</div>
        <div>
          <span class="eyebrow">{{ meta()?.family }}</span>
          <h1 class="chapter__title">{{ meta()?.title }}</h1>
          <p class="chapter__blurb">{{ meta()?.blurb }}</p>
          <div class="chapter__code">
            <span class="chapter__code-label">implemented in</span>
            @for (f of meta()?.code; track f) {
              <a class="chapter__chip" [href]="blob(f)" target="_blank" rel="noopener">{{ short(f) }} ↗</a>
            }
          </div>
        </div>
      </header>

      <div class="chapter__body">
        <ng-content />
      </div>

      <nav class="pager">
        @if (prev(); as p) {
          <a class="pager__link" [routerLink]="'/' + p.slug">
            <span class="pager__dir">← previous</span>
            <span class="pager__name">{{ p.label }}</span>
          </a>
        } @else { <span></span> }
        @if (next(); as n) {
          <a class="pager__link pager__link--next" [routerLink]="'/' + n.slug">
            <span class="pager__dir">next →</span>
            <span class="pager__name">{{ n.label }}</span>
          </a>
        } @else { <span></span> }
      </nav>
    </article>
  `,
  styles: [`
    .chapter { max-width: 860px; margin: 0 auto; padding: clamp(1.5rem, 4vw, 3.5rem) 0 5rem; }
    .chapter__head { display: grid; grid-template-columns: auto 1fr; gap: 1.4rem; margin-bottom: 2.4rem; }
    .chapter__no {
      font-family: var(--font-mono); font-size: clamp(1.6rem, 5vw, 2.6rem);
      font-weight: 600; line-height: 1; color: transparent;
      background: var(--grad-plasma); -webkit-background-clip: text; background-clip: text;
      padding-top: 0.5rem;
    }
    .chapter__title { font-size: var(--step-3); margin: 0.5rem 0 0.6rem; }
    .chapter__blurb { color: var(--ink-1); font-size: var(--step-1); max-width: 60ch; }
    .chapter__code { display: flex; flex-wrap: wrap; align-items: center; gap: 0.5rem; margin-top: 1.1rem; }
    .chapter__code-label {
      font-family: var(--font-mono); font-size: 0.7rem; letter-spacing: 0.14em;
      text-transform: uppercase; color: var(--ink-3);
    }
    .chapter__chip {
      font-family: var(--font-mono); font-size: 0.74rem;
      padding: 0.28em 0.7em; border-radius: 999px;
      border: 1px solid rgba(124,92,255,0.35); background: var(--accent-soft); color: #cdbcff;
    }
    .chapter__chip:hover { color: #fff; border-color: var(--plasma-a); }
    .chapter__body { font-size: var(--step-0); }
    .chapter__body :where(h2) { font-size: var(--step-2); margin-top: 2.6rem; }
    .chapter__body :where(h3) { font-size: var(--step-1); margin-top: 2rem; }

    .pager { display: flex; justify-content: space-between; gap: 1rem; margin-top: 4rem; padding-top: 1.6rem; border-top: 1px solid var(--line); }
    .pager__link { display: grid; gap: 0.2rem; padding: 0.9rem 1.2rem; border: 1px solid var(--line); border-radius: var(--radius-sm); background: var(--bg-2); min-width: 180px; transition: border-color .15s, transform .12s; }
    .pager__link:hover { border-color: rgba(124,92,255,0.6); transform: translateY(-2px); }
    .pager__link--next { text-align: right; }
    .pager__dir { font-family: var(--font-mono); font-size: 0.72rem; color: var(--ink-3); }
    .pager__name { color: var(--ink-0); font-family: var(--font-display); font-weight: 500; }

    @media (max-width: 640px) {
      .chapter__head { grid-template-columns: 1fr; gap: 0.4rem; }
      .pager__link { min-width: 0; }
    }
  `],
})
export class Chapter {
  readonly slug = input.required<string>();

  private readonly index = computed(() => SECTIONS.findIndex((s) => s.slug === this.slug()));
  readonly meta = computed(() => sectionBySlug(this.slug()));
  readonly prev = computed(() => { const i = this.index(); return i > 0 ? SECTIONS[i - 1] : null; });
  readonly next = computed(() => { const i = this.index(); return i >= 0 && i < SECTIONS.length - 1 ? SECTIONS[i + 1] : null; });

  blob(f: string): string { return ghBlob(f); }
  short(f: string): string { return f.split('/').slice(-2).join('/'); }
}
