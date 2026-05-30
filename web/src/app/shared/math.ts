import { Component, ChangeDetectionStrategy, input, computed } from '@angular/core';
import { DomSanitizer, type SafeHtml } from '@angular/platform-browser';
import { inject } from '@angular/core';
import katex from 'katex';

/**
 * Renders a LaTeX string with KaTeX. Use inline in prose:
 *   <app-math expr="\\alpha_t" />
 * or as a centered display block:
 *   <app-math display expr="x_t = \\sqrt{\\bar\\alpha_t}\\,x_0 + \\sqrt{1-\\bar\\alpha_t}\\,\\epsilon" />
 */
@Component({
  selector: 'app-math',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `<span [class.math-display]="display()" [innerHTML]="html()"></span>`,
  styles: [`
    :host { display: inline; }
    :host:has(.math-display) { display: block; }
    .math-display { display: block; }
  `],
})
export class Math {
  readonly expr = input.required<string>();
  readonly display = input(false, { transform: (v: boolean | '') => v === '' || v === true });

  private readonly sanitizer = inject(DomSanitizer);

  readonly html = computed<SafeHtml>(() => {
    let out: string;
    try {
      out = katex.renderToString(this.expr(), {
        displayMode: this.display(),
        throwOnError: false,
        output: 'html',
      });
    } catch {
      out = this.expr();
    }
    return this.sanitizer.bypassSecurityTrustHtml(out);
  });
}
