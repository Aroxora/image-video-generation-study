import { Component, ChangeDetectionStrategy, signal, inject, DOCUMENT } from '@angular/core';
import { RouterOutlet, RouterLink, RouterLinkActive, Router, NavigationEnd } from '@angular/router';
import { filter } from 'rxjs/operators';
import { SECTIONS } from './core/sections';
import { REPO } from './core/repo';
import { FirebaseService } from './core/firebase.service';

@Component({
  selector: 'app-root',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [RouterOutlet, RouterLink, RouterLinkActive],
  templateUrl: './app.html',
  styleUrl: './app.scss',
})
export class App {
  private readonly router = inject(Router);
  private readonly fb = inject(FirebaseService);
  private readonly doc = inject(DOCUMENT);

  readonly sections = SECTIONS;
  readonly repo = REPO;
  readonly menuOpen = signal(false);

  constructor() {
    this.router.events.pipe(filter((e): e is NavigationEnd => e instanceof NavigationEnd)).subscribe((e) => {
      this.menuOpen.set(false);
      this.fb.page(e.urlAfterRedirects, this.doc.title);
    });
  }

  toggleMenu(): void {
    this.menuOpen.update((v) => !v);
  }
}
