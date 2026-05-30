import { Injectable, inject } from '@angular/core';
import { DOCUMENT } from '@angular/common';
import { initializeApp, type FirebaseApp } from 'firebase/app';
import { getAnalytics, isSupported, logEvent, type Analytics } from 'firebase/analytics';

/** Web app configuration for the public `image-video-gen-models` Firebase project. */
export const firebaseConfig = {
  apiKey: 'AIzaSyA-OlS49nKpSROzFQL2aNOblShRd9HqD98',
  authDomain: 'image-video-gen-models.firebaseapp.com',
  projectId: 'image-video-gen-models',
  storageBucket: 'image-video-gen-models.firebasestorage.app',
  messagingSenderId: '745193454796',
  appId: '1:745193454796:web:e6220294f6cb8da45f5669',
  measurementId: 'G-FJKP33SZP8',
} as const;

/**
 * Initializes Firebase once and lazily attaches Analytics when the browser
 * supports it. Analytics is best-effort: blockers / unsupported environments
 * degrade silently rather than breaking the app.
 */
@Injectable({ providedIn: 'root' })
export class FirebaseService {
  private readonly doc = inject(DOCUMENT);
  readonly app: FirebaseApp = initializeApp(firebaseConfig);
  private analytics: Analytics | null = null;

  constructor() {
    if (this.hasWindow()) {
      isSupported()
        .then((ok) => {
          if (ok) this.analytics = getAnalytics(this.app);
        })
        .catch(() => void 0);
    }
  }

  /** Record a page view (called on every router navigation). */
  page(path: string, title: string): void {
    this.event('page_view', { page_path: path, page_title: title, page_location: this.location() });
  }

  /** Record a custom interaction event from an interactive figure. */
  event(name: string, params: Record<string, unknown> = {}): void {
    if (this.analytics) {
      try {
        logEvent(this.analytics, name, params);
      } catch {
        /* never let telemetry throw into the UI */
      }
    }
  }

  private hasWindow(): boolean {
    return typeof this.doc?.defaultView !== 'undefined' && this.doc.defaultView !== null;
  }

  private location(): string {
    return this.hasWindow() ? this.doc.defaultView!.location.href : '';
  }
}
