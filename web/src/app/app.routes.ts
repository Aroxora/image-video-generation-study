import { Routes } from '@angular/router';

/**
 * Each chapter is a lazily-loaded standalone component so the initial bundle
 * stays small and every interactive figure ships its own canvas code only
 * when that page is visited. Order mirrors the SECTIONS table.
 */
export const routes: Routes = [
  { path: '', loadComponent: () => import('./features/overview/overview').then((m) => m.Overview), title: 'How Image & Video Generation Works' },
  { path: 'diffusion', loadComponent: () => import('./features/diffusion/diffusion').then((m) => m.Diffusion), title: 'Diffusion — Forward & Reverse' },
  { path: 'latent', loadComponent: () => import('./features/latent/latent').then((m) => m.Latent), title: 'Latent Space & the VAE' },
  { path: 'denoiser', loadComponent: () => import('./features/denoiser/denoiser').then((m) => m.Denoiser), title: 'The Denoiser — U-Net → DiT' },
  { path: 'text', loadComponent: () => import('./features/text/text').then((m) => m.TextConditioning), title: 'Text Steering — Cross-Attention' },
  { path: 'guidance', loadComponent: () => import('./features/guidance/guidance').then((m) => m.Guidance), title: 'Classifier-Free Guidance' },
  { path: 'video', loadComponent: () => import('./features/video/video').then((m) => m.Video), title: 'Video — Spacetime Latents' },
  { path: 'autoregressive', loadComponent: () => import('./features/autoregressive/autoregressive').then((m) => m.Autoregressive), title: 'Autoregressive — Interleaved Tokens' },
  { path: 'build', loadComponent: () => import('./features/build/build').then((m) => m.Build), title: 'Build It Yourself in PyTorch' },
  { path: 'playground', loadComponent: () => import('./features/playground/playground').then((m) => m.Playground), title: 'Live Diffusion Playground' },
  { path: '**', redirectTo: '' },
];
