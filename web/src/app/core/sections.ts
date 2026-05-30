/**
 * The ordered table of contents. Every entry is one route, one nav item, and
 * one card on the home grid. `code` lists the PyTorch files in this repo that
 * implement the idea, so navigation and "build it yourself" stay in lockstep.
 */
export interface SectionMeta {
  /** route path (also the in-page slug) */
  slug: string;
  /** zero-padded index shown in the rail, e.g. "01" */
  no: string;
  /** short rail label */
  label: string;
  /** full page title */
  title: string;
  /** one-line summary for the home grid */
  blurb: string;
  /** which family this idea belongs to */
  family: 'foundations' | 'diffusion' | 'video' | 'autoregressive' | 'build';
  /** PyTorch files (repo-relative) that implement this idea */
  code: string[];
}

export const SECTIONS: SectionMeta[] = [
  {
    slug: '',
    no: '00',
    label: 'Overview',
    title: 'The two ways a machine dreams a picture',
    blurb: 'Diffusion denoises a whole clip at once; autoregression predicts tokens in sequence. Why the "next-frame" intuition is backwards.',
    family: 'foundations',
    code: ['pytorch/README.md'],
  },
  {
    slug: 'diffusion',
    no: '01',
    label: 'Diffusion',
    title: 'Forward & reverse: sculpting an image out of static',
    blurb: 'Add Gaussian noise step by step (no learning), then train a network to undo one step. Run it in reverse from pure static.',
    family: 'diffusion',
    code: ['pytorch/diffusion/schedule.py', 'pytorch/diffusion/ddpm.py'],
  },
  {
    slug: 'latent',
    no: '02',
    label: 'Latent space',
    title: 'Why Stable Diffusion denoises in a tiny latent grid',
    blurb: 'A VAE compresses 512×512×3 pixels to a 64×64×4 latent. Diffusion happens there; a decoder expands it back. That is what made it cheap.',
    family: 'diffusion',
    code: ['pytorch/diffusion/vae.py'],
  },
  {
    slug: 'denoiser',
    no: '03',
    label: 'The denoiser',
    title: 'U-Net → DiT: the network that predicts the noise',
    blurb: 'Inside the reverse step is one network. It started as a U-Net; increasingly it is a diffusion Transformer over patches.',
    family: 'diffusion',
    code: ['pytorch/diffusion/unet.py', 'pytorch/diffusion/dit.py'],
  },
  {
    slug: 'text',
    no: '04',
    label: 'Text steering',
    title: 'Cross-attention: how the prompt steers every step',
    blurb: 'Text becomes a sequence of embeddings (CLIP/T5). The image features query those tokens at every denoising step — a continuous steering signal.',
    family: 'diffusion',
    code: ['pytorch/diffusion/text_encoder.py', 'pytorch/diffusion/cross_attention.py'],
  },
  {
    slug: 'guidance',
    no: '05',
    label: 'Guidance',
    title: 'Classifier-free guidance: the prompt amplifier',
    blurb: 'Run the model with and without the prompt, then push the result in the direction the prompt adds. One slider, enormous effect.',
    family: 'diffusion',
    code: ['pytorch/diffusion/guidance.py'],
  },
  {
    slug: 'video',
    no: '06',
    label: 'Video',
    title: 'Spacetime latents: generating the whole clip at once',
    blurb: 'The latent becomes a frames×H×W volume. Chop it into spacetime patches, run a diffusion Transformer over all of them — motion holds together.',
    family: 'video',
    code: ['pytorch/video/spacetime.py', 'pytorch/video/temporal_attention.py', 'pytorch/video/video_dit.py'],
  },
  {
    slug: 'autoregressive',
    no: '07',
    label: 'Autoregressive',
    title: 'The other family: tokens, interleaved like an LLM',
    blurb: 'A VQ-VAE turns video into discrete tokens. A plain Transformer predicts them one at a time — where text and frames truly interleave.',
    family: 'autoregressive',
    code: ['pytorch/autoregressive/vqvae.py', 'pytorch/autoregressive/transformer.py', 'pytorch/autoregressive/sample.py'],
  },
  {
    slug: 'build',
    no: '08',
    label: 'Build it',
    title: 'Build it yourself in PyTorch',
    blurb: 'Every idea on this site is a real, small, runnable module. Here is how the pieces assemble into a training loop you can run today.',
    family: 'build',
    code: ['pytorch/train.py', 'pytorch/sample.py', 'pytorch/requirements.txt'],
  },
  {
    slug: 'playground',
    no: '09',
    label: 'Playground',
    title: 'Live diffusion playground',
    blurb: 'Real reverse diffusion, running in your browser: watch a cloud of noise denoise into a target shape using an exact score field.',
    family: 'build',
    code: ['pytorch/toy/toy_diffusion_2d.py'],
  },
];

/** Sections that appear in the rail / grid (everything except the overview root). */
export const CHAPTERS = SECTIONS.filter((s) => s.slug !== '');

export function sectionBySlug(slug: string): SectionMeta | undefined {
  return SECTIONS.find((s) => s.slug === slug);
}
