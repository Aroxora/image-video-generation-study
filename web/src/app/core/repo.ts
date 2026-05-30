/**
 * Single source of truth for linking the website's explanations back to the
 * open PyTorch implementation in this repository. Every <app-code-ref> uses
 * these helpers so a description on the page is one click from the code that
 * implements it.
 */
export const REPO = {
  owner: 'Aroxora',
  name: 'image-video-generation-study',
  branch: 'main',
  base: 'https://github.com/Aroxora/image-video-generation-study',
} as const;

/** Link to a file (optionally a line range) in the repo on GitHub. */
export function ghBlob(path: string, lines?: readonly [number, number]): string {
  const clean = path.replace(/^\/+/, '');
  const frag = lines ? `#L${lines[0]}-L${lines[1]}` : '';
  return `${REPO.base}/blob/${REPO.branch}/${clean}${frag}`;
}

/** Link to a directory in the repo on GitHub. */
export function ghTree(path: string): string {
  return `${REPO.base}/tree/${REPO.branch}/${path.replace(/^\/+/, '')}`;
}
