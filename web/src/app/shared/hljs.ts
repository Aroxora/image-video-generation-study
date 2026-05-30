import hljs from 'highlight.js/lib/core';
import python from 'highlight.js/lib/languages/python';
import typescript from 'highlight.js/lib/languages/typescript';
import bash from 'highlight.js/lib/languages/bash';
import json from 'highlight.js/lib/languages/json';

let ready = false;

/** Register only the languages we ship, then highlight one snippet. */
export function highlight(code: string, lang: string): string {
  if (!ready) {
    hljs.registerLanguage('python', python);
    hljs.registerLanguage('typescript', typescript);
    hljs.registerLanguage('bash', bash);
    hljs.registerLanguage('json', json);
    ready = true;
  }
  const language = hljs.getLanguage(lang) ? lang : 'plaintext';
  try {
    return hljs.highlight(code, { language, ignoreIllegals: true }).value;
  } catch {
    return code.replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c] ?? c));
  }
}
