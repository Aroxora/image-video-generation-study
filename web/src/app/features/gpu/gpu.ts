import {
  Component, ChangeDetectionStrategy, inject, DestroyRef, signal, computed, afterNextRender,
} from '@angular/core';
import { RouterLink } from '@angular/router';
import { Chapter } from '../../shared/chapter';
import { CodeRef } from '../../shared/code-ref';
import { Math as MathTex } from '../../shared/math';
import { FirebaseService } from '../../core/firebase.service';

/**
 * Chapter 10 — Rent a GPU.
 *
 * The bridge from "I built a teaching-scale model" to "I ran the real, full-size
 * open-weight models economically on rented hardware." Three interactive figures:
 *   1. COST EXPLORER — the account's exact Lambda lineup; pick an instance + hours
 *      (or pick a task) and watch the cheapest-that-fits answer + estimate update.
 *   2. $/OUTPUT — turn throughput into dollars-per-1000-images / per-minute-of-video
 *      and compare to a pay-per-call API to find the break-even volume.
 *   3. RUN SIMULATOR — the agentic state machine animated: interrupt it (SSH drops),
 *      resume (it skips completed steps), or starve the budget (it force-terminates).
 * Then the real orchestrator code, a verified quickstart, the ARM/GH200 note, and
 * an honest responsible-use close. Everything maps to the runnable lambda_lab/ pkg.
 */

interface Inst { id: string; label: string; vram: number; hr: number; arch: string; }
interface Task { key: string; label: string; vram: number; trick: string; }
interface SimStep { name: string; dur: number; }

@Component({
  selector: 'app-gpu',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [Chapter, CodeRef, MathTex, RouterLink],
  template: `
<app-chapter slug="gpu">

  <p class="lede">
    The last chapter ended on a promise: the billion-parameter systems are
    <em>architecturally the same machine</em> as the tiny one you can run on a
    laptop — just bigger boxes on the same diagram. This chapter is how you
    actually run those bigger boxes without owning a single GPU. You
    <strong>rent</strong> one of NVIDIA's best accelerators by the minute, fine-tune
    an open-weight model on it, generate, and give it back — often for less than the
    cost of lunch. And because babysitting a $4/hr meter is a great way to lose
    money, the whole loop is driven by a small, resumable, budget-capped agent.
  </p>

  <div class="callout">
    <span class="callout__tag tag tag--accent">what "unlocked" means here</span>
    <p>
      <strong>Open-weight</strong> models you download and fully control — FLUX, SDXL,
      SD&nbsp;3.5, Wan, HunyuanVideo, LTX-Video — instead of a closed API that meters
      every call and decides what you may make. You rent the metal, run the weights,
      own the pipeline. With control comes responsibility: respect each model's
      <em>license</em> and the law (more at the end).
    </p>
  </div>

  <h2>1 · The lineup, and the one that wins</h2>
  <p>
    These are the exact on-demand instances available on the account this site was
    built for. Sort by what matters — dollars per gigabyte of VRAM — and an
    unexpected winner appears: the <strong>GH200</strong> has <em>more</em> memory
    than an H100 and costs <em>less</em> than the H100 PCIe. The catch is that it's
    ARM64 (we'll get to that).
  </p>

  <figure class="fig">
    <div class="ctl__row ctl__row--btns">
      <span class="ctl__label">mode</span>
      <button type="button" class="btn" [class.btn--primary]="costMode() === 'pick'" (click)="setCostMode('pick')">pick an instance</button>
      <button type="button" class="btn" [class.btn--primary]="costMode() === 'task'" (click)="setCostMode('task')">pick a task</button>
    </div>

    <div class="tbl-wrap">
      <table class="tbl">
        <thead>
          <tr><th>instance</th><th>VRAM</th><th>$/hr</th><th>arch</th><th>$/hr ÷ GB</th><th></th></tr>
        </thead>
        <tbody>
          @for (i of instancesSorted(); track i.id) {
            <tr
              [class.tbl__row--on]="costMode() === 'pick' && selected().id === i.id"
              [class.tbl__row--fit]="costMode() === 'task' && fitId() === i.id"
              [class.tbl__row--dim]="costMode() === 'task' && i.vram < task().vram"
              (click)="pick(i)">
              <td class="mono">{{ i.label }}<span class="tbl__id">{{ i.id }}</span></td>
              <td class="mono">{{ i.vram }} GB</td>
              <td class="mono">\${{ i.hr.toFixed(2) }}</td>
              <td><span class="pill" [class.pill--arm]="i.arch === 'arm64'">{{ i.arch }}</span></td>
              <td class="mono">\${{ (i.hr / i.vram).toFixed(3) }}</td>
              <td class="tbl__mark">
                @if (costMode() === 'pick' && selected().id === i.id) { ◀ }
                @if (costMode() === 'task' && fitId() === i.id) { <span class="tbl__cheapest">cheapest fit ✓</span> }
              </td>
            </tr>
          }
        </tbody>
      </table>
    </div>

    @if (costMode() === 'pick') {
      <div class="ctl">
        <span class="ctl__label">how long will you keep it? <span class="ctl__t">{{ hoursLabel() }}</span></span>
        <input type="range" min="0.1" max="8" step="0.1" [value]="hours()" (input)="onHours($event)" />
      </div>
      <div class="readout">
        <div class="readout__item"><span class="readout__k">instance</span><span class="readout__v">{{ selected().label }}</span></div>
        <div class="readout__item"><span class="readout__k">rate</span><span class="readout__v">\${{ selected().hr.toFixed(2) }}/hr</span></div>
        <div class="readout__item readout__item--big">
          <span class="readout__k">estimated cost</span>
          <span class="readout__v gradient-text">\${{ estimate() }}</span>
        </div>
      </div>
    } @else {
      <div class="ctl__row ctl__row--btns">
        @for (t of tasks; track t.key) {
          <button type="button" class="chip chip--task" [class.chip--on]="task().key === t.key" (click)="pickTask(t)">{{ t.label }}</button>
        }
      </div>
      <div class="readout">
        <div class="readout__item"><span class="readout__k">needs ≈</span><span class="readout__v">{{ task().vram }} GB</span></div>
        <div class="readout__item readout__item--big"><span class="readout__k">cheapest box that fits</span><span class="readout__v gradient-text">{{ fit()?.label }}</span></div>
        <div class="readout__item"><span class="readout__k">at</span><span class="readout__v">\${{ fit()?.hr?.toFixed(2) }}/hr</span></div>
      </div>
      <p class="fig__cap"><strong>{{ task().label }}.</strong> {{ task().trick }}</p>
    }
  </figure>

  <h2>2 · Renting beats renting-by-the-call — past a point</h2>
  <p>
    A hosted image API charges per picture; a rented GPU charges per minute. The GPU
    wins as soon as you generate enough to keep it busy. Cost per output is just the
    hourly rate divided by throughput:
  </p>
  <app-math display
    expr="\\text{cost per image} = \\frac{\\text{hourly rate}}{\\text{images per hour}} \\qquad \\text{cost per minute of video} = \\frac{\\text{hourly rate}}{3600}\\times\\big(\\text{compute-sec per output-sec}\\big)\\times 60" />

  <figure class="fig">
    <div class="ctl__row ctl__row--btns">
      <span class="ctl__label">output</span>
      <button type="button" class="btn" [class.btn--primary]="outMode() === 'image'" (click)="setOutMode('image')">images</button>
      <button type="button" class="btn" [class.btn--primary]="outMode() === 'video'" (click)="setOutMode('video')">video</button>
      <span class="ctl__label ctl__label--r">on</span>
      <select class="select" (change)="onOutInst($event)">
        @for (i of instances; track i.id) { <option [value]="i.id" [selected]="outInst().id === i.id">{{ i.label }} — \${{ i.hr.toFixed(2) }}/hr</option> }
      </select>
    </div>

    @if (outMode() === 'image') {
      <div class="ctl">
        <span class="ctl__label">throughput <span class="ctl__t">{{ imgPerMin() }} images / min</span></span>
        <input type="range" min="2" max="120" step="1" [value]="imgPerMin()" (input)="onImgRate($event)" />
      </div>
      <div class="ctl">
        <span class="ctl__label">a hosted API charges <span class="ctl__t">\${{ apiImg().toFixed(3) }} / image</span></span>
        <input type="range" min="0.005" max="0.1" step="0.005" [value]="apiImg()" (input)="onApiImg($event)" />
      </div>
      <div class="readout">
        <div class="readout__item readout__item--big"><span class="readout__k">your cost / 1000 imgs</span><span class="readout__v gradient-text">\${{ selfPer1k() }}</span></div>
        <div class="readout__item"><span class="readout__k">API cost / 1000</span><span class="readout__v">\${{ apiPer1k() }}</span></div>
        <div class="readout__item"><span class="readout__k">verdict</span><span class="readout__v" [class.win]="selfWins()">{{ selfWins() ? 'self-host wins ✓' : 'API still cheaper' }}</span></div>
      </div>
      <p class="fig__cap">
        @if (selfWins()) {
          Per-minute billing means you pay for ~15 min of boot + weight download up front,
          then come out ahead after <strong>{{ breakevenImgs() }}</strong> images in a session
          versus paying \${{ apiImg().toFixed(3) }}/image.
        } @else {
          At this throughput the API is still cheaper per image — raise throughput (bigger
          batches, fewer steps, a faster model) or pick a cheaper instance to flip it.
        }
      </p>
    } @else {
      <div class="ctl">
        <span class="ctl__label">render cost <span class="ctl__t">{{ secPerSec() }} compute-sec per 1s of video</span></span>
        <input type="range" min="5" max="120" step="1" [value]="secPerSec()" (input)="onSecPerSec($event)" />
      </div>
      <div class="readout">
        <div class="readout__item readout__item--big"><span class="readout__k">cost / minute of video</span><span class="readout__v gradient-text">\${{ videoPerMin() }}</span></div>
        <div class="readout__item"><span class="readout__k">cost / 5s clip</span><span class="readout__v">\${{ videoPerClip() }}</span></div>
        <div class="readout__item"><span class="readout__k">on</span><span class="readout__v">{{ outInst().label }}</span></div>
      </div>
      <p class="fig__cap">
        Open video models trade compute for time: a few dozen seconds of GPU per second
        of footage. At these rates a short clip is pennies to a few cents — drag to match
        your model (Wan-1.3B is fast; HunyuanVideo-14B is slow).
      </p>
    }
  </figure>

  <h3>What a whole job actually costs</h3>
  <p>
    Sliders are nice, but here are concrete end-to-end expectations on the cheapest box
    that fits each job — <strong>including</strong> the ~10–15 min of boot + weight
    download you pay for before the first output (the real hidden cost, which a
    persistent filesystem amortizes across later runs).
  </p>
  <div class="tbl-wrap">
    <table class="tbl">
      <thead><tr><th>job</th><th>cheapest box that fits</th><th>~wall-clock (incl. setup)</th><th>~total</th></tr></thead>
      <tbody>
        @for (r of jobCosts; track r.job) {
          <tr>
            <td>{{ r.job }}</td>
            <td class="mono">{{ r.inst }}</td>
            <td class="mono">{{ r.time }}</td>
            <td class="mono"><strong class="gradient-text">{{ r.total }}</strong></td>
          </tr>
        }
      </tbody>
    </table>
  </div>
  <p class="fig__cap">
    Ballpark estimates at the verified on-demand rates — real cost scales with steps,
    resolution, and model size. The two things that blow a budget are forgetting to tear
    down and re-downloading weights every boot; the orchestrator's guard and a persistent
    filesystem handle both. A whole afternoon of experiments rarely tops <strong>$10–15</strong>.
  </p>

  <h2>3 · The agent that keeps the meter honest</h2>
  <p>
    Here is the part that turns "rent a GPU" from a foot-gun into a routine. A run is
    a <strong>persisted state machine</strong> of idempotent steps. It can be
    interrupted at any point and <em>resumed</em> — every step is safe to re-enter, so
    nothing is redone and no instance is double-launched. A <strong>budget guard</strong>
    checks spend on every poll and forces teardown before the bill can overrun. And
    teardown is the default exit. Drive the simulation:
  </p>

  <figure class="fig">
    <div class="sim">
      <ol class="ladder">
        @for (s of simView(); track s.name) {
          <li class="ladder__step" [class.is-done]="s.status === 'done'" [class.is-running]="s.status === 'running'" [class.is-skip]="s.status === 'skipped'" [class.is-paused]="s.status === 'paused'">
            <span class="ladder__mark">{{ mark(s.status) }}</span>
            <span class="ladder__name">{{ s.name }}</span>
            <span class="ladder__bar"><span class="ladder__fill" [style.width.%]="s.pct"></span></span>
          </li>
        }
      </ol>

      <div class="meter">
        <div class="meter__row">
          <span class="meter__k">spent</span>
          <span class="meter__bartrack"><span class="meter__barfill" [class.meter__barfill--hot]="tripped()" [style.width.%]="budgetPct()"></span></span>
          <span class="meter__v mono">\${{ spent().toFixed(2) }} / \${{ budget().toFixed(2) }}</span>
        </div>
        <div class="ctl">
          <span class="ctl__label">budget cap <span class="ctl__t">\${{ budget().toFixed(2) }}</span> — lower it to watch the guard trip</span>
          <input type="range" min="0.25" max="6" step="0.25" [value]="budget()" (input)="onBudget($event)" />
        </div>
        <span class="sim__status" [class.sim__status--hot]="tripped()" [class.sim__status--ok]="finished() && !tripped()">{{ statusText() }}</span>
      </div>
    </div>

    <div class="ctl__row ctl__row--btns">
      <button type="button" class="btn btn--primary" (click)="play()" [disabled]="running() || finished()">▶ run</button>
      <button type="button" class="btn" (click)="interrupt()" [disabled]="!running()">⚡ interrupt (SSH drops)</button>
      <button type="button" class="btn" (click)="resume()" [disabled]="running() || finished()">↻ resume</button>
      <button type="button" class="btn" (click)="reset()">reset</button>
    </div>

    <div class="logbox">
      @for (line of log(); track $index) { <div class="logbox__line">{{ line }}</div> }
    </div>
    <figcaption class="fig__cap">
      The real engine in <code>lambda_lab/engine.py</code> behaves exactly like this:
      interrupt mid-run and <strong>resume</strong> — completed steps are cached and
      skipped; drop the budget and the <strong>guard</strong> forces teardown so the
      meter stops. State is flushed to disk after every transition, so the instance id
      is recorded the instant it launches — no orphaned GPUs.
    </figcaption>
  </figure>

  <h2>4 · The orchestrator, real code</h2>
  <p>
    None of this is pseudocode — it's the <code>lambda_lab/</code> package in this repo,
    standard-library only (the Lambda API client is built on <code>urllib</code>; remote
    work rides your system <code>ssh</code>/<code>rsync</code>). The tabs are its load-bearing
    pieces, quoted verbatim.
  </p>

  <figure class="fig">
    <div class="tabs" role="tablist">
      @for (t of codeTabs; track t.id) {
        <button type="button" role="tab" class="tabs__tab" [class.tabs__tab--on]="tab() === t.id" [attr.aria-selected]="tab() === t.id" (click)="setTab(t.id)">
          <span class="tabs__lab">{{ t.label }}</span>
          <span class="tabs__sub">{{ t.sub }}</span>
        </button>
      }
    </div>
    <div class="tabs__panel">
      @if (tab() === 'engine') {
        <p class="tabs__prose"><strong>The resume-aware loop.</strong> Steps already marked <code>done</code> are logged and skipped, so re-running picks up exactly where it left off. A fatal failure on a step flagged <code>teardown_on_fail</code> runs teardown before bailing — you never strand a live instance.</p>
        <app-code-ref file="lambda_lab/engine.py" lang="python" [code]="snipEngine" [lines]="[204, 229]" caption="Engine.execute — the heart of resumability: skip cached steps, and guarantee teardown after a fatal step failure." />
      }
      @if (tab() === 'budget') {
        <p class="tabs__prose"><strong>The budget guard.</strong> Spend is just rate × elapsed hours. <code>over_budget()</code> is checked between steps and on every job poll; cross the line and the engine forces teardown. <code>budget_runs_out_in_min()</code> tells you the runway left at the current burn.</p>
        <app-code-ref file="lambda_lab/costs.py" lang="python" [code]="snipBudget" [lines]="[81, 94]" caption="Cost — the per-minute meter and the hard cap that makes a forgotten GPU nearly impossible." />
      }
      @if (tab() === 'launch') {
        <p class="tabs__prose"><strong>Idempotent launch.</strong> Before creating anything, it looks for an instance already tagged with this run id and reuses it. That single check is why a resumed run can never accidentally launch (and bill for) a second box.</p>
        <app-code-ref file="lambda_lab/steps.py" lang="python" [code]="snipLaunch" [lines]="[54, 62]" caption="steps.launch — reuse-by-run-id makes launch safe to call again after any interruption." />
      }
      @if (tab() === 'job') {
        <p class="tabs__prose"><strong>Long-horizon jobs.</strong> Training runs for hours, so it starts <em>detached</em> on the box and polls a sentinel file. On resume, if the job is still running it re-attaches instead of restarting; the poll callback updates spend and checks the budget every cycle.</p>
        <app-code-ref file="lambda_lab/steps.py" lang="python" [code]="snipJob" [lines]="[153, 181]" caption="steps.run_job — survives disconnects (detached + sentinel) and re-attaches on resume; the budget guard rides the poll." />
      }
      @if (tab() === 'detach') {
        <p class="tabs__prose"><strong>How a job outlives the SSH session.</strong> <code>setsid</code> detaches the process group, output tees to a log, and the exit code lands in a <code>.done</code> sentinel. A resumed run reads those files to tell "still running" from "finished(code)" — no live channel required.</p>
        <app-code-ref file="lambda_lab/ssh.py" lang="python" [code]="snipDetach" [lines]="[73, 81]" caption="ssh.start_detached — the trick that lets a 6-hour train survive your laptop going to sleep." />
      }
      @if (tab() === 'arm') {
        <p class="tabs__prose"><strong>ARM-aware bootstrap.</strong> The cheap 96 GB box is GH200 / aarch64. PyTorch publishes CUDA wheels for both x86_64 and aarch64 from the same index, so the install is identical; a couple of extras may build from source, which the installer treats as non-fatal.</p>
        <app-code-ref file="lambda_lab/bootstrap.sh" lang="bash" [code]="snipArm" [lines]="[47, 53]" caption="bootstrap.sh — one torch install path for both architectures; the GH200's only real cost is occasional source builds." />
      }
    </div>
    <figcaption class="fig__cap">Each tab is a real excerpt with its true line numbers — the chip in the header deep-links to the file on GitHub.</figcaption>
  </figure>

  <h2>5 · Run it</h2>
  <p>
    Two one-time setup steps (an API key and an SSH key), then a single command does
    the whole loop — launch, install, train, pull the LoRA back, terminate — under a
    hard dollar ceiling. <code>plan</code> and <code>costs</code> need no key, so you
    can look before you leap:
  </p>
  <app-code-ref file="lambda_lab/README.md" lang="bash" [code]="snipQuick" caption="The full path: plan is a dry run, start runs the pipeline, resume continues an interrupted run, teardown is the always-safe kill switch. Flags verified against lambda_lab/run.py." />

  <p>
    Want an agent (this one, even) to drive it? Every step is also a discrete CLI
    command, so it can run <code>plan</code>, then <code>start</code> with an explicit
    <code>--budget</code>, watch <code>status</code>, and <code>resume</code> across
    turns. The budget cap is the hard rail; auto-teardown is the safety net.
  </p>

  <p class="studio-cta-wrap">
    <a class="studio-cta" routerLink="/studio">▸ Open the live Image Studio</a>
    <span>— a browser front-end that talks to <em>your</em> SDXL box. Start it with
    <code>lambda_lab.run start serve-sdxl</code>, open the tunnel it prints, and generate
    right here. It runs nothing itself — your GPU does the work, with no prompt filter.</span>
  </p>

  <h2>6 · The GH200 question, honestly</h2>
  <div class="callout callout--warn">
    <span class="callout__tag tag">read before you pick the cheap one</span>
    <p>
      The GH200 (96 GB, \$2.29/hr) is the best dollars-per-GB on the menu and the only
      single box that swallows the largest video models without aggressive offloading.
      But it is <strong>ARM64</strong>: stock PyTorch + diffusers work fine, yet a few
      prebuilt kernels (flash-attention, the odd quantization wheel) may need a source
      build. If you want zero friction, the <strong>A100 40 GB (\$1.99)</strong> handles
      FLUX/SDXL LoRA and the <strong>H100 PCIe 80 GB (\$3.29)</strong> handles video — both
      x86. Reach for the GH200 when you specifically need the 96 GB.
    </p>
  </div>

  <h2>Same machine, rented by the minute</h2>
  <p>
    That's the whole arc of this site closed. The math from the diffusion chapters,
    the latent trick, cross-attention, guidance, spacetime patches — they're all
    running, unchanged, inside the open-weight checkpoints you just rented a GPU to
    fine-tune. The only thing the cloud adds is a meter, and a small agent to keep it
    honest. Build the toy to understand it; rent the H100 to ship it.
  </p>

  <div class="callout">
    <span class="callout__tag tag tag--accent">responsible use</span>
    <p>
      Self-hosting open weights moves every guardrail onto you. Respect each model's
      license (FLUX.1-dev is <em>non-commercial</em>; SDXL and Wan are more permissive —
      read them). Only train on data you have the rights to, never model a real person
      without consent, and don't generate illegal content. Control implies
      accountability.
    </p>
  </div>

  <div class="takeaways panel">
    <h3>Takeaways</h3>
    <ul>
      <li>Open-weight models + a by-the-minute GPU beat pay-per-call APIs above a modest volume — and you keep the weights.</li>
      <li>Sort by <strong>$/GB of VRAM</strong>, not sticker price: the GH200 (96 GB, \$2.29) is the value pick; the A100 40 GB (\$1.99) is the zero-friction x86 default.</li>
      <li>A LoRA — not a full fine-tune — is the economical way to teach a model a face or style: minutes on one GPU, dollars total.</li>
      <li>The only real risk is a forgotten instance. An <strong>idempotent, resumable, budget-capped</strong> orchestrator with auto-teardown removes it.</li>
    </ul>
  </div>

</app-chapter>
  `,
  styles: [`
    :host { display: block; }
    .lede { font-size: var(--step-1); color: var(--ink-1); }
    .lede em, p em { color: var(--ink-0); font-style: italic; }
    code { font-family: var(--font-mono); font-size: 0.86em; background: var(--bg-3); border: 1px solid var(--line); border-radius: 6px; padding: 0.05em 0.4em; color: #cdbcff; }

    .fig { margin: 1.8rem 0 2.2rem; padding: 1.1rem; border: 1px solid var(--line); border-radius: var(--radius); background: linear-gradient(180deg, rgba(255,255,255,0.025), rgba(255,255,255,0)), var(--bg-1); box-shadow: var(--shadow-1); }
    .fig__cap { margin: 0.85rem 0 0; color: var(--ink-2); font-size: 0.85rem; line-height: 1.55; }
    .fig__cap strong { color: var(--ink-1); }

    .ctl { display: flex; flex-direction: column; gap: 0.45rem; margin: 0.8rem 0; }
    .ctl__row { display: flex; flex-wrap: wrap; align-items: center; gap: 0.5rem; margin-bottom: 0.6rem; }
    .ctl__label { font-family: var(--font-mono); font-size: 0.72rem; letter-spacing: 0.04em; text-transform: uppercase; color: var(--ink-2); }
    .ctl__label--r { margin-left: auto; }
    .ctl__t { color: var(--plasma-b); font-weight: 600; text-transform: none; letter-spacing: 0; }
    input[type=range] { width: 100%; accent-color: var(--plasma-a); }
    .mono { font-family: var(--font-mono); }

    .btn { font-family: var(--font-display); font-size: 0.84rem; padding: 0.4em 0.9em; border-radius: var(--radius-sm); border: 1px solid var(--line-strong); background: var(--bg-2); color: var(--ink-1); cursor: pointer; transition: border-color .15s, color .15s, background .15s; }
    .btn:hover:not(:disabled) { color: #fff; border-color: rgba(124,92,255,0.6); }
    .btn:disabled { opacity: 0.4; cursor: not-allowed; }
    .btn--primary { background: var(--grad-plasma); color: #0a0a12; border-color: transparent; font-weight: 600; }
    .select { font-family: var(--font-mono); font-size: 0.8rem; background: var(--bg-3); color: var(--ink-0); border: 1px solid var(--line-strong); border-radius: 7px; padding: 0.3em 0.5em; }
    .studio-cta-wrap { display: flex; flex-wrap: wrap; align-items: baseline; gap: 0.5rem; padding: 1rem 1.2rem; border: 1px solid rgba(124,92,255,0.4); border-radius: var(--radius-sm); background: var(--accent-soft); }
    .studio-cta { font-family: var(--font-display); font-weight: 600; color: #cdbcff; text-decoration: none; white-space: nowrap; font-size: 1.02rem; }
    .studio-cta:hover { color: #fff; }
    .studio-cta-wrap span { color: var(--ink-2); font-size: 0.9rem; }

    /* table */
    .tbl-wrap { overflow-x: auto; border: 1px solid var(--line); border-radius: var(--radius-sm); }
    .tbl { width: 100%; border-collapse: collapse; font-size: 0.84rem; }
    .tbl th { text-align: left; font-family: var(--font-mono); font-size: 0.66rem; letter-spacing: 0.1em; text-transform: uppercase; color: var(--ink-3); padding: 0.6rem 0.8rem; border-bottom: 1px solid var(--line); }
    .tbl td { padding: 0.55rem 0.8rem; border-bottom: 1px solid rgba(255,255,255,0.04); color: var(--ink-1); cursor: pointer; }
    .tbl tbody tr:last-child td { border-bottom: 0; }
    .tbl tbody tr:hover td { background: rgba(255,255,255,0.025); }
    .tbl__row--on td { background: var(--accent-soft) !important; color: #fff; }
    .tbl__row--fit td { background: rgba(65,214,255,0.10) !important; }
    .tbl__row--dim td { opacity: 0.38; }
    .tbl__id { display: block; font-size: 0.66rem; color: var(--ink-3); }
    .tbl__mark { color: var(--plasma-a); font-weight: 700; }
    .tbl__cheapest { color: var(--plasma-b); font-size: 0.7rem; font-family: var(--font-mono); }
    .pill { font-family: var(--font-mono); font-size: 0.68rem; padding: 0.1em 0.5em; border-radius: 999px; border: 1px solid var(--line-strong); color: var(--ink-2); }
    .pill--arm { border-color: rgba(255,92,138,0.5); color: #ff9bb5; }

    .readout { display: flex; flex-wrap: wrap; gap: 0.7rem; margin-top: 1rem; }
    .readout__item { display: flex; flex-direction: column; gap: 0.15rem; padding: 0.6rem 0.9rem; border: 1px solid var(--line); border-radius: var(--radius-sm); background: var(--bg-2); min-width: 120px; }
    .readout__item--big { border-color: rgba(124,92,255,0.45); background: var(--accent-soft); }
    .readout__k { font-family: var(--font-mono); font-size: 0.62rem; letter-spacing: 0.1em; text-transform: uppercase; color: var(--ink-3); }
    .readout__v { font-family: var(--font-display); font-size: 1.05rem; font-weight: 600; color: var(--ink-0); }
    .readout__item--big .readout__v { font-size: 1.7rem; }
    .win { color: var(--plasma-b); }

    .chip { display: inline-flex; align-items: center; font-family: var(--font-mono); font-size: 0.74rem; padding: 0.3em 0.7em; border-radius: 999px; border: 1px solid var(--line); background: var(--bg-2); color: var(--ink-1); cursor: pointer; }
    .chip--task:hover { border-color: rgba(124,92,255,0.6); color: #fff; }
    .chip--on { border-color: var(--plasma-a); background: var(--accent-soft); color: #cdbcff; }

    /* callout */
    .callout { display: flex; gap: 1rem; align-items: flex-start; margin: 1.6rem 0; padding: 1rem 1.2rem; border-left: 2px solid var(--plasma-a); border-radius: var(--radius-sm); background: rgba(124,92,255,0.07); }
    .callout--warn { border-left-color: #ff5c8a; background: rgba(255,92,138,0.07); }
    .callout__tag { flex: none; margin-top: 0.15rem; }
    .callout p { margin: 0; }

    /* run simulator */
    .sim { display: grid; gap: 1rem; }
    @media (min-width: 720px) { .sim { grid-template-columns: 1.3fr 1fr; } }
    .ladder { list-style: none; margin: 0; padding: 0; display: grid; gap: 0.3rem; }
    .ladder__step { display: grid; grid-template-columns: 1.4rem 8.5rem 1fr; align-items: center; gap: 0.5rem; padding: 0.3rem 0.5rem; border-radius: 7px; border: 1px solid transparent; font-family: var(--font-mono); font-size: 0.78rem; color: var(--ink-2); }
    .ladder__step.is-done { color: var(--ink-0); }
    .ladder__step.is-running { border-color: rgba(124,92,255,0.5); background: var(--accent-soft); color: #fff; }
    .ladder__step.is-skip { opacity: 0.4; text-decoration: line-through; }
    .ladder__step.is-paused { border-color: rgba(255,200,80,0.4); }
    .ladder__mark { text-align: center; }
    .ladder__bar { height: 5px; border-radius: 3px; background: var(--bg-3); overflow: hidden; }
    .ladder__fill { display: block; height: 100%; background: var(--grad-plasma); transition: width .12s linear; }

    .meter { display: flex; flex-direction: column; gap: 0.7rem; }
    .meter__row { display: grid; grid-template-columns: auto 1fr auto; align-items: center; gap: 0.6rem; }
    .meter__k { font-family: var(--font-mono); font-size: 0.66rem; text-transform: uppercase; letter-spacing: 0.1em; color: var(--ink-3); }
    .meter__bartrack { height: 10px; border-radius: 6px; background: var(--bg-3); overflow: hidden; border: 1px solid var(--line); }
    .meter__barfill { display: block; height: 100%; background: var(--grad-plasma); transition: width .12s linear; }
    .meter__barfill--hot { background: linear-gradient(90deg, #ff5c8a, #ff8a5c); }
    .meter__v { font-size: 0.8rem; color: var(--ink-0); }
    .sim__status { font-family: var(--font-mono); font-size: 0.78rem; color: var(--ink-2); padding: 0.4rem 0.6rem; border: 1px dashed var(--line-strong); border-radius: 7px; text-align: center; }
    .sim__status--hot { color: #ff9bb5; border-color: rgba(255,92,138,0.5); border-style: solid; }
    .sim__status--ok { color: var(--plasma-b); border-color: rgba(65,214,255,0.5); border-style: solid; }

    .logbox { margin-top: 1rem; padding: 0.7rem 0.9rem; background: #0c0f17; border: 1px solid var(--line); border-radius: var(--radius-sm); font-family: var(--font-mono); font-size: 0.74rem; color: var(--ink-2); min-height: 4.5rem; max-height: 9rem; overflow-y: auto; }
    .logbox__line { white-space: pre-wrap; line-height: 1.5; }

    /* tabs (shared look with the build chapter) */
    .tabs { display: flex; flex-wrap: wrap; gap: 0.3rem; padding: 0 0 0.6rem; border-bottom: 1px solid var(--line); }
    .tabs__tab { display: grid; gap: 0.1rem; text-align: left; padding: 0.5rem 0.8rem; cursor: pointer; background: var(--bg-2); border: 1px solid var(--line); border-bottom: none; border-radius: var(--radius-sm) var(--radius-sm) 0 0; color: var(--ink-2); transition: color .15s, background .15s, border-color .15s; }
    .tabs__tab:hover { color: var(--ink-0); border-color: var(--line-strong); }
    .tabs__tab--on { background: var(--bg-1); color: var(--ink-0); border-color: rgba(124,92,255,0.5); box-shadow: inset 0 2px 0 0 var(--plasma-a); }
    .tabs__lab { font-family: var(--font-display); font-weight: 600; font-size: 0.84rem; }
    .tabs__sub { font-family: var(--font-mono); font-size: 0.64rem; color: var(--ink-3); }
    .tabs__panel { padding: 1.1rem 0 0.2rem; }
    .tabs__prose { margin: 0 0 0.4rem; font-size: 0.92rem; color: var(--ink-1); }
    .tabs__prose strong { color: var(--ink-0); }

    .takeaways { margin: 2.4rem 0 1rem; padding: 1.2rem 1.4rem; }
    .takeaways h3 { margin-top: 0; }
    .takeaways ul { margin: 0; padding-left: 1.1rem; display: grid; gap: 0.6rem; }
    .takeaways li { color: var(--ink-1); }
    .takeaways strong { color: var(--ink-0); }
  `],
})
export class Gpu {
  private readonly fb = inject(FirebaseService);
  private readonly destroyRef = inject(DestroyRef);

  // ---- the account's exact lineup (mirrors lambda_lab/costs.py PRICES) ----
  readonly instances: readonly Inst[] = [
    { id: 'gpu_1x_a10', label: '1× A10 (24 GB)', vram: 24, hr: 1.29, arch: 'x86_64' },
    { id: 'gpu_1x_a100_sxm4', label: '1× A100 (40 GB)', vram: 40, hr: 1.99, arch: 'x86_64' },
    { id: 'gpu_1x_gh200', label: '1× GH200 (96 GB)', vram: 96, hr: 2.29, arch: 'arm64' },
    { id: 'gpu_1x_h100_pcie', label: '1× H100 (80 GB PCIe)', vram: 80, hr: 3.29, arch: 'x86_64' },
    { id: 'gpu_1x_h100_sxm5', label: '1× H100 (80 GB SXM5)', vram: 80, hr: 4.29, arch: 'x86_64' },
    { id: 'gpu_8x_v100', label: '8× V100 (16 GB)', vram: 16, hr: 6.32, arch: 'x86_64' },
    { id: 'gpu_2x_h100_sxm5', label: '2× H100 (80 GB)', vram: 80, hr: 8.38, arch: 'x86_64' },
  ];
  // shown sorted by best $/GB (the GH200 reveal)
  readonly instancesSorted = computed(() => [...this.instances].sort((a, b) => a.hr / a.vram - b.hr / b.vram));

  readonly tasks: readonly Task[] = [
    { key: 'sdxl-infer', label: 'SDXL · generate', vram: 12, trick: 'SDXL inference fits in ~12 GB — the A10 is plenty and cheapest at $1.29/hr.' },
    { key: 'sdxl-lora', label: 'SDXL · LoRA', vram: 16, trick: 'SDXL LoRA with 8-bit Adam + gradient checkpointing trains under 16 GB; the A10 handles it.' },
    { key: 'flux-infer', label: 'FLUX · generate', vram: 24, trick: 'FLUX in FP8/quantized + model offload runs in ~24 GB on the A10; full-precision wants 40 GB+.' },
    { key: 'flux-lora', label: 'FLUX · LoRA', vram: 24, trick: 'A quantized base + cached latents fits FLUX LoRA in ~24 GB; the A100 40 GB ($1.99) is the comfortable, fast pick.' },
    { key: 'video-infer', label: 'Wan/LTX · video', vram: 24, trick: 'Wan-1.3B / LTX run in ~24 GB; the 14B models want 48–80 GB or heavy block-swap offload.' },
    { key: 'video-lora', label: 'Video · LoRA', vram: 48, trick: 'Video LoRA with block-swap fits 80 GB cleanly — the H100 PCIe ($3.29) or the 96 GB GH200 ($2.29) are ideal.' },
  ];

  // ---- FIG 1: cost explorer ----
  readonly costMode = signal<'pick' | 'task'>('pick');
  readonly selected = signal<Inst>(this.instances[2]); // GH200 by default — the reveal
  readonly hours = signal(1);
  readonly task = signal<Task>(this.tasks[3]); // flux-lora

  readonly estimate = computed(() => (this.selected().hr * this.hours()).toFixed(2));
  readonly hoursLabel = computed(() => {
    const h = this.hours();
    return h < 1 ? `${Math.round(h * 60)} min` : `${h.toFixed(1)} hr`;
  });
  readonly fit = computed(() => {
    const need = this.task().vram;
    return [...this.instances].sort((a, b) => a.hr - b.hr).find((i) => i.vram >= need) ?? null;
  });
  readonly fitId = computed(() => this.fit()?.id ?? '');

  setCostMode(m: 'pick' | 'task') { this.costMode.set(m); this.fb.event('interact', { section: 'gpu', control: 'cost_mode', value: m }); }
  pick(i: Inst) { if (this.costMode() === 'pick') { this.selected.set(i); this.fb.event('interact', { section: 'gpu', control: 'pick_instance', value: i.id }); } }
  pickTask(t: Task) { this.task.set(t); this.fb.event('interact', { section: 'gpu', control: 'pick_task', value: t.key }); }
  onHours(e: Event) { this.hours.set(Number((e.target as HTMLInputElement).value)); }

  // ---- FIG 2: $/output ----
  readonly outMode = signal<'image' | 'video'>('image');
  readonly outInst = signal<Inst>(this.instances[1]); // A100
  readonly imgPerMin = signal(30);
  readonly apiImg = signal(0.03);
  readonly secPerSec = signal(30);

  readonly selfPer1k = computed(() => {
    const perImg = this.outInst().hr / (this.imgPerMin() * 60);
    return (perImg * 1000).toFixed(2);
  });
  readonly apiPer1k = computed(() => (this.apiImg() * 1000).toFixed(2));
  readonly selfWins = computed(() => Number(this.selfPer1k()) < Number(this.apiPer1k()));
  /** One-time spin-up (boot + weight download) you pay for before the first image. */
  private readonly overheadHr = 0.25;
  readonly breakevenImgs = computed(() => {
    const selfPerImg = this.outInst().hr / (this.imgPerMin() * 60);
    const saving = this.apiImg() - selfPerImg;
    if (saving <= 0) return '—';
    const overheadCost = this.outInst().hr * this.overheadHr;
    return Math.ceil(overheadCost / saving).toLocaleString();
  });
  readonly videoPerMin = computed(() => ((this.outInst().hr / 3600) * this.secPerSec() * 60).toFixed(2));
  readonly videoPerClip = computed(() => ((this.outInst().hr / 3600) * this.secPerSec() * 5).toFixed(3));

  // concrete end-to-end cost expectations (verified $/hr; setup overhead included)
  readonly jobCosts = [
    { job: 'SDXL LoRA — 1.5k steps', inst: 'A10 · $1.29/hr', time: '~35 min', total: '~$0.75' },
    { job: 'FLUX LoRA — 2k steps', inst: 'A100 40 GB · $1.99/hr', time: '~50 min', total: '~$1.65' },
    { job: '1,000 images — FLUX-schnell', inst: 'A100 40 GB · $1.99/hr', time: '~30 min', total: '~$1.00' },
    { job: 'one 5-second clip — Wan-1.3B', inst: 'H100 PCIe · $3.29/hr', time: '~2 min', total: '~$0.10' },
    { job: 'video LoRA — Wan-1.3B', inst: 'H100 PCIe · $3.29/hr', time: '~3 hr', total: '~$10' },
    { job: 'ComfyUI, interactive', inst: 'A10 · $1.29/hr', time: 'per hour, up', total: '$1.29/hr' },
  ];

  setOutMode(m: 'image' | 'video') { this.outMode.set(m); }
  onOutInst(e: Event) { const id = (e.target as HTMLSelectElement).value; const i = this.instances.find((x) => x.id === id); if (i) this.outInst.set(i); }
  onImgRate(e: Event) { this.imgPerMin.set(Number((e.target as HTMLInputElement).value)); }
  onApiImg(e: Event) { this.apiImg.set(Number((e.target as HTMLInputElement).value)); }
  onSecPerSec(e: Event) { this.secPerSec.set(Number((e.target as HTMLInputElement).value)); }

  // ---- FIG 3: agentic run simulator ----
  private readonly simSteps: readonly SimStep[] = [
    { name: 'ensure_ssh_key', dur: 6 },
    { name: 'ensure_filesystem', dur: 6 },
    { name: 'launch', dur: 45 },
    { name: 'wait_active', dur: 70 },   // meter starts at the END of this step
    { name: 'bootstrap', dur: 520 },
    { name: 'sync_up', dur: 40 },
    { name: 'run_job', dur: 1400 },
    { name: 'sync_down', dur: 40 },
    { name: 'teardown', dur: 12 },
  ];
  private readonly cumEnd: number[] = (() => {
    const out: number[] = []; let s = 0;
    for (const st of this.simSteps) { s += st.dur; out.push(s); }
    return out;
  })();
  private readonly tActive = (() => {
    // billing starts when the instance is active = end of wait_active
    let s = 0; for (const st of this.simSteps) { s += st.dur; if (st.name === 'wait_active') return s; } return s;
  })();
  private readonly tJobEnd = this.cumEnd[this.simSteps.length - 2]; // end of sync_down (before teardown)
  private readonly speed = 26; // sim-seconds advanced per ~110ms tick

  readonly simSec = signal(0);
  readonly running = signal(false);
  readonly tripped = signal(false);
  readonly tornDown = signal(false);
  readonly interrupted = signal(false);
  readonly budget = signal(4);
  readonly rate = computed(() => this.selected().hr); // tie to FIG-1 instance choice
  readonly log = signal<string[]>(['idle — press ▶ run']);
  private loggedDone = 0;

  readonly spent = computed(() => {
    const billed = Math.max(0, this.simSec() - this.tActive);
    return Math.round((this.rate() / 3600) * billed * 100) / 100;
  });
  readonly budgetPct = computed(() => Math.min(100, (this.spent() / this.budget()) * 100));
  readonly finished = computed(() => this.tornDown());

  readonly simView = computed(() => {
    const t = this.simSec();
    const tripped = this.tripped();
    const torn = this.tornDown();
    let start = 0;
    return this.simSteps.map((st, i) => {
      const end = this.cumEnd[i];
      let status: 'pending' | 'running' | 'done' | 'skipped' | 'paused';
      let pct = 0;
      if (st.name === 'teardown') {
        status = torn ? 'done' : 'pending';
        pct = torn ? 100 : 0;
      } else if (tripped && start >= this.trippedAt) {
        status = 'skipped';
      } else if (t >= end) {
        status = 'done'; pct = 100;
      } else if (t > start) {
        status = this.running() ? 'running' : 'paused';
        pct = Math.round(((t - start) / st.dur) * 100);
      } else {
        status = 'pending';
      }
      start = end;
      return { name: st.name, status, pct };
    });
  });

  readonly statusText = computed(() => {
    if (this.tripped()) return '⚠ budget hit → instance terminated, meter stopped';
    if (this.tornDown()) return '✓ complete — instance terminated, LoRA downloaded';
    if (this.interrupted()) return '↯ interrupted — state saved to disk, safe to resume';
    if (this.running()) return '▶ running …';
    return 'ready';
  });

  private trippedAt = Infinity;
  private timer: ReturnType<typeof setInterval> | null = null;

  constructor() {
    afterNextRender(() => {
      this.timer = setInterval(() => this.tick(), 110);
    });
    this.destroyRef.onDestroy(() => { if (this.timer) clearInterval(this.timer); });
  }

  mark(s: string): string {
    return ({ done: '✓', running: '▶', skipped: '—', paused: '⏸', pending: '·' } as Record<string, string>)[s] ?? '·';
  }

  private pushLog(line: string) { this.log.update((l) => [...l, line].slice(-7)); }

  private tick() {
    if (!this.running() || this.tornDown() || this.tripped()) return;
    const next = this.simSec() + this.speed;
    this.simSec.set(next);

    // log newly-completed steps
    let doneCount = 0;
    for (let i = 0; i < this.simSteps.length - 1; i++) if (next >= this.cumEnd[i]) doneCount++;
    while (this.loggedDone < doneCount) {
      this.pushLog(`✓ ${this.simSteps[this.loggedDone].name}`);
      this.loggedDone++;
    }

    // budget guard — checked every tick, like ctx.check_budget() on every poll
    if (this.spent() >= this.budget()) {
      this.tripped.set(true);
      this.trippedAt = this.activeStepStart();
      this.running.set(false);
      this.tornDown.set(true);
      this.pushLog(`✗ budget $${this.budget().toFixed(2)} reached (spent $${this.spent().toFixed(2)})`);
      this.pushLog('↳ forcing teardown …');
      this.pushLog('✓ teardown — meter stopped');
      this.fb.event('interact', { section: 'gpu', control: 'sim', value: 'budget_trip' });
      return;
    }

    // natural completion -> teardown
    if (next >= this.tJobEnd) {
      this.simSec.set(this.tJobEnd);
      this.running.set(false);
      this.tornDown.set(true);
      this.pushLog('✓ teardown — meter stopped');
      this.pushLog(`done — total spend $${this.spent().toFixed(2)}`);
      this.fb.event('interact', { section: 'gpu', control: 'sim', value: 'complete' });
    }
  }

  /** sim-seconds at which the currently-active step began (for skip marking). */
  private activeStepStart(): number {
    const t = this.simSec(); let start = 0;
    for (let i = 0; i < this.simSteps.length; i++) {
      if (t < this.cumEnd[i]) return start;
      start = this.cumEnd[i];
    }
    return start;
  }

  play() {
    if (this.finished()) return;
    this.interrupted.set(false);
    this.running.set(true);
    if (this.simSec() === 0) this.pushLog(`▶ run — ${this.selected().label} @ $${this.rate().toFixed(2)}/hr, budget $${this.budget().toFixed(2)}`);
    this.fb.event('interact', { section: 'gpu', control: 'sim', value: 'play' });
  }
  interrupt() {
    if (!this.running()) return;
    this.running.set(false);
    this.interrupted.set(true);
    this.pushLog('↯ connection dropped — state flushed to .lambda_lab/runs/');
    this.fb.event('interact', { section: 'gpu', control: 'sim', value: 'interrupt' });
  }
  resume() {
    if (this.finished() || this.running()) return;
    const done = this.simView().filter((s) => s.status === 'done').length;
    this.pushLog(`↻ resume — ${done} step${done === 1 ? '' : 's'} cached, continuing`);
    this.interrupted.set(false);
    this.running.set(true);
    this.fb.event('interact', { section: 'gpu', control: 'sim', value: 'resume' });
  }
  reset() {
    this.running.set(false); this.tripped.set(false); this.tornDown.set(false); this.interrupted.set(false);
    this.simSec.set(0); this.trippedAt = Infinity; this.loggedDone = 0;
    this.log.set(['idle — press ▶ run']);
  }
  onBudget(e: Event) { this.budget.set(Number((e.target as HTMLInputElement).value)); }

  // ---- FIG 4: code reader ----
  readonly codeTabs = [
    { id: 'engine', label: 'resume loop', sub: 'engine.py · execute' },
    { id: 'budget', label: 'budget guard', sub: 'costs.py · Cost' },
    { id: 'launch', label: 'idempotent launch', sub: 'steps.py · launch' },
    { id: 'job', label: 'long-horizon job', sub: 'steps.py · run_job' },
    { id: 'detach', label: 'survive disconnect', sub: 'ssh.py · start_detached' },
    { id: 'arm', label: 'ARM bootstrap', sub: 'bootstrap.sh' },
  ] as const;
  readonly tab = signal<string>('engine');
  setTab(id: string) { this.tab.set(id); this.fb.event('interact', { section: 'gpu', control: 'code_tab', value: id }); }

  // ===== verbatim excerpts (line-accurate against the repo) =====
  readonly snipEngine = `    def execute(self, steps: list[Step]) -> RunState:
        st = self.state
        st.status = "running"
        st.save()
        ctx = Context(self, st)
        teardown_step = next((s for s in steps if s.name == "teardown"), None)

        for step in steps:
            rec = st.steps.setdefault(step.name, StepRecord(step.name))
            if rec.status == "done":
                st.log(f"✓ {step.name} (cached)")
                continue

            ok = self._run_step(ctx, step, rec)
            if not ok:
                st.status = "failed"
                st.save()
                if step.teardown_on_fail and teardown_step and step.name != "teardown":
                    st.log("running teardown after fatal failure …")
                    self._run_step(ctx, teardown_step, st.steps.setdefault("teardown", StepRecord("teardown")))
                return st

        st.status = "done"
        st.save()
        st.log(f"run {st.run_id} complete — total spend \${ctx.cost.spent_usd()}")
        return st`;

  readonly snipBudget = `    def spent_usd(self) -> float:
        return round(self.fixed_usd + self.usd_hr * self.elapsed_hr(), 4)

    def remaining_usd(self) -> float:
        return round(self.budget_usd - self.spent_usd(), 4)

    def over_budget(self) -> bool:
        return self.spent_usd() >= self.budget_usd

    def budget_runs_out_in_min(self) -> float:
        """Minutes of runway left at the current burn rate."""
        if self.usd_hr <= 0:
            return float("inf")
        return max(0.0, self.remaining_usd() / self.usd_hr * 60.0)`;

  readonly snipLaunch = `def launch(ctx) -> dict:
    """Launch the instance — but reuse one already tagged with this run id, so a
    resumed run never double-spends. Records the instance id immediately."""
    run_id = ctx.state.run_id
    existing = ctx.api.find_instance_by_name(run_id)
    if existing:
        ctx.set(instance_id=existing["id"], ip=existing.get("ip"))
        ctx.log(f"reusing instance {existing['id']} (status={existing.get('status')})")
        return {"instance_id": existing["id"], "reused": True}`;

  readonly snipJob = `def run_job(ctx) -> dict:
    """Run the main long-horizon job detached; poll with the budget guard active.
    On resume this re-attaches to a still-running job via its sentinel files."""
    r = ctx.remote()
    command = ctx.params["job_command"]
    st = r.job_status(tag="job")
    if st["state"] == "done":
        ctx.log(f"job already finished (exit {st['code']}) — not restarting")
        if st["code"] != 0:
            raise RuntimeError(f"prior job exited {st['code']}; inspect …job.log")
        return {"reattached": True, "code": st["code"]}
    if st["state"] != "running":
        ctx.log("starting job (detached, survives disconnects) …")
        r.start_detached(command, tag="job")
    else:
        ctx.log("re-attaching to job already running on the instance …")

    def on_poll(status, log_tail):
        cost = ctx.cost
        ctx.save_cost(cost)
        ctx.log(f"  job … \${cost.spent_usd()}/\${cost.budget_usd}")
        ctx.check_budget()

    code = r.wait_for_job(tag="job", poll_s=float(ctx.params.get("poll_s", 30)), on_poll=on_poll)
    if code != 0:
        raise RuntimeError(f"job exited {code}")
    return {"code": 0}`;

  readonly snipDetach = `        d = job_dir
        script = (
            f"mkdir -p {d} && "
            f"rm -f {d}/{tag}.done && "
            f"setsid bash -lc {shlex.quote(command + f'; echo $? > {d}/{tag}.done')} "
            f"> {d}/{tag}.log 2>&1 < /dev/null & "
            f"echo $! > {d}/{tag}.pid"
        )
        self.run(script, stream=False)`;

  readonly snipArm = `# CUDA 12.4 wheels exist for x86_64 AND aarch64 (sbsa). Lambda boxes ship a
# recent driver; if torch is already importable with CUDA we skip the reinstall.
if ! python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "installing torch (cu124, $ARCH) …"
  pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu124
fi
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"`;

  readonly snipQuick = `# 1. credentials (one time)
export LAMBDA_API_KEY=...              # cloud.lambda.ai/api-keys
ssh-keygen -t ed25519                  # if you don't have ~/.ssh/id_ed25519

# 2. look before you leap — no API key needed
python -m lambda_lab.run costs
python -m lambda_lab.run plan train-flux-lora

# 3. put 15-40 images + .txt captions in ./dataset, then go (with an $8 ceiling):
python -m lambda_lab.run start train-flux-lora --budget 8 --instance-type gpu_1x_a100_sxm4

# interrupted? continue exactly where it stopped:
python -m lambda_lab.run resume flux-1a2b
# always-safe manual kill:
python -m lambda_lab.run teardown flux-1a2b`;
}
