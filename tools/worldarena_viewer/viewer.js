'use strict';
const data = JSON.parse(document.querySelector('#viewerData').textContent);
// The generator provides one canonical order for options, videos and parameter links.
const { runs, samples, prompts, upsampledPrompts } = data;
const familyLabels = { baseline: '发布版基线', native: '原生 VLM', v1: '自研 VLM v1', v2: '自研 VLM v2' };
const selectionKey = 'worldarena-eval25-checkpoints-reasoner-baseline-v2';
const baselineIds = runs.filter(run => run.family === 'baseline').map(run => run.id);
let selected = new Set(baselineIds);
try {
  const saved = JSON.parse(localStorage.getItem(selectionKey));
  if (Array.isArray(saved)) selected = new Set(saved.filter(id => runs.some(run => run.id === id)));
} catch (_) { /* Storage may be unavailable when opened from disk. */ }
const select = document.querySelector('#sample');
for (const sample of samples) select.add(new Option(sample.name, sample.name));
const views = new Map();
const options = document.querySelector('#checkpointOptions');
const search = document.querySelector('#search');
const familyFilter = document.querySelector('#familyFilter');
const onlySelected = document.querySelector('#onlySelected');

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}
function badge(text, color) { return element('span', `tag ${color}`, text); }
function stepBlock(run) {
  const block = element('span', 'step-block');
  block.append(element('span', 'step-number', run.step === null ? 'Baseline' : run.step.toLocaleString('en-US')),
    element('span', 'step-unit', run.step === null ? '基线对照' : '训练 step'));
  return block;
}
function titleTags(run, tag = 'div') {
  const title = element(tag, 'option-title');
  title.append(badge(familyLabels[run.family], run.family));
  return title;
}
function configTags(run) {
  const row = element('div', 'option-tags');
  row.append(run.prompt === 'reasoner' ? badge('Reasoner 扩写', 'reasoner') : badge('原 Prompt', 'original'));
  for (const [text, color] of run.tags) row.append(badge(text, color));
  return row;
}
function link(text, path) {
  const node = element('a', '', text);
  node.href = path;
  node.target = '_blank';
  node.rel = 'noopener';
  return node;
}
for (const run of runs) {
  const label = element('label', 'checkpoint-option');
  label.title = run.title;
  label.dataset.id = run.id;
  label.dataset.step = run.step ?? '';
  const checkbox = element('input');
  checkbox.type = 'checkbox';
  checkbox.value = run.id;
  checkbox.setAttribute('aria-label', run.title);
  checkbox.addEventListener('change', () => {
    if (checkbox.checked) selected.add(run.id); else selected.delete(run.id);
    applySelection();
  });
  const content = element('div', 'option-content');
  content.append(titleTags(run), configTags(run));
  label.append(checkbox, stepBlock(run), content);
  document.querySelector(run.step === null ? '#baselineOptions' : '#trainingOptions').append(label);

  const card = element('article', 'card');
  card.dataset.id = run.id;
  const heading = element('div', 'card-heading');
  const headingContent = element('div');
  headingContent.append(titleTags(run, 'h2'), configTags(run));
  heading.append(stepBlock(run), headingContent);
  const video = element('video');
  video.id = run.id;
  video.controls = true;
  video.loop = true;
  video.preload = 'metadata';
  video.playsInline = true;
  video.setAttribute('aria-label', run.title);
  const error = element('p', 'media-error', '视频未加载：请确认已解压此实验的压缩包，且当前样本存在。');
  error.hidden = true;
  video.addEventListener('error', () => { if (selected.has(run.id) && video.hasAttribute('src')) error.hidden = false; });
  video.addEventListener('loadedmetadata', () => { error.hidden = true; });
  const footer = element('div', 'card-footer');
  footer.append(element('span', 'muted', run.step === null ? '发布版模型' : 'T2V SFT · online 权重'));
  const args = link('运行参数 ↗', '#');
  footer.append(args);
  if (run.tags.some(([, color]) => color === 'fp8')) footer.append(link('性能数据 ↗', `${run.directory}/metrics.json`));
  const details = element('details', 'run-detail');
  details.append(element('summary', '', '完整实验名称与目录'), element('p', '', run.title), element('p', '', run.directory));
  card.append(heading, video, error, footer, details);
  document.querySelector('#videoGrid').append(card);
  const stepTitle = run.step === null ? 'Baseline' : run.step.toLocaleString('en-US');
  const metaLink = link(`${stepTitle} · ${run.title}`, '#');
  document.querySelector('#meta').append(metaLink);
  const searchText = `${run.title} ${run.directory} ${stepTitle} ${familyLabels[run.family]} ${run.tags.map(t => t[0]).join(' ')} ${run.prompt === 'reasoner' ? 'Reasoner 扩写' : '原 Prompt'}`.toLowerCase();
  views.set(run.id, { label, checkbox, card, video, error, args, metaLink, searchText });
}
function applyFilters() {
  const terms = search.value.trim().toLowerCase().split(/\s+/).filter(Boolean);
  let baselineCount = 0;
  let trainingCount = 0;
  for (const run of runs) {
    const view = views.get(run.id);
    const visible = (familyFilter.value === 'all' || familyFilter.value === run.family)
      && (!onlySelected.checked || selected.has(run.id)) && terms.every(term => view.searchText.includes(term));
    view.label.hidden = !visible;
    if (visible) { if (run.step === null) baselineCount++; else trainingCount++; }
  }
  document.querySelector('#baselineSection').hidden = baselineCount === 0;
  document.querySelector('#trainingSection').hidden = trainingCount === 0;
  document.querySelector('#filterEmpty').hidden = baselineCount + trainingCount !== 0;
  document.querySelector('#filterCount').textContent = `匹配 ${baselineCount + trainingCount} / ${runs.length} 组`;
}
document.querySelector('#expandOptions').addEventListener('click', event => {
  const expanded = options.classList.toggle('is-expanded');
  event.currentTarget.textContent = expanded ? '收起列表' : '展开全部';
  event.currentTarget.setAttribute('aria-expanded', String(expanded));
});
function updateScrollHint() {
  document.querySelector('#scrollHint').hidden = options.scrollHeight <= options.clientHeight;
}
new ResizeObserver(updateScrollHint).observe(options);
new ResizeObserver(updateScrollHint).observe(document.querySelector('#trainingOptions'));
for (const control of [search, familyFilter, onlySelected]) control.addEventListener('input', applyFilters);
document.querySelector('#resetFilters').addEventListener('click', () => {
  search.value = ''; familyFilter.value = 'all'; onlySelected.checked = false; applyFilters();
});
function updateVideos() {
  for (const run of runs) {
    const view = views.get(run.id);
    const active = selected.has(run.id);
    view.card.hidden = !active;
    view.metaLink.hidden = !active;
    const directory = `${run.directory}/${select.value}`;
    view.args.href = view.metaLink.href = `${directory}/sample_args.json`;
    if (active) {
      const src = `${directory}/vision.mp4`;
      if (view.video.getAttribute('src') !== src) {
        view.error.hidden = true;
        view.video.src = src;
      }
    } else if (view.video.hasAttribute('src')) {
      view.video.pause(); view.video.removeAttribute('src'); view.video.load(); view.error.hidden = true;
    }
  }
}
function applySelection() {
  for (const [id, view] of views) {
    view.checkbox.checked = selected.has(id);
    view.label.classList.toggle('is-selected', selected.has(id));
  }
  document.querySelector('#selectedCount').textContent = `${selected.size} / ${runs.length} 已选`;
  document.querySelector('#videoCount').textContent = `${selected.size} 组`;
  document.querySelector('#emptySelection').hidden = selected.size !== 0;
  document.querySelector('#sync').disabled = document.querySelector('#pause').disabled = selected.size === 0;
  try { localStorage.setItem(selectionKey, JSON.stringify([...selected])); } catch (_) {}
  applyFilters();
  updateVideos();
}
function baselineSelection() { selected = new Set(baselineIds); applySelection(); }
document.querySelector('#selectLatest').addEventListener('click', baselineSelection);
document.querySelector('#restoreBaseline').addEventListener('click', baselineSelection);
document.querySelector('#selectAll').addEventListener('click', () => { selected = new Set(runs.map(run => run.id)); applySelection(); });
document.querySelector('#selectNone').addEventListener('click', () => { selected.clear(); applySelection(); });
function loadSample() {
  const sample = samples[select.selectedIndex];
  document.querySelector('#sampleCount').textContent = `${select.selectedIndex + 1} / ${samples.length}`;
  document.querySelector('#summary').textContent = `${sample.modelMode} · ${sample.numFrames} frames · ${sample.fps} FPS · seed ${sample.seed}`;
  const prompt = document.querySelector('#promptText');
  prompt.textContent = prompt.title = prompts[sample.name];
  const upsampled = upsampledPrompts[sample.name];
  try { document.querySelector('#upsampledPromptText').textContent = JSON.stringify(JSON.parse(upsampled), null, 2); }
  catch (_) { document.querySelector('#upsampledPromptText').textContent = upsampled; }
  updateVideos();
}
document.querySelector('#promptToggle').addEventListener('click', event => {
  const expanded = document.querySelector('#promptBar').classList.toggle('expanded');
  event.currentTarget.textContent = expanded ? '收起' : '展开';
  event.currentTarget.setAttribute('aria-expanded', String(expanded));
});
select.addEventListener('change', loadSample);
for (const [id, offset] of [['prev', -1], ['next', 1]]) document.querySelector(`#${id}`).addEventListener('click', () => {
  select.selectedIndex = (select.selectedIndex + offset + samples.length) % samples.length;
  loadSample();
});
function activeVideos() { return [...views].filter(([id]) => selected.has(id)).map(([, view]) => view.video); }
document.querySelector('#sync').addEventListener('click', () => {
  const videos = activeVideos();
  for (const video of videos) video.currentTime = 0;
  Promise.allSettled(videos.map(video => video.play()));
});
document.querySelector('#pause').addEventListener('click', () => { for (const video of activeVideos()) video.pause(); });
// Distinguish user seeks from seeks propagated to peers, avoiding feedback loops.
const pendingSeeks = new WeakMap();
for (const [id, { video }] of views) video.addEventListener('seeked', () => {
  const target = pendingSeeks.get(video);
  pendingSeeks.delete(video);
  if (target !== undefined && Math.abs(video.currentTime - target) < .15) return;
  if (!selected.has(id) || video.readyState === 0) return;
  for (const peer of activeVideos()) {
    if (peer !== video && peer.readyState > 0 && Math.abs(peer.currentTime - video.currentTime) > .15) {
      const time = Math.min(video.currentTime, Number.isFinite(peer.duration) ? peer.duration : video.currentTime);
      pendingSeeks.set(peer, time);
      peer.currentTime = time;
    }
  }
});
applySelection();
loadSample();
