// The page of recorded graphs. It reads the graphs from the JSON in the page,
// keeps which groups are folded, asks ELK where everything goes, and draws SVG.
// Nothing here changes the data: folding only changes what is drawn.
(() => {
  'use strict';

  const DATA = JSON.parse(document.getElementById('data').textContent);
  const SVG = 'http://www.w3.org/2000/svg';
  const $ = id => document.getElementById(id);
  const canvas = $('canvas');
  const viewport = $('viewport');
  const layers = { boxes: $('boxes'), edges: $('edges'), nodes: $('nodes') };

  // Sizes, in pixels of the drawing before zoom.
  const FONT = '12.5px ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif';
  const TITLE_FONT = '600 ' + FONT;
  const CODE_FONT = '600 11.5px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace';   // A line of code as a title.
  const NODE_HEIGHT = 42;
  const FOLDED_HEIGHT = 46;
  const PAD = 12;
  const MIN_WIDTH = 70;
  const BAR_HEIGHT = 26;         // The title bar of an open group.
  const STACK = 3;               // Offset of the cards drawn behind a folded group.
  const LONGEST_TITLE = 48;      // Characters of a title kept on a folded group.
  const ZOOM_LIMITS = [0.03, 4];
  const DRAG_THRESHOLD = 4;      // A press that moves less than this is a click.
  const READABLE = 0.75;         // A graph does not open smaller than this unless its width needs it.
  const SHOWN_INPUTS = 4;        // A folded group with more inputs than this shows them as one pill.

  const CATEGORIES = [
    ['elementwise', 'element by element'], ['matmul', 'matrix product'], ['reduction', 'reduction'],
    ['view', 'view'], ['write', 'write into an array'], ['input', 'input'], ['atonce', 'made at once'], ['earlier', 'from an earlier graph'],
  ];
  const TRIGGER_TEXT = {
    'gather': call => `computed by ${call}`,
    'at once': call => `computed because ${call} was made at once`,
    'write': call => `computed before the write ${call}`,
    'needed': call => `computed without gather(): ${call} needed values`,
    'limit': call => `computed at ${call}`,
    'pending': call => `not computed: ${call}`,
  };
  const ROOT_OPTIONS = {
    'elk.algorithm': 'layered',
    'elk.direction': 'DOWN',
    'elk.hierarchyHandling': 'INCLUDE_CHILDREN',
    'elk.edgeRouting': 'ORTHOGONAL',
    'elk.layered.spacing.nodeNodeBetweenLayers': '34',
    'elk.spacing.nodeNode': '20',
    'elk.spacing.edgeNode': '14',
    'elk.spacing.edgeEdge': '9',
    'elk.layered.spacing.edgeNodeBetweenLayers': '14',
    'elk.layered.spacing.edgeEdgeBetweenLayers': '9',
    'elk.layered.considerModelOrder.strategy': 'NODES_AND_EDGES',
    'elk.padding': '[top=20,left=20,bottom=20,right=20]',
  };
  const BOX_PADDING = `[top=${BAR_HEIGHT + 12},left=14,bottom=14,right=14]`;

  const state = {
    index: 0,            // Which graph is shown.
    mode: 'code',        // 'code' for the program's lines, 'graph' for the drawing.
    file: null,          // The file the code view shows.
    flash: null,         // A line the code view was asked to point at.
    folds: new Map(),    // Graph index -> the set of folded group ids.
    selected: null,      // Id of the selected node, or null.
    query: '',           // The search text, lower case.
    matches: [],         // Ids of the nodes that match it.
    cursor: -1,          // Which match Enter last went to.
    model: null,         // The graph as the script needs it; see modelOf.
    view: null,          // What is drawn now; see visible.
    place: new Map(),    // Element id -> {x, y, w, h} in the drawing.
    routes: new Map(),   // Edge id -> the points its line goes through.
    size: { w: 1, h: 1 },
    pan: { x: 0, y: 0, k: 1 },
    fitted: null,        // 'whole' or 'top' while the drawing sits where a fit put it: a resize then fits again.
    draws: 0,            // Counts layouts, so a slow one that was overtaken is dropped.
  };
  const models = new Map();
  const measurer = document.createElement('canvas').getContext('2d');
  const elk = window.ELK ? new window.ELK() : null;

  // ---- Small helpers.

  function el(name, attributes, text) {
    const node = document.createElementNS(SVG, name);
    for (const [key, value] of Object.entries(attributes || {})) node.setAttribute(key, value);
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function html(name, className, text) {
    const node = document.createElement(name);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function push(map, key, value) {
    if (!map.has(key)) map.set(key, []);
    map.get(key).push(value);
  }

  function textWidth(text, font) {
    measurer.font = font;
    return measurer.measureText(text).width;
  }

  function cut(text, length) {
    return text.length <= length ? text : text.slice(0, length - 1) + '…';
  }

  function clamp(value, low, high) {
    return Math.min(high, Math.max(low, value));
  }

  function shapeText(shape) {
    return `(${shape.join(', ')}${shape.length === 1 ? ',' : ''})`;
  }

  function bytesText(bytes) {
    const units = ['bytes', 'kB', 'MB', 'GB', 'TB'];
    let value = bytes, unit = 0;
    while (value >= 1000 && unit < units.length - 1) { value /= 1000; unit += 1; }
    return unit === 0 ? `${value} ${units[0]}` : `${value.toFixed(value < 10 ? 1 : 0)} ${units[unit]}`;
  }

  function plural(count, word, many) {
    return `${count.toLocaleString('en')} ${count === 1 ? word : many || word + 's'}`;
  }

  // What a node is called on the drawing: an index is shown the way it is written.
  function titleOf(node) {
    if (node.name === 'getitem') return `[${node.arguments.filter(arg => typeof arg === 'string').join(', ')}]`;
    return node.name;
  }

  function subOf(node) {
    return `${shapeText(node.shape)} ${node.dtype}`;
  }

  function categoryOf(node) {
    if (node.kind === 'earlier') return 'earlier';
    if (node.kind === 'input') return node.category === 'at once' ? 'atonce' : 'input';
    return node.category === 'indexing' ? 'view' : node.category || 'elementwise';   // A selection by arrays is drawn as a view is.
  }

  // ---- The model: the graph's JSON with the lookups the script needs.

  function modelOf(index) {
    if (models.has(index)) return models.get(index);
    const graph = DATA.graphs[index];
    const model = {
      graph,
      nodes: new Map(graph.nodes.map(node => [node.id, node])),
      consumers: new Map(),   // Node id -> ids of the nodes that read it.
      groups: new Map(),      // Group id -> group.
      parentOf: new Map(),    // Node id -> id of the group it is directly in.
      root: graph.groups,
      inputsIn: new Map(),    // Group id -> ids of the inputs drawn in that group.
      outputs: new Set(graph.outputs),
      counts: new Map(),      // Group id -> operations by name, worked out when first asked.
      asked: new Map(),       // Group id -> how many of the nodes asked for it holds.
    };
    for (const node of graph.nodes) for (const operand of node.operands) push(model.consumers, operand, node.id);
    let serial = 0;
    const walk = (group, parent) => {
      group.gid = `g${index}_${serial++}`;
      group.parent = parent;
      model.groups.set(group.gid, group);
      for (const child of group.children) {
        if (typeof child === 'number') model.parentOf.set(child, group.gid);
        else walk(child, group.gid);
      }
    };
    walk(model.root, null);
    // An input is drawn in the innermost group that holds everything that reads
    // it, so a weight sits beside the one product that uses it and not in a row
    // above the whole graph.
    const chain = id => {
      const found = [];
      for (let gid = model.parentOf.get(id); gid; gid = model.groups.get(gid).parent) found.push(gid);
      return found.reverse();
    };
    for (const node of graph.nodes) {
      if (node.kind === 'operation') continue;
      let shared = null;
      for (const reader of model.consumers.get(node.id) || []) {
        const path = chain(reader);
        if (shared === null) { shared = path; continue; }
        let same = 0;
        while (same < shared.length && same < path.length && shared[same] === path[same]) same += 1;
        shared = shared.slice(0, same);
      }
      const home = shared && shared.length ? shared[shared.length - 1] : model.root.gid;
      model.parentOf.set(node.id, home);
      push(model.inputsIn, home, node.id);
    }
    models.set(index, model);
    return model;
  }

  function askedIn(model, group) {
    if (!model.asked.has(group.gid)) model.asked.set(group.gid, nodesUnder(model, group, []).filter(node => model.outputs.has(node.id)).length);
    return model.asked.get(group.gid);
  }

  function nodesUnder(model, group, into) {
    for (const child of group.children) {
      if (typeof child === 'number') into.push(model.nodes.get(child));
      else nodesUnder(model, child, into);
    }
    return into;
  }

  function countsOf(model, group) {
    if (!model.counts.has(group.gid)) {
      const counts = new Map();
      for (const node of nodesUnder(model, group, [])) counts.set(titleName(node), (counts.get(titleName(node)) || 0) + 1);
      model.counts.set(group.gid, [...counts.entries()].sort((a, b) => b[1] - a[1]));
    }
    return model.counts.get(group.gid);
  }

  function titleName(node) {
    return node.name === 'getitem' ? 'index' : node.name;
  }

  // ---- Folding.

  // A group of more operations than the page's constant starts folded; so does
  // the middle of a long loop. While the top holds one folded group and nothing
  // else, that group is opened: a page that shows one box says nothing.
  function defaultFolds(model) {
    const folded = new Set();
    for (const group of model.groups.values()) {
      if (group !== model.root && (group.kind === 'rest' || group.operations > DATA.fold_above)) folded.add(group.gid);
    }
    let level = model.root;
    while (level.children.length === 1 && typeof level.children[0] !== 'number' && folded.has(level.children[0].gid)) {
      folded.delete(level.children[0].gid);
      level = level.children[0];
    }
    return folded;
  }

  function foldsOf(index) {
    if (!state.folds.has(index)) state.folds.set(index, defaultFolds(modelOf(index)));
    return state.folds.get(index);
  }

  // ---- What is drawn: one element per visible node, folded group and open group,
  // and one edge per pair of elements that data flows between.

  function visible(model, folded) {
    const elements = new Map();   // Element id -> {id, type, parent, node | group}.
    const stands = new Map();     // Node id -> id of the element that stands for it.
    const hide = (group, id) => {
      for (const child of group.children) {
        if (typeof child === 'number') stands.set(child, id);
        else hide(child, id);
      }
    };
    const inputsUnder = (group, into) => {
      into.push(...(model.inputsIn.get(group.gid) || []));
      for (const child of group.children) if (typeof child !== 'number') inputsUnder(child, into);
      return into;
    };
    // What a folded group reads from outside stays in sight, beside it: each
    // input by itself, or all of them as one pill when they are many.
    const addInputsOf = (group, parent) => {
      const inputs = inputsUnder(group, []);
      if (inputs.length > SHOWN_INPUTS) {
        const id = `${group.gid}_in`;
        elements.set(id, { id, type: 'inputs', parent, group, inputs: inputs.map(input => model.nodes.get(input)) });
        for (const input of inputs) stands.set(input, id);
        return;
      }
      for (const input of inputs) {
        elements.set(`n${input}`, { id: `n${input}`, type: 'node', parent, node: model.nodes.get(input) });
        stands.set(input, `n${input}`);
      }
    };
    const add = (group, parent) => {
      for (const input of model.inputsIn.get(group.gid) || []) {
        elements.set(`n${input}`, { id: `n${input}`, type: 'node', parent, node: model.nodes.get(input) });
        stands.set(input, `n${input}`);
      }
      for (const child of group.children) {
        if (typeof child === 'number') {
          elements.set(`n${child}`, { id: `n${child}`, type: 'node', parent, node: model.nodes.get(child) });
          stands.set(child, `n${child}`);
        } else if (folded.has(child.gid)) {
          addInputsOf(child, parent);
          elements.set(child.gid, { id: child.gid, type: 'folded', parent, group: child });
          hide(child, child.gid);
        } else {
          elements.set(child.gid, { id: child.gid, type: 'box', parent, group: child });
          add(child, child.gid);
        }
      }
    };
    add(model.root, null);
    const edges = new Map();
    for (const node of model.graph.nodes) {
      for (const operand of node.operands) {
        const source = stands.get(operand), target = stands.get(node.id);
        if (!source || !target || source === target) continue;
        const key = `${source}>${target}`;
        if (!edges.has(key)) edges.set(key, { id: `e${edges.size}`, source, target, carries: [] });
        edges.get(key).carries.push(operand);
      }
    }
    return { elements, stands, edges: [...edges.values()] };
  }

  function titleOfGroup(group) {
    if (group.kind === 'repeat') return `${group.label}  ${group.where}`;
    return group.label;
  }

  function subOfGroup(group) {
    const where = group.kind === 'repeat' || group.kind === 'block' ? '' : group.where;
    return plural(group.operations, 'operation') + (where ? ` · ${where}` : '');
  }

  function titleOfInputs(element) {
    return plural(element.inputs.length, 'input');
  }

  function subOfInputs(element) {
    return bytesText(element.inputs.reduce((total, node) => total + node.nbytes, 0));
  }

  function sizeOf(element) {
    if (element.type === 'inputs') {
      const width = Math.max(textWidth(titleOfInputs(element), TITLE_FONT), textWidth(subOfInputs(element), FONT) * 0.9);
      return { width: Math.max(MIN_WIDTH, Math.ceil(width) + 2 * PAD + 30), height: NODE_HEIGHT };
    }
    if (element.type === 'node') {
      // The tag stands on the title's line, in small capitals with spaced letters.
      const tag = tagOf(element.node);
      const tagged = textWidth(titleOf(element.node), TITLE_FONT) + (tag ? textWidth(tag.toUpperCase(), FONT) * 0.86 + 12 : 0);
      const width = Math.max(tagged, textWidth(subOf(element.node), FONT) * 0.9);
      return { width: Math.max(MIN_WIDTH, Math.ceil(width) + 2 * PAD + (element.node.kind === 'operation' ? 0 : 8)), height: NODE_HEIGHT };
    }
    const title = cut(titleOfGroup(element.group), LONGEST_TITLE);
    const font = element.group.kind === 'line' ? CODE_FONT : TITLE_FONT;
    const width = Math.max(textWidth(title, font), textWidth(subOfGroup(element.group), FONT) * 0.9);
    return { width: Math.max(MIN_WIDTH, Math.ceil(width) + 2 * PAD + 22 + 2 * STACK), height: FOLDED_HEIGHT + 2 * STACK };
  }

  function tagOf(node) {
    if (node.kind === 'earlier') return node.graph === null ? 'before' : `graph ${node.graph + 1}`;
    if (state.model.outputs.has(node.id)) return 'out';
    return '';
  }

  // ---- Layout: the visible elements as ELK's nested graph, and ELK's answer as places.

  function toElk(view) {
    const children = new Map();
    for (const element of view.elements.values()) push(children, element.parent, element);
    const make = element => {
      if (element.type !== 'box') return { id: element.id, ...sizeOf(element) };
      // No minimum size is asked for the title's sake: with the layout running
      // downwards, ELK applied a minimum width to the height. A title that does
      // not fit its box is cut.
      return { id: element.id, layoutOptions: { 'elk.padding': BOX_PADDING }, children: (children.get(element.id) || []).map(make) };
    };
    return {
      id: 'root',
      layoutOptions: ROOT_OPTIONS,
      children: (children.get(null) || []).map(make),
      edges: view.edges.map(edge => ({ id: edge.id, sources: [edge.source], targets: [edge.target] })),
    };
  }

  // The groups an element is in, innermost first.
  function boxesAround(view, id) {
    const found = [];
    for (let parent = view.elements.get(id).parent; parent; parent = view.elements.get(parent).parent) found.push(parent);
    return found;
  }

  // ELK gives an edge's points relative to the innermost group that holds both
  // of its ends. Newer versions name that group; for older ones it is worked out.
  function originOf(view, place, laid, edge) {
    let holder = laid.container;
    if (holder === undefined) {
      const around = new Set(boxesAround(view, edge.target));
      holder = boxesAround(view, edge.source).find(id => around.has(id)) || 'root';
    }
    return holder === 'root' || !place.has(holder) ? { x: 0, y: 0 } : place.get(holder);
  }

  function readLayout(view, laid) {
    const place = new Map();
    const walk = (node, x, y) => {
      for (const child of node.children || []) {
        place.set(child.id, { x: x + child.x, y: y + child.y, w: child.width, h: child.height });
        walk(child, x + child.x, y + child.y);
      }
    };
    walk(laid, 0, 0);
    const routes = new Map();
    const byId = new Map(view.edges.map(edge => [edge.id, edge]));
    const gather = node => {
      for (const edge of node.edges || []) {
        const origin = originOf(view, place, edge, byId.get(edge.id));
        const points = [];
        for (const section of edge.sections || []) {
          for (const point of [section.startPoint, ...(section.bendPoints || []), section.endPoint]) {
            points.push({ x: origin.x + point.x, y: origin.y + point.y });
          }
        }
        routes.set(edge.id, points);
      }
      for (const child of node.children || []) gather(child);
    };
    gather(laid);
    return { place, routes, size: { w: laid.width || 1, h: laid.height || 1 } };
  }

  // ---- Drawing.

  function pathThrough(points) {
    if (points.length < 2) return '';
    const corner = 7;
    let d = `M${points[0].x},${points[0].y}`;
    for (let i = 1; i < points.length - 1; i += 1) {
      const before = points[i - 1], here = points[i], after = points[i + 1];
      const toward = (to, by) => {
        const length = Math.hypot(to.x - here.x, to.y - here.y) || 1;
        const reach = Math.min(by, length / 2);
        return { x: here.x + (to.x - here.x) / length * reach, y: here.y + (to.y - here.y) / length * reach };
      };
      const enter = toward(before, corner), leave = toward(after, corner);
      d += ` L${enter.x},${enter.y} Q${here.x},${here.y} ${leave.x},${leave.y}`;
    }
    const last = points[points.length - 1];
    return `${d} L${last.x},${last.y}`;
  }

  function drawNode(element, at) {
    const node = element.node;
    const classes = ['node', `cat-${categoryOf(node)}`, `kind-${node.kind}`];
    if (state.model.outputs.has(node.id)) classes.push('output');
    if (node.state === 'failed') classes.push('failed');
    const group = el('g', { class: classes.join(' '), 'data-el': element.id, transform: `translate(${at.x},${at.y})` });
    const round = node.kind === 'operation' ? 7 : at.h / 2;
    group.append(el('rect', { width: at.w, height: at.h, rx: round }));
    const inset = node.kind === 'operation' ? PAD : PAD + 4;
    group.append(el('text', { class: 'title', x: inset, y: 18 }, titleOf(node)));
    group.append(el('text', { class: 'sub', x: inset, y: 33 }, subOf(node)));
    const tag = tagOf(node);
    if (tag) group.append(el('text', { class: 'tag', x: at.w - inset, y: 14, 'text-anchor': 'end' }, tag));
    return group;
  }

  function drawInputs(element, at) {
    const group = el('g', { class: 'node cat-input kind-inputs', 'data-el': element.id, transform: `translate(${at.x},${at.y})` });
    group.append(el('rect', { width: at.w, height: at.h, rx: at.h / 2 }));
    group.append(el('text', { class: 'title', x: PAD + 4, y: 18 }, titleOfInputs(element)));
    group.append(el('text', { class: 'sub', x: PAD + 4, y: 33 }, subOfInputs(element)));
    group.append(el('text', { class: 'tag', x: at.w - PAD - 4, y: 14, 'text-anchor': 'end' }, '+'));
    return group;
  }

  function drawFolded(element, at) {
    const width = at.w - 2 * STACK, height = at.h - 2 * STACK;
    const asked = askedIn(state.model, element.group) ? ' output' : '';
    const group = el('g', { class: `folded kind-${element.group.kind}${asked}`, 'data-el': element.id, transform: `translate(${at.x},${at.y})` });
    group.append(el('rect', { class: 'back', x: 2 * STACK, y: 2 * STACK, width, height, rx: 8 }));
    group.append(el('rect', { class: 'back', x: STACK, y: STACK, width, height, rx: 8 }));
    group.append(el('rect', { class: 'front', width, height, rx: 8 }));
    const coded = element.group.kind === 'line';
    group.append(el('text', { class: coded ? 'title code' : 'title', x: PAD, y: 19 }, cut(titleOfGroup(element.group), LONGEST_TITLE)));
    group.append(el('text', { class: 'sub', x: PAD, y: 35 }, subOfGroup(element.group)));
    group.append(el('text', { class: 'plus', x: width - 18, y: 28 }, '+'));
    return group;
  }

  function drawBox(element, at) {
    const group = el('g', { class: `box kind-${element.group.kind}`, 'data-el': element.id, transform: `translate(${at.x},${at.y})` });
    group.append(el('rect', { class: 'frame', width: at.w, height: at.h, rx: 9 }));
    group.append(el('rect', { class: 'bar', width: at.w, height: BAR_HEIGHT, rx: 9 }));
    const room = Math.max(6, Math.floor((at.w - 30) / 6.6));
    const coded = element.group.kind === 'line';
    const text = `▾ ${titleOfGroup(element.group)}${element.group.kind === 'line' ? '' : '   ' + subOfGroup(element.group)}`;
    group.append(el('text', { class: coded ? 'title code' : 'title', x: 10, y: 17 }, cut(text, room)));
    return group;
  }

  function drawEdge(edge, points) {
    const group = el('g', { class: 'edge', 'data-edge': edge.id });
    const d = pathThrough(points);
    const line = el('path', { class: 'line', d, 'marker-end': 'url(#arrow)' });
    line.style.strokeWidth = String(1.3 + Math.min(3, Math.log2(edge.carries.length)) * 0.7);
    group.append(line, el('path', { class: 'hit', d }));
    return group;
  }

  function render() {
    for (const layer of Object.values(layers)) layer.replaceChildren();
    for (const element of state.view.elements.values()) {
      const at = state.place.get(element.id);
      if (!at) continue;
      if (element.type === 'box') layers.boxes.append(drawBox(element, at));
      else if (element.type === 'inputs') layers.nodes.append(drawInputs(element, at));
      else layers.nodes.append(element.type === 'node' ? drawNode(element, at) : drawFolded(element, at));
    }
    for (const edge of state.view.edges) layers.edges.append(drawEdge(edge, state.routes.get(edge.id) || []));
    mark();
  }

  // ---- Selection and search, shown as classes on what is drawn.

  function reach(start, edgesFrom, pick) {
    const seen = new Set([start]);
    const queue = [start];
    while (queue.length) {
      for (const edge of edgesFrom.get(queue.pop()) || []) {
        if (!seen.has(pick(edge))) { seen.add(pick(edge)); queue.push(pick(edge)); }
      }
    }
    return seen;
  }

  function mark() {
    const view = state.view;
    const chosen = state.selected === null ? null : view.stands.get(state.selected);
    let up = null, down = null;
    if (chosen) {
      const into = new Map(), out = new Map();
      for (const edge of view.edges) { push(into, edge.target, edge); push(out, edge.source, edge); }
      up = reach(chosen, into, edge => edge.source);
      down = reach(chosen, out, edge => edge.target);
    }
    const hits = new Map();
    for (const id of state.matches) {
      const stand = view.stands.get(id);
      if (stand) hits.set(stand, (hits.get(stand) || 0) + 1);
    }
    for (const drawn of layers.nodes.children) {
      const id = drawn.getAttribute('data-el');
      drawn.classList.toggle('dim', Boolean(chosen) && !up.has(id) && !down.has(id));
      drawn.classList.toggle('selected', id === chosen && view.elements.get(id).type === 'node');
      drawn.classList.toggle('match', hits.has(id));
      drawn.querySelector('.badge')?.remove();
      if (hits.has(id) && view.elements.get(id).type === 'folded') {
        const badge = el('g', { class: 'badge', transform: `translate(${state.place.get(id).w - 2 * STACK - 2},2)` });
        badge.append(el('circle', { r: 9 }), el('text', { y: 3.5 }, String(hits.get(id))));
        drawn.append(badge);
      }
    }
    for (const drawn of layers.edges.children) {
      const edge = view.edges.find(each => each.id === drawn.getAttribute('data-edge'));
      const hot = Boolean(chosen) && ((up.has(edge.source) && up.has(edge.target)) || (down.has(edge.source) && down.has(edge.target)));
      drawn.classList.toggle('hot', hot);
      drawn.classList.toggle('dim', Boolean(chosen) && !hot);
      drawn.querySelector('.line').setAttribute('marker-end', hot ? 'url(#arrow-hot)' : 'url(#arrow)');
    }
  }

  function search(text) {
    state.query = text.trim().toLowerCase();
    state.cursor = -1;
    state.matches = [];
    if (state.query) {
      // An operation whose name matches comes before one that only stands on a line that does.
      const named = [], other = [];
      for (const node of state.model.graph.nodes) {
        const name = `${titleOf(node)} ${node.name}`.toLowerCase();
        const rest = `${node.where} ${node.code} ${node.dtype} ${shapeText(node.shape)} ${node.category}`.toLowerCase();
        if (name.includes(state.query)) named.push(node.id);
        else if (rest.includes(state.query)) other.push(node.id);
      }
      state.matches = [...named, ...other];
    }
    $('matches').textContent = state.query ? `${plural(state.matches.length, 'match', 'matches')} · Enter for the next` : '';
    mark();
  }

  // ---- Pan and zoom.

  function applyPan() {
    viewport.setAttribute('transform', `translate(${state.pan.x},${state.pan.y}) scale(${state.pan.k})`);
  }

  // 'whole' shows all of the drawing. 'top' is how a graph opens: all of it if
  // that leaves the text readable, and otherwise its width, from the top, at a
  // size that can be read, with the rest a drag away.
  function fit(how) {
    const frame = canvas.getBoundingClientRect();
    const across = (frame.width - 60) / state.size.w;
    const whole = Math.min(across, (frame.height - 90) / state.size.h, 1.3);
    const k = clamp(how === 'top' ? Math.max(whole, Math.min(READABLE, across)) : whole, ...ZOOM_LIMITS);
    const tall = state.size.h * k > frame.height - 60;
    state.pan = { k, x: (frame.width - state.size.w * k) / 2, y: tall ? 16 : Math.max(16, (frame.height - 40 - state.size.h * k) / 2) };
    applyPan();
    state.fitted = how;
  }

  function zoomAt(x, y, factor) {
    const k = clamp(state.pan.k * factor, ...ZOOM_LIMITS);
    const by = k / state.pan.k;
    state.pan = { k, x: x - (x - state.pan.x) * by, y: y - (y - state.pan.y) * by };
    state.fitted = null;
    applyPan();
  }

  function centreOn(id) {
    const at = state.place.get(id);
    if (!at) return;
    const frame = canvas.getBoundingClientRect();
    const k = clamp(Math.max(state.pan.k, 0.7), ...ZOOM_LIMITS);
    state.pan = { k, x: frame.width / 2 - (at.x + at.w / 2) * k, y: frame.height / 2 - (at.y + at.h / 2) * k };
    state.fitted = null;
    applyPan();
  }

  // ---- The code view: the program's lines, with what each wrote and where the layer computed.

  const LINES = new Map();   // "file:line" -> what happened on that line; see touch.
  const FILES = new Map();   // File name -> {text: Map(line number -> text), ops: operations its lines wrote}.

  function placeOf(where) {
    const cut = where.lastIndexOf(':');
    return { file: where.slice(0, cut), line: Number(where.slice(cut + 1)) };
  }

  function touch(where) {
    const { file, line } = placeOf(where);
    if (!FILES.has(file)) FILES.set(file, { text: new Map(), ops: 0 });
    if (!FILES.get(file).text.has(line)) FILES.get(file).text.set(line, '');
    // ops: operations written here. graphs: graph index -> how many of them went into it.
    // computed: indices of the graphs computed because of this line. eager: calls made at once here.
    if (!LINES.has(where)) LINES.set(where, { ops: 0, graphs: new Map(), computed: [], eager: [] });
    return LINES.get(where);
  }

  function indexLines() {
    for (const source of DATA.sources || []) {
      FILES.set(source.file, { text: new Map(Object.entries(source.text).map(([number, line]) => [Number(number), line])), ops: 0 });
    }
    DATA.graphs.forEach((graph, index) => {
      for (const node of graph.nodes) {
        if (node.kind !== 'operation' || !node.where) continue;
        const info = touch(node.where);
        info.ops += 1;
        info.graphs.set(index, (info.graphs.get(index) || 0) + 1);
        FILES.get(placeOf(node.where).file).ops += 1;
      }
      if (graph.where && graph.trigger !== 'pending') touch(graph.where).computed.push(index);
    });
    for (const event of DATA.events || []) if (event.where && event.kind === 'at once') touch(event.where).eager.push(event);
  }

  function linesOfGraph(index) {
    return new Set(DATA.graphs[index].nodes.filter(node => node.kind === 'operation' && node.where).map(node => node.where));
  }

  // "regression.py:27, 32-36" for the lines a graph's operations were written on.
  function coverage(index) {
    const byFile = new Map();
    for (const where of linesOfGraph(index)) push(byFile, placeOf(where).file, placeOf(where).line);
    const parts = [];
    for (const [file, numbers] of byFile) {
      numbers.sort((a, b) => a - b);
      const runs = [];
      for (const number of numbers) {
        if (runs.length && number === runs[runs.length - 1][1] + 1) runs[runs.length - 1][1] = number;
        else runs.push([number, number]);
      }
      parts.push(`${file}:${runs.map(([first, last]) => (first === last ? first : `${first}-${last}`)).join(', ')}`);
    }
    return parts.join('; ');
  }

  function chipFor(text, kind, title, act) {
    const made = html('button', `chip ${kind}`, text);
    made.title = title;
    made.addEventListener('click', event => { event.stopPropagation(); act(); });
    return made;
  }

  async function openGraph(index, where) {
    state.mode = 'graph';
    await showGraph(index);
    const node = state.model.graph.nodes.find(each => each.kind === 'operation' && each.where === where);
    if (node) await reveal(node.id);
  }

  function chipsOf(where, info) {
    const chips = html('span', 'chips');
    const times = count => (count > 1 ? ` × ${count.toLocaleString('en')}` : '');
    // How often each call was made at once on this line. One that also made the
    // layer compute a graph gets one chip, below, that says both.
    const eager = new Map(info.eager.map(event => [event.call, event.count]));
    const byCause = new Map();
    for (const index of info.computed) push(byCause, `${DATA.graphs[index].trigger}|${DATA.graphs[index].call}`, index);
    for (const [cause, indices] of byCause) {
      const [trigger, call] = cause.split('|');
      const target = indices.length === 1 ? `graph ${indices[0] + 1}` : `${indices.length} graphs`;
      let text = trigger === call ? trigger : `${trigger} ${call}`, kind = 'cause unasked';
      if (trigger === 'gather') kind = 'cause asked';
      if (trigger === 'at once') {
        text = `eager ${call}${times(eager.get(call) || 0)}`;
        kind = 'eager';
        eager.delete(call);
      }
      chips.append(chipFor(`${text} → ${target}`, kind,
        'This line made the layer compute. Click for the graph it computed', () => openGraph(indices[0], where)));
    }
    for (const [call, count] of eager) {
      chips.append(chipFor(`eager ${call}${times(count)}`, 'eager still',
        'Not in the deferred set: NumPy made the call at once. Nothing was pending, so no graph was computed here', () => {}));
    }
    const graphs = [...info.graphs.keys()];
    if (graphs.length <= 3) {
      for (const index of graphs) {
        chips.append(chipFor(`in graph ${index + 1}`, index === state.index ? 'member current' : 'member',
          'The operations this line wrote are in this graph. Click to see them there', () => openGraph(index, where)));
      }
    } else {
      const index = graphs.includes(state.index) ? state.index : graphs[0];
      chips.append(chipFor(`in ${graphs.length} graphs`, graphs.includes(state.index) ? 'member current' : 'member',
        'The operations this line wrote, over all its runs, are in this many graphs. Click for one of them', () => openGraph(index, where)));
    }
    return chips;
  }

  function drawCode() {
    const holder = $('code');
    const kept = holder.querySelector('.listing') ? holder.querySelector('.listing').scrollTop : 0;
    holder.replaceChildren();
    const files = [...FILES.entries()].sort((a, b) => b[1].ops - a[1].ops);
    if (!files.length) { holder.append(html('p', 'muted', 'This page carries no line of code.')); return; }
    if (!FILES.has(state.file)) state.file = files[0][0];
    const tabs = html('div', 'files');
    for (const [name, file] of files) {
      const tab = html('button', name === state.file ? 'file current' : 'file', name);
      tab.append(html('span', 'muted', file.ops ? `  ${plural(file.ops, 'operation')}` : ''));
      tab.addEventListener('click', () => { state.file = name; drawCode(); });
      tabs.append(tab);
    }
    const listing = html('div', 'listing');
    const text = FILES.get(state.file).text;
    const current = linesOfGraph(state.index);
    let before = 0;
    for (const number of [...text.keys()].sort((a, b) => a - b)) {
      if (before && number > before + 1) listing.append(html('div', 'gap', '⋯'));
      before = number;
      const where = `${state.file}:${number}`;
      const info = LINES.get(where);
      const row = html('div', 'row');
      row.dataset.where = where;
      if (info && info.ops) row.classList.add('wrote');
      if (current.has(where)) row.classList.add('in-graph');
      if (info && info.computed.includes(state.index)) row.classList.add('computed-here');
      if (where === state.flash) row.classList.add('flash');
      const line = html('code', 'text', text.get(number));
      line.title = text.get(number);
      row.append(html('span', 'no', String(number)), html('span', 'ops', info && info.ops ? info.ops.toLocaleString('en') : ''), line);
      if (info) row.append(chipsOf(where, info));
      listing.append(row);
    }
    holder.append(tabs, listing);
    listing.scrollTop = kept;
  }

  // Bring the lines of the graph that is shown into view, in the file that holds most of them.
  function showLinesOfGraph() {
    const counts = new Map();
    for (const where of linesOfGraph(state.index)) counts.set(placeOf(where).file, (counts.get(placeOf(where).file) || 0) + 1);
    const best = [...counts.entries()].sort((a, b) => b[1] - a[1])[0];
    if (best) state.file = best[0];
    else if (state.model.graph.where) state.file = placeOf(state.model.graph.where).file;
    drawCode();
    const first = $('code').querySelector('.row.in-graph, .row.computed-here');
    if (first) first.scrollIntoView({ block: 'center' });
  }

  function showCode(where) {
    state.mode = 'code';
    state.file = placeOf(where).file;
    state.flash = where;
    applyMode();
    banner();
    drawCode();
    const row = $('code').querySelector('.row.flash');
    if (row) row.scrollIntoView({ block: 'center' });
  }

  function applyMode() {
    const code = state.mode === 'code';
    $('code').hidden = !code;
    canvas.style.display = code ? 'none' : '';
    $('legend').hidden = code;
    if (code) { $('busy').hidden = true; $('tip').hidden = true; }
  }

  // ---- One drawing: fold state -> elements -> layout -> SVG.

  async function draw(options) {
    const turn = ++state.draws;
    const model = state.model = modelOf(state.index);
    const view = visible(model, foldsOf(state.index));
    if (!elk) {
      $('problem').hidden = false;
      $('problem').textContent = 'The layout library could not be loaded. This prototype\'s page loads it from a CDN and needs a network connection.';
      return;
    }
    // Where the element being folded or unfolded is on the screen now, to put it back there afterwards.
    const anchor = options.keep && state.place.has(options.keep)
      ? { id: options.keep, x: state.place.get(options.keep).x * state.pan.k + state.pan.x, y: state.place.get(options.keep).y * state.pan.k + state.pan.y }
      : null;
    $('busy').hidden = view.elements.size < 250;
    let laid;
    try {
      laid = await elk.layout(toElk(view));
    } catch (error) {
      $('busy').hidden = true;
      $('problem').hidden = false;
      $('problem').textContent = `The layout failed: ${error && error.message ? error.message : error}`;
      return;
    }
    if (turn !== state.draws) return;
    $('busy').hidden = true;
    $('problem').hidden = true;
    const read = readLayout(view, laid);
    Object.assign(state, { view, place: read.place, routes: read.routes, size: read.size });
    render();
    if (anchor && state.place.has(anchor.id)) {
      const at = state.place.get(anchor.id);
      state.pan = { k: state.pan.k, x: anchor.x - at.x * state.pan.k, y: anchor.y - at.y * state.pan.k };
      // A drawing that now fits the window is not left cut off at its edge.
      const frame = canvas.getBoundingClientRect();
      const wide = state.size.w * state.pan.k, tall = state.size.h * state.pan.k;
      if (wide <= frame.width - 24) state.pan.x = clamp(state.pan.x, 12, frame.width - wide - 12);
      if (tall <= frame.height - 24) state.pan.y = clamp(state.pan.y, 12, frame.height - tall - 12);
      state.fitted = null;
      applyPan();
    } else if (options.fit) {
      fit(options.fit);
    }
  }

  async function reveal(id) {
    if (state.mode !== 'graph') {
      state.mode = 'graph';
      applyMode();
      banner();
      await draw({ fit: 'top' });
    }
    const folded = foldsOf(state.index);
    let changed = false;
    for (let gid = state.model.parentOf.get(id); gid; gid = state.model.groups.get(gid).parent) {
      if (folded.delete(gid)) changed = true;
    }
    if (changed) await draw({});
    select(id);
    centreOn(`n${id}`);
  }

  function select(id) {
    state.selected = id;
    mark();
    describe();
  }

  async function showGraph(index, selected) {
    state.index = clamp(index, 0, DATA.graphs.length - 1);
    state.selected = null;
    state.model = modelOf(state.index);
    if (selected !== undefined) state.mode = 'graph';
    listGraphs();
    banner();
    applyMode();
    if (state.mode === 'code') {
      state.flash = null;
      showLinesOfGraph();
      describe();
      return;
    }
    await draw({ fit: 'top' });
    search($('search').value);
    if (selected !== undefined && state.model.nodes.has(selected)) await reveal(selected);
    else describe();
  }

  // ---- The list of graphs. Graphs that are alike share one entry.

  function families() {
    const found = [];
    const bySignature = new Map();
    DATA.graphs.forEach((graph, index) => {
      if (!bySignature.has(graph.signature)) {
        bySignature.set(graph.signature, { members: [] });
        found.push(bySignature.get(graph.signature));
      }
      bySignature.get(graph.signature).members.push(index);
    });
    return found;
  }

  function listGraphs() {
    const list = $('graphs');
    list.replaceChildren();
    for (const family of families()) {
      const here = family.members.includes(state.index);
      const graph = DATA.graphs[here ? state.index : family.members[0]];
      const entry = html('button', here ? 'entry current' : 'entry');
      const top = html('div', 'top');
      top.append(html('span', 'muted', `${graph.index + 1}`), html('span', 'call', graph.call));
      // A computation nobody asked for by name is the kind a reader is looking for.
      if (graph.trigger === 'needed') top.append(html('span', 'unasked', 'no gather'));
      if (family.members.length > 1) top.append(html('span', 'alike', `× ${family.members.length} alike`));
      const operations = graph.nodes.filter(node => node.kind === 'operation').length;
      entry.append(top, html('div', 'sub', `${graph.where || 'no line'} · ${plural(operations, 'operation')}`));
      entry.title = family.members.length > 1
        ? `Graphs ${family.members.map(index => index + 1).join(', ')} are alike in every operation, shape, type and line. Click again for the next.`
        : graph.code;
      entry.addEventListener('click', () => {
        const next = here ? family.members[(family.members.indexOf(state.index) + 1) % family.members.length] : family.members[0];
        showGraph(next);
      });
      list.append(entry);
    }
  }

  function banner() {
    const graph = state.model.graph;
    const bar = $('banner');
    bar.replaceChildren();
    for (const [mode, text, title] of [['code', 'Code', 'The program, and what each line wrote'], ['graph', 'Graph', 'The drawing of this graph']]) {
      const tab = html('button', state.mode === mode ? 'tab current' : 'tab', text);
      tab.title = title;
      tab.addEventListener('click', () => { state.mode = mode; showGraph(state.index); });
      bar.append(tab);
    }
    const back = html('button', '', '‹'), on = html('button', '', '›');
    back.title = 'The graph before'; on.title = 'The graph after';
    back.addEventListener('click', () => showGraph(state.index - 1));
    on.addEventListener('click', () => showGraph(state.index + 1));
    const told = (TRIGGER_TEXT[graph.trigger] || (call => call))(graph.call);
    bar.append(back, on, html('b', '', `Graph ${graph.index + 1} of ${DATA.graphs.length}`), html('span', '', `${told}${graph.where ? ' at ' + graph.where : ''}`));
    if (graph.code) bar.append(html('code', '', graph.code));
  }

  function legend() {
    const box = $('legend');
    for (const [key, text] of CATEGORIES) {
      const item = html('span', '', text), swatch = html('i');
      swatch.style.background = `var(--${key}-fill)`;
      swatch.style.borderColor = `var(--${key}-line)`;
      if (key === 'earlier') swatch.style.borderStyle = 'dashed';
      item.prepend(swatch);
      box.append(item);
    }
    box.append(html('span', '', 'thick border: asked for'));
  }

  // ---- The details panel: the selected node, or the graph when nothing is selected.

  function chip(node) {
    const key = categoryOf(node);
    const made = html('span', 'chip', CATEGORIES.find(each => each[0] === key)[1]);
    made.style.borderColor = `var(--${key}-line)`;
    made.style.background = `var(--${key}-fill)`;
    return made;
  }

  function facts(pairs) {
    const list = html('dl', 'facts');
    for (const [name, value] of pairs) list.append(html('dt', '', name), html('dd', '', value));
    return list;
  }

  function linkTo(node) {
    const link = html('button', 'link', `${titleOf(node)}  ${subOf(node)}${node.where ? '  ·  ' + node.where : ''}`);
    link.addEventListener('click', () => reveal(node.id));
    return link;
  }

  function describeNode(panel, node) {
    const model = state.model;
    panel.append(chip(node), html('h2', '', titleOf(node)));
    const elements = node.shape.reduce((total, length) => total * length, 1);
    const pairs = [['type', node.dtype], ['shape', shapeText(node.shape)], ['size', `${plural(elements, 'element')}, ${bytesText(node.nbytes)}`], ['state', node.state]];
    if (node.where) pairs.push([node.kind === 'operation' ? 'written at' : 'entered at', node.where]);
    if (model.outputs.has(node.id)) pairs.push(['asked for', `by ${model.graph.call}`]);
    panel.append(facts(pairs));
    if (node.code) panel.append(html('pre', '', node.code));
    if (node.where && FILES.has(placeOf(node.where).file)) {
      const there = html('button', 'link', `Show ${node.where} in the code`);
      there.addEventListener('click', () => showCode(node.where));
      panel.append(there);
    }
    if (node.kind === 'earlier') {
      if (node.graph === null) {
        panel.append(html('p', 'muted', 'Computed before the recording began.'));
      } else {
        const jump = html('button', 'link', `Computed by graph ${node.graph + 1}: go there`);
        jump.addEventListener('click', () => showGraph(node.graph, node.id));
        panel.append(jump);
      }
    }
    if (node.arguments.length) {
      panel.append(html('h3', '', 'Arguments, as written'));
      for (const argument of node.arguments) {
        if (typeof argument === 'number') panel.append(linkTo(model.nodes.get(argument)));
        else panel.append(html('div', 'value mono', argument));
      }
    }
    const readers = model.consumers.get(node.id) || [];
    if (readers.length) {
      panel.append(html('h3', '', `Read by ${plural(readers.length, 'operation')}`));
      for (const reader of readers.slice(0, 40)) panel.append(linkTo(model.nodes.get(reader)));
      if (readers.length > 40) panel.append(html('p', 'muted', `and ${readers.length - 40} more`));
    }
  }

  function meters(panel, counts) {
    const most = counts.length ? counts[0][1] : 1;
    for (const [name, count] of counts) {
      const row = html('button', 'count');
      const meter = html('span', 'meter');
      meter.style.width = `${Math.max(3, Math.round(100 * count / most))}%`;
      const track = html('span');
      track.append(meter);
      row.append(html('span', 'mono', name), track, html('span', 'number', count.toLocaleString('en')));
      row.title = `Find every ${name}`;
      row.addEventListener('click', () => { $('search').value = name === 'index' ? 'getitem' : name; search($('search').value); });
      panel.append(row);
    }
  }

  function describeGraph(panel) {
    const model = state.model, graph = model.graph;
    const operations = graph.nodes.filter(node => node.kind === 'operation');
    panel.append(html('h2', '', `Graph ${graph.index + 1}`));
    const told = (TRIGGER_TEXT[graph.trigger] || (call => call))(graph.call);
    panel.append(html('div', 'muted', `${told}${graph.where ? ' at ' + graph.where : ''}`));
    if (graph.code) panel.append(html('pre', '', graph.code));
    const written = operations.reduce((total, node) => total + node.nbytes, 0);
    panel.append(facts([
      ['operations', operations.length.toLocaleString('en')],
      ['inputs', String(graph.nodes.filter(node => node.kind === 'input').length)],
      ['from earlier graphs', String(graph.nodes.filter(node => node.kind === 'earlier').length)],
      ['asked for', String(graph.outputs.length)],
      ['results, if all were kept', bytesText(written)],
      ['written on', coverage(state.index) || 'no line known'],
    ]));
    if (DATA.dropped) panel.append(html('p', 'muted', `${plural(DATA.dropped, 'operation')} of the block were written and dropped without being computed.`));
    panel.append(html('h3', '', 'Operations by name'));
    meters(panel, countsOf(model, model.root));
    panel.append(html('h3', '', 'Using the page'));
    const help = html('ul');
    for (const line of ['Code shows the program: the lines of this graph are marked, and each line says what it wrote and what it made the layer compute.',
      'Click a stacked box to unfold it; click the title of an open group to fold it.',
      'Click a node for its line, its arguments and what reads it.',
      'Drag to move, scroll to zoom, f to fit.',
      'Type / to search; Enter goes to the next match and unfolds what hides it.']) help.append(html('li', '', line));
    panel.append(help);
  }

  function describe() {
    const panel = $('details');
    panel.replaceChildren();
    const node = state.selected === null ? null : state.model.nodes.get(state.selected);
    if (node) describeNode(panel, node); else describeGraph(panel);
  }

  // ---- The tip that follows the pointer.

  function tipFor(target) {
    const tip = html('div');
    const edgeHolder = target.closest('[data-edge]');
    if (edgeHolder) {
      const edge = state.view.edges.find(each => each.id === edgeHolder.getAttribute('data-edge'));
      const carried = [...new Set(edge.carries)].map(id => state.model.nodes.get(id));
      if (carried.length === 1) tip.append(html('b', '', titleOf(carried[0])), html('div', 'row', `${subOf(carried[0])} · ${bytesText(carried[0].nbytes)}`));
      else tip.append(html('b', '', plural(carried.length, 'array')), html('div', 'row', bytesText(carried.reduce((total, node) => total + node.nbytes, 0))));
      return tip;
    }
    const holder = target.closest('[data-el]');
    if (!holder) return null;
    const element = state.view.elements.get(holder.getAttribute('data-el'));
    if (element.type === 'node') {
      const node = element.node;
      tip.append(html('b', '', titleOf(node)), html('div', 'row', `${subOf(node)} · ${bytesText(node.nbytes)}`));
      if (node.where) tip.append(html('div', 'row muted', node.where));
      if (node.code) tip.append(html('code', '', node.code));
      return tip;
    }
    if (element.type === 'inputs') {
      tip.append(html('b', '', `${titleOfInputs(element)} of ${titleOfGroup(element.group)}`), html('div', 'row', subOfInputs(element)));
      for (const node of element.inputs.slice(0, 8)) tip.append(html('div', 'row muted', `${titleOf(node)}  ${subOf(node)}`));
      if (element.inputs.length > 8) tip.append(html('div', 'row muted', `and ${element.inputs.length - 8} more`));
      tip.append(html('div', 'row muted', 'Click to unfold the group and see each where it is read'));
      return tip;
    }
    if (element.type === 'box' && !target.closest('.bar')) return null;
    const group = element.group;
    tip.append(html('b', '', titleOfGroup(group)), html('div', 'row', subOfGroup(group)));
    if (group.code && (group.kind === 'call' || group.kind === 'repeat')) tip.append(html('code', '', group.code));
    const counts = countsOf(state.model, group).slice(0, 6).map(([name, count]) => `${count} ${name}`).join(', ');
    tip.append(html('div', 'row muted', counts));
    const asked = askedIn(state.model, group);
    if (asked) tip.append(html('div', 'row muted', `holds ${asked} of the ${plural(state.model.outputs.size, 'array')} asked for`));
    tip.append(html('div', 'row muted', element.type === 'box' ? 'Click to fold' : 'Click to unfold'));
    return tip;
  }

  function hover(event) {
    const tip = $('tip');
    const content = tipFor(event.target);
    if (!content) { tip.hidden = true; return; }
    tip.replaceChildren(...content.childNodes);
    tip.hidden = false;
    const width = tip.offsetWidth, height = tip.offsetHeight;
    tip.style.left = `${Math.min(event.clientX + 14, window.innerWidth - width - 8)}px`;
    tip.style.top = `${event.clientY + 18 + height > window.innerHeight ? event.clientY - height - 12 : event.clientY + 18}px`;
  }

  // ---- What the reader does.

  function click(target) {
    $('tip').hidden = true;
    const holder = target.closest('[data-el]');
    const element = holder ? state.view.elements.get(holder.getAttribute('data-el')) : null;
    const folded = foldsOf(state.index);
    if (element && (element.type === 'folded' || element.type === 'inputs')) {
      folded.delete(element.group.gid);
      draw({ keep: element.group.gid });
    } else if (element && element.type === 'box' && target.closest('.bar')) {
      folded.add(element.id);
      draw({ keep: element.id });
    } else if (element && element.type === 'node') {
      select(element.node.id);
    } else {
      select(null);
    }
  }

  let press = null;
  canvas.addEventListener('pointerdown', event => {
    if (event.button !== 0) return;
    press = { x: event.clientX, y: event.clientY, pan: { ...state.pan }, moved: false, target: event.target };
    canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener('pointermove', event => {
    if (!press) { hover(event); return; }
    const dx = event.clientX - press.x, dy = event.clientY - press.y;
    if (!press.moved && Math.hypot(dx, dy) < DRAG_THRESHOLD) return;
    press.moved = true;
    canvas.classList.add('dragging');
    $('tip').hidden = true;
    state.pan = { k: press.pan.k, x: press.pan.x + dx, y: press.pan.y + dy };
    state.fitted = null;
    applyPan();
  });
  canvas.addEventListener('pointerup', () => {
    const was = press;
    press = null;
    canvas.classList.remove('dragging');
    if (was && !was.moved) click(was.target);
  });
  canvas.addEventListener('pointerleave', () => { $('tip').hidden = true; });
  canvas.addEventListener('wheel', event => {
    event.preventDefault();
    const frame = canvas.getBoundingClientRect();
    // A pinch on a trackpad arrives as a wheel event with ctrlKey set and small deltas.
    zoomAt(event.clientX - frame.left, event.clientY - frame.top, Math.exp(-event.deltaY * (event.ctrlKey ? 0.01 : 0.0018)));
  }, { passive: false });

  // A page opened in a hidden tab has no size yet, and a window can be resized:
  // while the reader has not moved the drawing, keep it fitted.
  new ResizeObserver(() => { if (state.fitted && state.view) fit(state.fitted); }).observe(canvas);
  $('fit').addEventListener('click', () => fit('whole'));
  $('reset').addEventListener('click', () => { state.folds.delete(state.index); draw({ fit: 'top' }); });
  $('unfold').addEventListener('click', () => {
    const folded = foldsOf(state.index);
    for (const group of state.model.groups.values()) if (group.kind !== 'rest') folded.delete(group.gid);
    draw({ fit: 'whole' });
  });
  $('download').addEventListener('click', () => {
    const link = document.createElement('a');
    link.href = URL.createObjectURL(new Blob([JSON.stringify({ graphs: DATA.graphs, dropped: DATA.dropped }, null, 1)], { type: 'application/json' }));
    link.download = 'lazy-graphs.json';
    link.click();
    URL.revokeObjectURL(link.href);
  });
  $('search').addEventListener('input', event => search(event.target.value));
  $('search').addEventListener('keydown', event => {
    if (event.key === 'Enter' && state.matches.length) {
      state.cursor = (state.cursor + (event.shiftKey ? -1 : 1) + state.matches.length) % state.matches.length;
      reveal(state.matches[state.cursor]);
    } else if (event.key === 'Escape') {
      event.target.value = '';
      search('');
      event.target.blur();
    }
  });
  document.addEventListener('keydown', event => {
    if (event.target.tagName === 'INPUT' || event.metaKey || event.ctrlKey || event.altKey) return;
    if (event.key === 'f') fit('whole');
    else if (event.key === '/') { event.preventDefault(); $('search').focus(); }
    else if (event.key === 'Escape') select(null);
  });

  // ---- Start.

  if (!DATA.graphs.length) {
    $('problem').hidden = false;
    $('problem').textContent = 'Nothing was recorded: the block made no lazy operation.';
    return;
  }
  const operations = DATA.graphs.reduce((total, graph) => total + graph.nodes.filter(node => node.kind === 'operation').length, 0);
  $('summary').textContent = `${plural(DATA.graphs.length, 'graph')}, ${plural(operations, 'operation')}`;
  legend();
  indexLines();
  if (!FILES.size) state.mode = 'graph';
  window.lazyGraphs = { state, DATA, showGraph, reveal, fit, showCode };   // For looking around from the console.
  showGraph(0);
})();
