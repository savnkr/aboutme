// Turns container directives into the maths-blog blocks used across posts:
//
//   :::definition[Gaussian process]      → a labelled "Definition" box
//   :::theorem / :::lemma / :::proposition / :::corollary
//   :::question                          → a labelled box
//   :::note / :::remark / :::example    → a collapsed <details> toggle, titled like a box
//   :::proof[optional title]             → a collapsed <details> "Proof" toggle
//   :::answer                            → a collapsed "Answer" toggle (for questions)
//
// Numbered blocks count separately per type within a post (Definition 1, 2…; Proposition 1, 2…).
import { visit } from 'unist-util-visit';

const LABELS = {
  definition: 'Definition',
  theorem: 'Theorem',
  lemma: 'Lemma',
  proposition: 'Proposition',
  corollary: 'Corollary',
  note: 'Note',
  question: 'Question',
  example: 'Example',
  remark: 'Remark',
};
const NUMBERED = new Set(['definition', 'theorem', 'lemma', 'proposition', 'corollary', 'example', 'question']);
const FOLDED = { proof: 'Proof', answer: 'Answer' };
// Asides fold away like proofs, but keep their label, number and colour.
const TOGGLED = new Set(['note', 'remark', 'example']);

// Text like "ratio:abc" parses as a directive; put anything that isn't ours back as plain text.
function restore(node) {
  const inner = (node.children || []).map((c) => c.value ?? '').join('');
  const prefix = node.type === 'leafDirective' ? '::' : ':';
  return { type: 'text', value: `${prefix}${node.name}${inner ? `[${inner}]` : ''}` };
}

export default function remarkBlocks() {
  return (tree) => {
    visit(tree, ['textDirective', 'leafDirective'], (node, index, parent) => {
      if (parent && typeof index === 'number') parent.children[index] = restore(node);
    });
    const counts = {};
    visit(tree, 'containerDirective', (node) => {
      const name = node.name;
      if (!(name in LABELS) && !(name in FOLDED)) {
        node.data = { hName: 'div' };
        return;
      }
      // The [label] part arrives as the first paragraph with data.directiveLabel.
      let title = '';
      const first = node.children[0];
      if (first && first.data && first.data.directiveLabel) {
        title = first.children.map((c) => c.value ?? '').join('');
        node.children.shift();
      }
      if (name in FOLDED) {
        node.data = { hName: 'details', hProperties: { className: ['blk', `blk-${name}`] } };
        node.children.unshift({
          type: 'paragraph',
          data: { hName: 'summary' },
          children: [{ type: 'text', value: title ? `${FOLDED[name]}: ${title}` : FOLDED[name] }],
        });
        return;
      }
      const n = NUMBERED.has(name) ? ` ${(counts[name] = (counts[name] || 0) + 1)}` : '';
      const toggled = TOGGLED.has(name);
      node.data = { hName: toggled ? 'details' : 'div', hProperties: { className: ['blk', `blk-${name}`, ...(toggled ? ['blk-toggle'] : [])] } };
      node.children.unshift({
        type: 'paragraph',
        data: { hName: toggled ? 'summary' : 'p', hProperties: { className: ['blk-head'] } },
        children: [
          { type: 'strong', children: [{ type: 'text', value: `${LABELS[name]}${n}` }] },
          ...(title ? [{ type: 'text', value: ` (${title})` }] : []),
        ],
      });
    });
  };
}
