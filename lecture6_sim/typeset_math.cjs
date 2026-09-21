// Compile the project's LaTeX mathematics to self-contained SVG paths.
const fs = require('fs');
const path = require('path');
const base = path.resolve(__dirname, '../.mathjax/node_modules/mathjax-full/js');
const {mathjax} = require(path.join(base, 'mathjax.js'));
const {TeX} = require(path.join(base, 'input/tex.js'));
const {SVG} = require(path.join(base, 'output/svg.js'));
const {liteAdaptor} = require(path.join(base, 'adaptors/liteAdaptor.js'));
const {RegisterHTMLHandler} = require(path.join(base, 'handlers/html.js'));
const {AllPackages} = require(path.join(base, 'input/tex/AllPackages.js'));

const adaptor = liteAdaptor();
RegisterHTMLHandler(adaptor);
const document = mathjax.document('', {
  // Do not turn malformed commands into visible text inside a saved figure.
  InputJax: new TeX({
    packages: AllPackages.filter(name => !['noundefined', 'noerrors'].includes(name)),
    formatError: (_jax, error) => { throw error; },
  }),
  OutputJax: new SVG({fontCache: 'none'}),
});
const equations = JSON.parse(fs.readFileSync(0, 'utf8'));
const rendered = {};
for (const [page, lines] of Object.entries(equations)) {
  rendered[page] = lines.map(latex => {
    const node = document.convert(latex, {display: true});
    const svg = adaptor.outerHTML(adaptor.firstChild(node));
    if (svg.includes('data-mjx-error') || svg.includes('data-mml-node="merror"')) {
      throw new Error(`Invalid equation on page ${page}: ${latex}\n${svg}`);
    }
    return svg;
  });
}
process.stdout.write(JSON.stringify(rendered));
