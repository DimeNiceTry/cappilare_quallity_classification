// Exercise the actual gallery filters against saved predictions without a browser.
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const source=fs.readFileSync('focus_classifier/static/app.js','utf8');
const gallery=JSON.parse(fs.readFileSync('focus_classifier/artifacts/gallery.json','utf8'));
const defaults={mode:'balanced',threshold:'0.5',dataset:'consensus',view:'all',truth:'all',sort:'desc',search:'',compare:'',model:''};
const nodes={};
function node(id){return nodes[id]??=(Object.assign({value:defaults[id]??'',checked:id==='labels',innerHTML:'',textContent:'',hidden:false,
 addEventListener(){},querySelectorAll(){return []},scrollIntoView(){},showModal(){this.open=true},close(){this.open=false},
 classList:{remove(){},add(){}}},{}));}
let blob;
const context=vm.createContext({console,setTimeout,document:{getElementById:node,createElement(){return {click(){}}}},Blob,
 URL:{createObjectURL(b){blob=b;return 'blob:test'},revokeObjectURL(){}},
 fetch:async url=>({ok:true,json:async()=>url.includes('gallery')?gallery:{models:{}}})});
vm.runInContext(source,context);
setTimeout(async()=>{
 try{
  assert.match(nodes.selection.textContent,/565/);
  assert.equal((nodes.gallery.innerHTML.match(/class="card"/g)||[]).length,24);
  nodes.mode.value='fixed';nodes.threshold.value='0';vm.runInContext('render()',context);
  assert.match(nodes.stats.innerHTML,/11\.0%/);
  nodes.view.value='false_accept';vm.runInContext('render()',context);assert.match(nodes.selection.textContent,/503/);
  nodes.view.value='false_reject';vm.runInContext('render()',context);assert.match(nodes.selection.textContent,/^0 /);
  nodes.mode.value='precise';nodes.model.value='trees_extended';nodes.view.value='accepted';vm.runInContext('render()',context);assert.match(nodes.selection.textContent,/^32 /);
  nodes.model.value='neural';vm.runInContext('render()',context);assert.match(nodes.selection.textContent,/^41 /);
  nodes.dataset.value='unresolved';nodes.view.value='all';vm.runInContext('render()',context);assert.match(nodes.selection.textContent,/^435 /);
  nodes.dataset.value='all';nodes.mode.value='balanced';nodes.view.value='different';nodes.model.value='trees_extended';nodes.compare.value='neural';vm.runInContext('render()',context);
  const expected=gallery.rows.filter(r=>(r.scores.trees_extended.p>=r.scores.trees_extended.threshold)!==(r.scores.neural.p>=r.scores.neural.threshold)).length;
  assert.equal(nodes.selection.textContent,`${expected} фотографий в отборе`);
  nodes.view.value='all';nodes.search.value=gallery.rows[0].file;vm.runInContext('render()',context);assert.match(nodes.selection.textContent,/^1 /);
  vm.runInContext('openDetail(data.rows[0]); exportSelection()',context);
  assert(nodes.detail.open);assert.match(nodes['detail-info'].innerHTML,/1024/);
  assert((await blob.text()).includes(gallery.rows[0].file));
  console.log('Viewer: filters, thresholds, errors, comparison, details and CSV export passed.');
 }catch(error){console.error(error);process.exitCode=1;}
},50);
