const fs=require('fs'),vm=require('vm'),assert=require('assert');
const source=fs.readFileSync('focus_classifier/static/upload.js','utf8');
const nodes={},articles=new Map();let savedBlob,postCount=0;
const inputs=[{value:'neural',checked:true},{value:'trees_extended',checked:true},{value:'logistic_basic',checked:false}];
function element(){return {innerHTML:'',textContent:'',value:'',checked:false,hidden:false,disabled:false,dataset:{},classList:{toggle(){},add(){},remove(){}},addEventListener(){},querySelectorAll(){return []},querySelector(){return element()},click(){},showModal(){this.open=true},close(){this.open=false},append(child){articles.set(child.dataset.record,child)}};}
function node(id){if(!nodes[id])nodes[id]=element();return nodes[id];}
node('inference-mode').value='precise';node('manual-threshold').value='0.5';
const config={token:'test-session',max_bytes:20*1024*1024,models:inputs.map(m=>({id:m.value,name:m.value,thresholds:{precise:m.value==='logistic_basic'?2:.7,balanced:.2},auroc:.9}))};
const context=vm.createContext({console,setTimeout,Blob,URLSearchParams,AbortController,
 URL:{createObjectURL(b){if(b instanceof Blob)savedBlob=b;return 'blob:test'},revokeObjectURL(){}},
 document:{getElementById:node,createElement:element,querySelector(selector){const id=selector.match(/data-record="(\d+)"/);return id?articles.get(id[1]):null},querySelectorAll(selector){return selector.includes(':checked')?inputs.filter(n=>n.checked):inputs}},
 fetch:async(url,options)=>{
  if(url.includes('/config'))return {ok:true,json:async()=>config};
  postCount++;assert.equal(options.method,'POST');assert.equal(options.headers['X-Inference-Token'],'test-session');
  const q=new URLSearchParams(url.split('?')[1]);assert.equal(q.get('models'),'neural,trees_extended');assert.equal(q.get('mode'),'precise');
  if(options.body.name==='bad.png')return {ok:false,json:async()=>({error:'Invalid image'})};
  return {ok:true,json:async()=>({image:{width:1024,height:768,format:'PNG',resized:false},mode:'precise',timing:{total_ms:120,cold_neural:false},known_image:{in_training:true},results:[{model:'neural',name:'Neural',p_ok:.8,threshold:.7,analysis_ms:32,decision:'ok'},{model:'trees_extended',name:'Trees',p_ok:.4,threshold:.7,analysis_ms:150,decision:'not_ok'}]})};
 }});
vm.runInContext(source,context);
setTimeout(async()=>{
 try{
  vm.runInContext(`addFiles([{name:'sample<script>.png',size:100,lastModified:1},{name:'bad.png',size:100,lastModified:2}])`,context);
  assert(!nodes.evaluate.disabled);assert(nodes['selection-list'].innerHTML.includes('&lt;script&gt;'));
  await vm.runInContext('evaluate()',context);assert.equal(postCount,2);
  assert.match(nodes['upload-status'].textContent,/ошибок 1/);
  assert(articles.get('1').innerHTML.includes('Годен по модели'));
  assert(articles.get('1').innerHTML.includes('обучающем наборе'));
  assert(!articles.get('1').innerHTML.includes('sample<script>'));
  assert(articles.get('2').innerHTML.includes('Invalid image'));
  vm.runInContext('saveCSV()',context);assert((await savedBlob.text()).includes('trees_extended'));
  nodes['inference-mode'].value='manual';vm.runInContext('modeHelp()',context);assert(!nodes['manual-box'].hidden);
  vm.runInContext('clearAll()',context);assert.equal(nodes['upload-results'].innerHTML,'');assert(nodes.evaluate.disabled);
  console.log('Upload UI: model choices, POST request, per-file errors, HTML escaping, known-image note, threshold mode and CSV passed.');
 }catch(error){console.error(error);process.exitCode=1;}
},50);
