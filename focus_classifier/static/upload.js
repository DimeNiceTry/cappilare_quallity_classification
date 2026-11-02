'use strict';
const $=id=>document.getElementById(id);
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const percent=x=>(x*100).toFixed(1)+'%';
const decisions={ok:'Годен по модели',not_ok:'Не годен по модели',disabled:'Автодопуск отключён'};
let config,files=[],records=[],running=false,controller=null,nextId=0;
function selectedModels(){return [...document.querySelectorAll('input[name="upload-model"]:checked')].map(n=>n.value);}
function status(message,error=false){$('upload-status').textContent=message;$('upload-status').classList.toggle('failure',error);}
function setBusy(value){running=value;$('evaluate').disabled=value||!config||!files.length||!selectedModels().length;$('cancel').hidden=!value;$('clear').disabled=value||!files.length;$('save-csv').disabled=!records.some(r=>r.result);$('photo-files').disabled=value;document.querySelectorAll('.settings input,.settings select').forEach(n=>n.disabled=value);renderFiles();}
function renderFiles(){
 $('selection-list').innerHTML=files.length?files.map((f,i)=>`<div class="file-chip"><span title="${esc(f.name)}">${esc(f.name)}</span><button data-remove="${i}" ${running?'disabled':''} aria-label="Убрать ${esc(f.name)}">×</button></div>`).join(''):'<span class="hint">Фотографии ещё не выбраны.</span>';
 $('selection-list').querySelectorAll('[data-remove]').forEach(b=>b.onclick=()=>{files.splice(Number(b.dataset.remove),1);setBusy(false);});
}
function addFiles(incoming){
 if(running)return;
 let rejected=0;
 for(const file of incoming){if(file.size>(config?.max_bytes||20*1024*1024)||files.length>=50){rejected++;continue;}if(!files.some(f=>f.name===file.name&&f.size===file.size&&f.lastModified===file.lastModified))files.push(file);}
 setBusy(false);status(rejected?`Добавлено ${files.length} файлов. Пропущено ${rejected}: лимит 20 МБ на файл и 50 файлов за раз.`:`Выбрано ${files.length} фотографий. Настрой модели и нажми «Оценить».`,rejected>0);
}
function modeHelp(){
 const mode=$('inference-mode').value;$('manual-box').hidden=mode!=='manual';$('manual-value').value=Number($('manual-threshold').value).toFixed(2);
 $('mode-help').textContent=mode==='precise'?'Меньше плохих среди принятых, но часть хороших будет отклонена. Для некоторых моделей строгий допуск недоступен.':mode==='balanced'?'Режим сохраняет больше хороших кадров и допускает больше ошибок.': 'Одинаковый ручной порог для всех выбранных моделей. Его качество на новых фотографиях ещё нужно проверить.';
}
function renderRecord(record){
 const element=document.querySelector(`[data-record="${record.id}"]`);if(!element)return;
 const head=`<div class="upload-result-head"><strong>${esc(record.file.name)}</strong><span>${(record.file.size/1024/1024).toFixed(2)} МБ</span></div>`;
 if(record.error){element.innerHTML=head+`<div class="upload-error">${esc(record.error)}</div>`;return;}
 if(!record.result){element.innerHTML=head+'<div class="processing">Оценка… Первый запуск нейросети может занять несколько секунд.</div>';return;}
 const r=record.result;
 const scores=r.results.map(s=>`<div class="score-row ${s.decision}"><div><strong>${esc(s.name)}</strong><span class="sub">${s.threshold>1?'Строгий порог не подобран':`Порог ${percent(s.threshold)}`} · анализ ${s.analysis_ms.toFixed(0)} мс</span></div><div><div class="score-value">${percent(s.p_ok)}</div><div class="decision">${decisions[s.decision]}</div></div></div>`).join('');
 let notes=[];
 if(r.image.resized)notes.push(`Размер ${r.image.width}×${r.image.height} преобразован в 1024×768. Это пробная оценка после изменения масштаба.`);
 if(r.known_image)notes.push(r.known_image.in_training?'Этот кадр уже был в обучающем наборе. Полученная оценка не является независимой проверкой модели.':'Этот кадр входит в исходный набор проекта. Для оценки эксперимента используй страницу сравнения на датасете.');
 if(r.image.format==='JPEG')notes.push('JPEG поддерживается; модели проверялись на PNG.');
 element.innerHTML=head+`<div class="upload-result-body"><button class="upload-thumb" data-preview="${record.id}" aria-label="Открыть исходную фотографию"><img src="${esc(record.url)}" alt="Загруженная фотография"><span>Открыть · 1:1</span></button><div class="upload-scores">${scores}</div></div>`+
 `<div class="upload-note ${notes.length?'caution':''}">${r.image.width}×${r.image.height} · ${esc(r.image.format)} · P(годен) показана в процентах. ${notes.map(esc).join(' ')}</div>`+
 `<div class="upload-note">Всё вычисление: ${(r.timing.total_ms/1000).toFixed(2)} с${r.timing.cold_neural?' · включая первый запуск нейросети':''}. ${r.results.some(s=>s.decision==='disabled')?'«Автодопуск отключён» относится к настройке модели, а не к качеству фотографии.':''}</div>`;
 element.querySelector('[data-preview]').onclick=()=>openPreview(record);
 const img=element.querySelector('img');img.onerror=()=>{img.alt='Превью этого формата недоступно в браузере. Оценка выполнена.';};
}
function summary(){
 const completed=records.filter(r=>r.result);$('batch-summary').hidden=!completed.length;
 if(!completed.length)return;
 const counts={};for(const rec of completed)for(const score of rec.result.results){const c=counts[score.model]??={name:score.name,ok:0,not_ok:0,disabled:0};c[score.decision]++;}
 $('batch-summary').innerHTML=`<strong>Оценено ${completed.length} фотографий</strong><br>`+Object.values(counts).map(c=>`${esc(c.name)}: годен — ${c.ok}, не годен — ${c.not_ok}${c.disabled?`, без автодопуска — ${c.disabled}`:''}`).join('<br>');
}
async function evaluate(){
 if(running||!config||!files.length)return;
 const models=selectedModels();if(!models.length){status('Выбери хотя бы одну модель.',true);return;}
 const chosen=files.slice(),mode=$('inference-mode').value,resize=$('resize-input').checked;
 const threshold=$('manual-threshold').value;
 setBusy(true);controller=new AbortController();let done=0,errors=0;
 for(const file of chosen){
  if(controller.signal.aborted)break;
  const rec={id:++nextId,file,url:URL.createObjectURL(file),result:null,error:null};records.push(rec);
  const container=document.createElement('article');container.className='upload-result';container.dataset.record=String(rec.id);$('upload-results').append(container);renderRecord(rec);
  status(`Оцениваю ${done+1} из ${chosen.length}: ${file.name}`);
  try{
   const query=new URLSearchParams({models:models.join(','),mode,resize:resize?'1':'0'});if(mode==='manual')query.set('threshold',threshold);
   const response=await fetch('/api/inference/predict?'+query.toString(),{method:'POST',headers:{'Content-Type':'application/octet-stream','X-Inference-Token':config.token},body:file,signal:controller.signal});
   let value;try{value=await response.json();}catch{throw Error('Сервер вернул неожиданный ответ. Проверь, что клиент запущен.');}
   if(!response.ok)throw Error(value.error||'Не удалось оценить фотографию');rec.result=value;
  }catch(error){rec.error=error.name==='AbortError'?'Оценка отменена.':String(error.message||error);errors++;}
  done++;renderRecord(rec);summary();
 }
 const stopped=controller.signal.aborted;controller=null;setBusy(false);
 status(`${stopped?'Остановлено.':'Готово.'} Обработано ${done} из ${chosen.length}; ошибок ${errors}.`,errors>0);
}
function openPreview(record){
 $('upload-detail-title').textContent=record.file.name;$('upload-detail-image').src=record.url;$('upload-detail-image').classList.remove('native');
 $('upload-detail-info').innerHTML=record.result.results.map(s=>`<b>${esc(s.name)}</b>: P(годен) ${percent(s.p_ok)} · ${decisions[s.decision]}`).join('<br>');$('upload-detail').showModal();
}
function clearAll(){if(running)return;records.forEach(r=>URL.revokeObjectURL(r.url));records=[];files=[];$('photo-files').value='';$('upload-results').innerHTML='';summary();setBusy(false);status('Выбери фотографии для оценки.');}
function saveCSV(){
 const rows=[['file','width','height','format','resized','model','p_ok','threshold','decision','mode','analysis_ms','known_training_image']];
 for(const record of records)if(record.result){const r=record.result;for(const score of r.results)rows.push([record.file.name,r.image.width,r.image.height,r.image.format,r.image.resized,score.model,score.p_ok,score.threshold,score.decision,r.mode,score.analysis_ms,r.known_image?.in_training??false]);}
 const text='\ufeff'+rows.map(r=>r.map(v=>'"'+String(v).replace(/"/g,'""')+'"').join(',')).join('\r\n');
 const link=document.createElement('a'),url=URL.createObjectURL(new Blob([text],{type:'text/csv;charset=utf-8'}));link.href=url;link.download='photo_quality_results.csv';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}
async function init(){
 modeHelp();$('inference-mode').onchange=modeHelp;$('manual-threshold').oninput=modeHelp;
 $('photo-files').onchange=e=>addFiles(e.target.files);$('evaluate').onclick=evaluate;$('clear').onclick=clearAll;$('save-csv').onclick=saveCSV;$('cancel').onclick=()=>controller?.abort();
 const drop=$('drop-zone');drop.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();$('photo-files').click();}});
 for(const type of ['dragenter','dragover'])drop.addEventListener(type,e=>{e.preventDefault();drop.classList.add('drag');});
 for(const type of ['dragleave','drop'])drop.addEventListener(type,e=>{e.preventDefault();drop.classList.remove('drag');if(type==='drop')addFiles(e.dataTransfer.files);});
 $('upload-close').onclick=()=>$('upload-detail').close();$('upload-fit').onclick=()=>$('upload-detail-image').classList.remove('native');$('upload-native').onclick=()=>$('upload-detail-image').classList.add('native');
 try{
  const response=await fetch('/api/inference/config');const value=await response.json();if(!response.ok)throw Error(value.error||'Не удалось загрузить модели');config=value;
  const markup=m=>`<label class="model-option"><input name="upload-model" type="checkbox" value="${m.id}" ${['neural','trees_extended'].includes(m.id)?'checked':''}><span><strong>${esc(m.name)}</strong><small>AUROC в эксперименте ${m.auroc.toFixed(3)}${m.thresholds.precise>1?' · строгий допуск отключён':''}</small></span></label>`;
  $('primary-models').innerHTML=config.models.filter(m=>['neural','trees_extended'].includes(m.id)).map(markup).join('');$('other-models').innerHTML=config.models.filter(m=>!['neural','trees_extended'].includes(m.id)).map(markup).join('');
  document.querySelectorAll('input[name="upload-model"]').forEach(n=>n.onchange=()=>setBusy(false));setBusy(false);status('Готово. Выбери фотографии для оценки.');
 }catch(error){status(String(error.message||error),true);}
}
init();
