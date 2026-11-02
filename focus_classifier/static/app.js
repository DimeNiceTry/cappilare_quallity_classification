'use strict';
const $=id=>document.getElementById(id);
const pct=x=>x==null?'—':`${(100*x).toFixed(1)}%`;
const num=x=>x==null?'—':x.toFixed(3);
const escapeText=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const featureNames={laplacian_raw:'Лапласиан',laplacian_smooth:'Лапласиан σ1',tenengrad_smooth:'Tenengrad',local_tenengrad_p75:'Локальный p75'};
const labelName=l=>({ok:'годен',not_ok:'не годен',uncertain:'не уверен'}[l]||l);
let data,speed={},page=0,selected=[],pageSize=24;
function threshold(row,model){const s=row.scores[model];return $('mode').value==='fixed'?Number($('threshold').value):$('mode').value==='precise'?s.precise_threshold:s.threshold;}
function accepted(row,model){return row.scores[model].p>=threshold(row,model);}
function errorName(row,m){if(row.truth<0)return '';return accepted(row,m)?(row.truth===0?'Ложный допуск':''):(row.truth===1?'Потерян хороший':'');}
function renderTable(){
 $('models').innerHTML=data.models.map(m=>{const r=data.report.models[m.id],s=speed.models?.[m.id];return `<tr data-model="${m.id}" class="${m.id===$('model').value?'selected':''}"><td>${escapeText(m.name)}</td><td>${num(r.calibrated.auroc)}</td><td>${num(r.calibrated.average_precision)}</td><td>${pct(r.balanced.precision_ok)}</td><td>${r.balanced.tp} / 62</td><td>${num(r.calibrated.ece_ok['10'])}</td><td>${s?s.end_to_end_ms_p50.toFixed(1):'—'}</td></tr>`;}).join('');
 $('models').querySelectorAll('tr').forEach(tr=>tr.onclick=()=>{$('model').value=tr.dataset.model;page=0;render();});
}
function render(){
 const m=$('model').value,b=$('compare').value,ds=$('dataset').value,v=$('view').value,t=$('truth').value,q=$('search').value.trim().toLowerCase();
 $('threshold-box').hidden=$('mode').value!=='fixed';$('threshold-value').value=Number($('threshold').value).toFixed(2);
 const evaluated=data.rows.filter(r=>r.truth>=0);let tp=0,fp=0,fn=0,tn=0;
 for(const r of evaluated){if(accepted(r,m)){if(r.truth===1)tp++;else fp++;}else if(r.truth===1)fn++;else tn++;}
 $('stats').innerHTML=[['good',pct(tp+fp?tp/(tp+fp):null),`годных среди ${tp+fp} принятых`],['good',`${tp} / 62`,'хороших кадров найдено'],['bad',String(fp),'плохих кадров допущено'],['',String(fn),'хороших кадров отклонено']].map(([c,n,l])=>`<div class="stat ${c}"><strong>${n}</strong><span>${l}</span></div>`).join('');
 selected=data.rows.filter(r=>{
  if(ds==='consensus'&&r.truth<0||ds==='unresolved'&&r.truth>=0)return false;
  if(t!=='all'&&r.truth!==Number(t))return false;
  if(q&&!`${r.file} ${r.patient}`.toLowerCase().includes(q))return false;
  const a=accepted(r,m);
  if(v==='accepted'&&!a||v==='rejected'&&a)return false;
  if(v==='false_accept'&&!(r.truth===0&&a)||v==='false_reject'&&!(r.truth===1&&!a))return false;
  if(v==='different'&&(!b||a===accepted(r,b)))return false;
  return true;
 });
 const order=$('sort').value;
 selected.sort((a,b)=>order==='file'?a.file.localeCompare(b.file):order==='asc'?a.scores[m].p-b.scores[m].p:order==='border'?Math.abs(a.scores[m].p-threshold(a,m))-Math.abs(b.scores[m].p-threshold(b,m)):b.scores[m].p-a.scores[m].p);
 const pages=Math.max(1,Math.ceil(selected.length/pageSize));page=Math.min(page,pages-1);
 $('selection').textContent=`${selected.length} фотографий в отборе`;
 $('context').textContent=`Показатели выше рассчитаны на всех 565 согласованных кадрах при текущем пороге. ${$('mode').value==='fixed'?'Порог выбран вручную после проверки.':'Каждая внешняя часть использует свой порог, выбранный на внутренних пациентах.'}`;
 $('page').textContent=`${page+1} / ${pages}`;$('prev').disabled=page===0;$('next').disabled=page>=pages-1;
 $('gallery').innerHTML=selected.slice(page*pageSize,(page+1)*pageSize).map(r=>{
 const s=r.scores[m],a=accepted(r,m),e=errorName(r,m),labels=$('labels').checked;
 return `<button class="card" data-file="${escapeText(r.file)}" aria-label="Открыть ${escapeText(r.file)}"><div class="image"><img src="/image/${encodeURIComponent(r.file)}" alt="Кадр капилляроскопии" loading="lazy"><span class="badge ${a?'good':'bad'}">Модель: ${a?'годен':'не годен'}</span></div><div class="card-content"><div class="prob"><strong>${pct(s.p)}</strong><span>P(годен) · порог ${threshold(r,m)>1?'нет допуска':pct(threshold(r,m))}</span></div><div class="bar"><i style="width:${100*s.p}%"></i></div><div class="filename">${escapeText(r.file)}</div>${labels?`<div class="truth"><span>${r.truth<0?'Нет согласованной метки':`Согласовано: ${r.truth?'годен':'не годен'}`}</span><span class="error">${e}</span></div>`:''}${b?`<div class="compare">Вторая модель: ${pct(r.scores[b].p)} · ${accepted(r,b)?'годен':'не годен'}</div>`:''}</div></button>`;
 }).join('')||'<div class="empty">В этом отборе нет фотографий. Измени фильтр или порог.</div>';
 $('gallery').querySelectorAll('.card').forEach(c=>c.onclick=()=>openDetail(data.rows.find(r=>r.file===c.dataset.file)));
 renderTable();
}
function openDetail(r){
 const m=$('model').value,b=$('compare').value;$('detail-title').textContent=r.file;$('detail-image').src='/image/'+encodeURIComponent(r.file);$('detail-image').classList.remove('native');
 $('detail-info').innerHTML=`<b>Основная модель:</b> P(годен) ${pct(r.scores[m].p)}; решение — ${accepted(r,m)?'годен':'не годен'}. ${b?`<b>Вторая:</b> ${pct(r.scores[b].p)}; ${accepted(r,b)?'годен':'не годен'}.`:''}<br>${$('labels').checked?`<b>Человек:</b> ${labelName(r.human)} · <b>Ассистент:</b> ${labelName(r.assistant)}<br>${escapeText(r.comment)}<br>`:''}<span class="hint">Пациент ${escapeText(r.patient)} · проверочная часть ${r.fold+1} · исходный размер 1024×768</span><div class="features">${Object.entries(r.features).map(([k,v])=>`<span>${featureNames[k]}: ${v.toExponential(4)}</span>`).join('')}</div>`;
 $('detail').showModal();
}
function exportSelection(){
 const m=$('model').value,b=$('compare').value;const rows=[['file','patient','consensus','human','assistant','model','p_ok','threshold','predicted','compare_model','compare_p_ok','compare_predicted']];
 for(const r of selected)rows.push([r.file,r.patient,r.truth<0?'':r.truth?'ok':'not_ok',r.human,r.assistant,m,r.scores[m].p,threshold(r,m),accepted(r,m)?'ok':'not_ok',b,b?r.scores[b].p:'',b?(accepted(r,b)?'ok':'not_ok'):'']);
 const csv='\ufeff'+rows.map(r=>r.map(x=>'"'+String(x).replace(/"/g,'""')+'"').join(',')).join('\r\n');
 const url=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'}));const a=document.createElement('a');a.href=url;a.download='selection_'+m+'.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}
async function init(){
 try{
  const responses=await Promise.all([fetch('/artifacts/gallery.json'),fetch('/artifacts/speed.json')]);
  if(!responses[0].ok)throw Error('gallery.json недоступен');data=await responses[0].json();if(responses[1].ok)speed=await responses[1].json();
  const opts=data.models.map(m=>`<option value="${m.id}">${escapeText(m.name)}</option>`).join('');$('model').innerHTML=opts;$('compare').innerHTML='<option value="">Без сравнения</option>'+opts;
  $('model').value='trees_extended';$('compare').value=data.models.some(m=>m.id==='neural')?'neural':'trees_basic';
  for(const id of ['model','compare','mode','threshold','dataset','view','truth','sort','search','labels'])$(id).addEventListener(id==='search'||id==='threshold'?'input':'change',()=>{page=0;render();});
  $('prev').onclick=()=>{page--;render();$('selection').scrollIntoView({block:'start'});};$('next').onclick=()=>{page++;render();$('selection').scrollIntoView({block:'start'});};
  $('close').onclick=()=>$('detail').close();$('fit').onclick=()=>$('detail-image').classList.remove('native');$('native').onclick=()=>$('detail-image').classList.add('native');$('export-selection').onclick=exportSelection;
  $('detail').addEventListener('click',e=>{if(e.target===$('detail'))$('detail').close();});render();
 }catch(e){$('selection').textContent='Не удалось загрузить результаты';$('context').textContent=String(e);}
}
init();
