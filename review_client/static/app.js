"use strict";
const $ = id => document.getElementById(id);
const names = {ok:"Годен",not_ok:"Не годен",uncertain:"Не уверен"};
let state, index = 0, busy = false, ready = false, scale = 1, x = 0, y = 0, dragging = null, lastVisited = [];
let errorTimer;
function error(message) { $("error").textContent=message; $("error").hidden=false; clearTimeout(errorTimer); errorTimer=setTimeout(()=>$("error").hidden=true,10000); }
async function api(path,data) {
  const options=data===undefined?{}:{method:"POST",headers:{"Content-Type":"application/json","X-Review-Token":state.token},body:JSON.stringify(data)};
  const response=await fetch(path,options); const body=await response.json();
  if(!response.ok) throw new Error(body.error||"Ошибка соединения"); return body;
}
function progress() {
 const p=state.progress; $("progressText").textContent=`Оценено ${p.human.done} из ${p.human.total}`;
 $("assistantProgress").textContent=`Ассистент: ${p.assistant.done} / ${p.assistant.total}`;
 $("progress").max=p.human.total; $("progress").value=p.human.done;
 $("showResults").disabled=!p.complete; $("finished").hidden=p.human.done!==p.human.total;
 $("totalFrames").textContent=`из ${state.items.length}`; $("jump").max=state.items.length;
}
function transform() { $("photo").style.transform=`translate(${x}px,${y}px) scale(${scale})`; $("zoomText").textContent=Math.round(scale*100)+"%"; }
function fit() { if(!ready)return; const v=$("viewport"),im=$("photo"); scale=Math.min(v.clientWidth/im.naturalWidth,v.clientHeight/im.naturalHeight); x=(v.clientWidth-im.naturalWidth*scale)/2;y=(v.clientHeight-im.naturalHeight*scale)/2;transform(); }
function zoomTo(next,px,py) { if(!ready)return; const v=$("viewport");px=px??v.clientWidth/2;py=py??v.clientHeight/2;next=Math.max(.1,Math.min(8,next));x=px-(px-x)*next/scale;y=py-(py-y)*next/scale;scale=next;transform(); }
function decisionsEnabled() { document.querySelectorAll("[data-label]").forEach(b=>b.disabled=busy||!ready); }
function show(i,remember=true) {
 if(!state.items.length)return;
 if(remember&&i!==index)lastVisited.push(index);
 index=Math.max(0,Math.min(state.items.length-1,i));const item=state.items[index];
 ready=false; decisionsEnabled(); $("loading").hidden=false;$("photo").style.visibility="hidden";
 $("imageTitle").textContent=`Кадр ${String(index+1).padStart(3,"0")}`;$("jump").value=index+1;
 $("ownLabel").textContent=item.label?"Ваша оценка: "+names[item.label]:"Не оценено";
 $("comment").value=item.comment;document.querySelectorAll("[data-reason]").forEach(b=>b.classList.toggle("selected",item.reasons.includes(b.dataset.reason)));
 $("original").href="/image/"+item.id;
 const expected=item.id;
 $("photo").onload=()=>{if(state.items[index].id!==expected)return;ready=true;$("loading").hidden=true;$("photo").style.visibility="visible";$("dimensions").textContent=`${$("photo").naturalWidth} × ${$("photo").naturalHeight}`;fit();decisionsEnabled();};
 $("photo").onerror=()=>{ready=false;$("loading").textContent="Не удалось открыть изображение";error("Не удалось открыть PNG. Обновите страницу.");};
 $("photo").src="/image/"+item.id;
}
function nextIndex() {
 for(let step=1;step<=state.items.length;step++){const i=(index+step)%state.items.length;if(!$("pendingOnly").checked||!state.items[i].label)return i;}return -1;
}
async function save(label) {
 if(busy||!ready)return;busy=true;decisionsEnabled(); const item=state.items[index];
 const reasons=[...document.querySelectorAll("[data-reason].selected")].map(b=>b.dataset.reason);
 $("saveStatus").textContent="Сохраняем…";
 try{
  const result=await api("/api/review",{id:item.id,label,reasons,comment:$("comment").value,revision:item.revision});
  Object.assign(item,{label,reasons,comment:$("comment").value,revision:result.saved.revision});state.progress=result.progress;progress();
  $("saveStatus").textContent="✓ Сохранено: "+names[label];
  const next=nextIndex();if(next>=0&&next!==index)show(next);else $("ownLabel").textContent="Ваша оценка: "+names[label];
 }catch(e){error(e.message);$("saveStatus").textContent="Не сохранено. Ваш ответ остался на экране.";}
 finally{busy=false;decisionsEnabled();}
}
async function results() {
 try{const r=await api("/api/results"),c=r.counts;
 const root=$("resultContent");root.replaceChildren();
 const cards=document.createElement("div");cards.className="result-cards";
 for(const [count,text] of [[c.agreed_ok,"Совпали: годен"],[c.agreed_not_ok,"Совпали: не годен"],[c.disagreement,"Расхождения"]]){const card=document.createElement("div");card.className="result-card";const number=document.createElement("strong");number.textContent=count;card.append(number,document.createTextNode(text));cards.append(card);}root.append(cards);
 const p=document.createElement("p");p.textContent=`Совпали «не уверен»: ${c.uncertain_agreement}. Они сохраняются отдельно и не входят в обучающий набор.`;p.className="small";root.append(p);
 const table=document.createElement("table");const header=document.createElement("tr");for(const h of ["Вы ↓ / ассистент →",...r.class_order.map(k=>names[k])]){const th=document.createElement("th");th.textContent=h;header.append(th);}table.append(header);
 r.matrix_human_rows_assistant_columns.forEach((row,i)=>{const tr=document.createElement("tr");const th=document.createElement("th");th.textContent=names[r.class_order[i]];tr.append(th);row.forEach(n=>{const td=document.createElement("td");td.textContent=n;tr.append(td);});table.append(tr);});root.append(table);
 $("results").showModal();
 }catch(e){error(e.message);}
}
document.querySelectorAll("[data-label]").forEach(b=>b.onclick=()=>save(b.dataset.label));
document.querySelectorAll("[data-reason]").forEach(b=>b.onclick=()=>b.classList.toggle("selected"));
$("fit").onclick=fit;$("native").onclick=()=>zoomTo(1);$("zoom2").onclick=()=>zoomTo(2);
$("fullscreen").onclick=()=>document.fullscreenElement?document.exitFullscreen():document.querySelector(".viewer-wrap").requestFullscreen();
document.addEventListener("fullscreenchange",()=>setTimeout(fit,100));
$("viewport").addEventListener("wheel",e=>{e.preventDefault();const r=$("viewport").getBoundingClientRect();zoomTo(scale*Math.exp(-e.deltaY*.0015),e.clientX-r.left,e.clientY-r.top);},{passive:false});
$("viewport").addEventListener("pointerdown",e=>{if(!ready)return;dragging={id:e.pointerId,px:e.clientX,py:e.clientY,x,y};$("viewport").setPointerCapture(e.pointerId);$("viewport").classList.add("dragging");});
$("viewport").addEventListener("pointermove",e=>{if(!dragging)return;x=dragging.x+e.clientX-dragging.px;y=dragging.y+e.clientY-dragging.py;transform();});
function release(){dragging=null;$("viewport").classList.remove("dragging");}$("viewport").addEventListener("pointerup",release);$("viewport").addEventListener("pointercancel",release);
$("next").onclick=()=>{if(!busy){const next=nextIndex();if(next>=0)show(next);}};
$("prev").onclick=()=>{if(!busy)show(lastVisited.length?lastVisited.pop():Math.max(0,index-1),false);};
$("jump").onchange=()=>{if(!busy)show(Number($("jump").value)-1);};
$("pendingOnly").onchange=()=>{if($("pendingOnly").checked&&state.items[index].label){const next=nextIndex();if(next>=0)show(next);}};
$("reviewAll").onclick=()=>{$("pendingOnly").checked=false;show(0);};
$("showResults").onclick=results;$("closeResults").onclick=()=>$("results").close();
$("export").onclick=async()=>{$("export").disabled=true;$("exportStatus").textContent="Собираем набор…";try{const r=await api("/api/export",{include_images:$("copyImages").checked});$("exportStatus").textContent=`Готово: ${r.count} кадров. Папка: ${r.path}`;}catch(e){$("exportStatus").textContent=e.message;}finally{$("export").disabled=false;}};
$("backup").onclick=async()=>{try{const r=await api("/api/backup"),blob=new Blob([JSON.stringify(r,null,2)],{type:"application/json"});const url=URL.createObjectURL(blob),a=document.createElement("a");a.href=url;a.download="my_reviews_"+new Date().toISOString().slice(0,10)+".json";a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}catch(e){error(e.message);}};
document.addEventListener("keydown",e=>{if(e.repeat||!state||$("results").open||e.ctrlKey||e.altKey||e.metaKey||["INPUT","TEXTAREA","SELECT"].includes(e.target.tagName))return;
 const keys={"1":"ok","2":"uncertain","3":"not_ok"};if(keys[e.key]){e.preventDefault();save(keys[e.key]);}
 else if(e.key==="ArrowRight"){e.preventDefault();$("next").click();}else if(e.key==="ArrowLeft"){e.preventDefault();$("prev").click();}else if(e.key==="0"){fit();}else if(e.key==="f"||e.key==="а"){$("fullscreen").click();}});
window.addEventListener("beforeunload",e=>{if(busy){e.preventDefault();e.returnValue="";}});
new ResizeObserver(()=>{if(ready)fit();}).observe($("viewport"));
(async()=>{try{state=await api("/api/state");progress();const first=state.items.findIndex(i=>!i.label);show(first>=0?first:0,false);if(first<0)$("pendingOnly").checked=false;}catch(e){error(e.message);}})();
setInterval(async()=>{if(!state||busy||document.hidden)return;try{const fresh=await api("/api/state");state.progress=fresh.progress;progress();}catch{}},15000);
