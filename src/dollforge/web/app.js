'use strict';
const $ = s => document.querySelector(s);
const $$ = s => [...document.querySelectorAll(s)];
const esc = v => String(v ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const labels = {front:'Frente',back:'Costas',left:'Esquerda',right:'Direita'};
const names = {head:'Cabeça',torso:'Torso',pelvis:'Pelve',arm:'Braço',leg:'Perna',footwear:'Calçado',hand:'Mão',foot:'Pé',upper_arm:'Braço superior',forearm:'Antebraço',thigh:'Coxa',shin:'Canela',hair:'Cabelo',face:'Rosto',top:'Roupa superior',bottom:'Roupa inferior',accessory:'Acessório'};
const sides = {left:'esquerdo',right:'direito',center:'central',unknown:'indefinido',bilateral:'bilateral'};
const states = {queued:'Na fila',running:'Em execução',succeeded:'Concluído',failed:'Falhou',waiting_for_review:'Aguardando revisão',approved:'Aprovado',needs_review:'Revisar',corrected:'Corrigido',rejected:'Rejeitado',unreviewed:'Sem revisão'};
const stageNames = {S03:'Câmeras e orientação',S05:'Segmentação de peças',S06:'Correspondência multi-view',S07:'Alinhamento de escala',S07C:'Calibração física das vistas',S08:'DollGraph',S09:'Perception Graph',S09V:'Volumetria calibrada + SDF',S10:'Reconstrução por peça',S15:'Projeto Blender',S16:'Validação geométrica'};
const colors = ['#c6e397','#85b9bc','#d8ae7d','#c394ad','#859dd1','#b6c37a','#d1917e','#7db598','#a9a2cc'];
const state = {projects:[],project:null,views:[],runs:[],run:null,segments:[],matches:null,graph:null,perception:null,calibration:null,volumetry:null,reconstruction:null,report:null,activeView:null,selected:null,tab:'workspace',tool:'inspect',dirty:false,meshes:[]};
let pollTimer, toastTimer, uploadLabel, baseImage, baseImageData, edgeData,
  maskLayer=document.createElement('canvas'), undo=[], paint=false, lastPoint=null,
  showEdges=false, lassoPoints=[], lassoPath=[];

async function api(path, options={}) {
  const headers = options.body && !(options.body instanceof FormData) ? {'Content-Type':'application/json'} : {};
  const response = await fetch('/api'+path,{...options,headers:{...headers,...options.headers}});
  if (!response.ok) {
    const error = await response.json().catch(()=>({detail:'Falha de conexão'}));
    throw new Error(typeof error.detail === 'string' ? error.detail : error.detail.map?.(e=>e.msg).join('; ') || 'Dados inválidos');
  }
  return response.json();
}
function toast(message,error=false){const el=$('#toast');el.textContent=message;el.className=error?'error':'';el.hidden=false;clearTimeout(toastTimer);toastTimer=setTimeout(()=>el.hidden=true,6000);}
async function action(fn){try{await fn();}catch(e){toast(e.message,true);}}
const content = id => api('/artifacts/'+id+'/content');
const artifactUrl = id => '/api/artifacts/'+id+'/content';
const partName = p => `${names[p.class]||p.class} · ${sides[p.side]||p.side}`;
const currentSegment = () => state.segments.find(s=>s.view_id===state.activeView);
const selectedObservation = () => currentSegment()?.observations.find(o=>o.observation_id===state.selected);
const node = code => state.run?.stages.find(s=>s.stage===code && !s.invalidated && s.output_artifact_id);

async function init(){
  await action(async()=>{const health=await api('/health');$('#health').textContent=health.blender==='unavailable'?'Local · Blender ausente':'Ambiente conectado';await loadProjects();});
  draw3D();
}
async function loadProjects(selectId){
  state.projects=await api('/projects');
  $('#project-count').textContent=state.projects.length;
  if(state.projects.length) await selectProject(selectId || localStorage.getItem('dollforge.project') || state.projects[0].project_id);
  else render();
}
async function selectProject(id){
  clearTimeout(pollTimer);
  state.project=state.projects.find(p=>p.project_id===id)||state.projects[0];
  localStorage.setItem('dollforge.project',state.project.project_id);
  state.views=await api(`/projects/${state.project.project_id}/views`);
  state.runs=await api(`/projects/${state.project.project_id}/runs`);
  state.activeView=state.views.find(v=>v.label==='front')?.view_id||state.views[0]?.view_id;
  state.selected=null;
  await selectRun(state.runs[0]||null);
  render();
}
async function selectRun(run){
  state.run=run;state.segments=[];state.matches=null;state.graph=null;state.perception=null;state.calibration=null;state.volumetry=null;state.reconstruction=null;state.report=null;state.meshes=[];
  if(run){
    const outputs=await Promise.all(run.stages.filter(s=>s.output_artifact_id&&!s.invalidated).map(async s=>({stage:s,data:await content(s.output_artifact_id)})));
    for(const {stage:s,data} of outputs){
      if(s.stage==='S05')state.segments.push({...data,artifact_id:s.output_artifact_id,view_id:data.observations[0]?.view_id});
      if(s.stage==='S06')state.matches={...data,artifact_id:s.output_artifact_id};
      if(s.stage==='S07C')state.calibration={...data,artifact_id:s.output_artifact_id};
      if(s.stage==='S08')state.graph={...data,artifact_id:s.output_artifact_id};
      if(s.stage==='S09')state.perception={...data,artifact_id:s.output_artifact_id};
      if(s.stage==='S09V')state.volumetry={...data,artifact_id:s.output_artifact_id};
      if(s.stage==='S10')state.reconstruction=data;
      if(s.stage==='S16')state.report=data;
    }
    if(state.reconstruction)state.meshes=await Promise.all(state.reconstruction.meshes.map(m=>content(m.preview_artifact_id)));
    if(['queued','running'].includes(run.status))pollTimer=setTimeout(()=>action(pollRun),1200);
  }
  render();
  if(state.tab==='segmentation')await loadMask();
  if(state.tab==='history')await renderHistory();
  draw3D();
}
async function pollRun(){
  if(!state.run)return;
  const run=await api('/runs/'+state.run.run_id);
  await selectRun(run);
  if(!['queued','running'].includes(run.status)){
    state.runs=await api(`/projects/${state.project.project_id}/runs`);
    const message=run.error||'Peças geradas. Seu projeto está pronto para revisão.';
    toast(message,run.status==='failed'||!!run.error);
    render();
  }
}
function setTab(tab){
  state.tab=tab;
  $$('.tab-panel').forEach(p=>p.hidden=p.id!=='tab-'+tab);
  $$('[data-tab]').forEach(b=>b.classList.toggle('active',b.dataset.tab===tab));
  if(tab==='segmentation')action(loadMask);
  if(tab==='history')action(renderHistory);
  if(tab==='geometry')requestAnimationFrame(draw3D);
}
function render(){
  const project=state.project,run=state.run;
  $('#project-title').textContent=project?.name||'Seu próximo personagem.';
  $('#breadcrumb').textContent=project?.name||'Projetos';
  $('#project-subtitle').textContent=project?`${project.style_family||'Estilo livre'} · Reconstrução multi-view supervisionada`:'Um espaço para transformar vistas em peças, com você no controle.';
  $('#project-list').innerHTML=state.projects.map(p=>`<button class="project-entry ${p.project_id===project?.project_id?'selected':''}" data-project="${p.project_id}">${esc(p.name)}</button>`).join('');
  $('#views-metric').innerHTML=`${state.views.length} <em>/ 4</em>`;
  $('#parts-metric').textContent=state.matches?.parts.length??'—';
  $('#scale-metric').textContent=project?.known_height_mm?`${project.known_height_mm} mm`:'Relativa';
  $('#run-metric').textContent=run?states[run.status]:(state.views.length===4?'Pronto para executar':'Aguardando vistas');
  $('#run-button').disabled=!project||state.views.length<4||['queued','running'].includes(run?.status);
  $('#run-button').innerHTML=['queued','running'].includes(run?.status)?'◌ Processando…':'▷ Executar pipeline';
  $('#review-count').textContent=state.segments.reduce((a,s)=>a+s.observations.filter(o=>o.review_state!=='approved').length,0)||'';
  const invalid=run?.stages.some(s=>s.invalidated);
  const changedInputs=run&&JSON.stringify([...run.project_snapshot.views].sort())!==JSON.stringify([...project.views].sort());
  const note=run?.error||(invalid?'Há correções salvas. Execute o pipeline para atualizar apenas as etapas dependentes.':changedInputs?'As vistas foram alteradas após esta execução. Execute o pipeline para usar as novas imagens.':'');
  $('#notice').hidden=!note;$('#notice').textContent=note;
  $('#views-grid').innerHTML=Object.entries(labels).map(([label,name],i)=>{
    const view=state.views.find(v=>v.label===label);
    return `<article class="view-card"><div class="view-card-top"><span>${String(i+1).padStart(2,'0')} <span class="muted">/</span> ${name}</span><small>${view?'● Importada':'○ Pendente'}</small></div><button class="view-image" data-upload="${label}" aria-label="Importar vista ${name}" ${!project?'disabled':''}>${view?`<img src="${artifactUrl(view.normalized_artifact_id)}" alt="Vista ${name}">`:`<span class="upload-symbol">＋</span><strong>Adicionar ${name.toLowerCase()}</strong><small>Clique para selecionar uma imagem</small>`}</button><div class="view-card-bottom"><span>${view?`${view.qa.width} × ${view.qa.height} px`:'VISTA OBRIGATÓRIA'}</span><button data-upload="${label}" ${!project?'disabled':''}>${view?'Substituir ↗':'Importar ↗'}</button></div></article>`;
  }).join('');
  const viewOptions=(run?.views.length?run.views:state.views);
  $('#seg-view').innerHTML=viewOptions.map(v=>`<option value="${v.view_id}" ${v.view_id===state.activeView?'selected':''}>${esc(labels[v.label]||v.label)}</option>`).join('');
  if(!viewOptions.some(v=>v.view_id===state.activeView))state.activeView=viewOptions[0]?.view_id;
  $('#run-select').innerHTML=state.runs.map(r=>`<option value="${r.run_id}" ${r.run_id===run?.run_id?'selected':''}>${new Date(r.created_at).toLocaleString('pt-BR')} · ${r.run_id.slice(0,8)}</option>`).join('');
  renderPipeline();renderObservations();renderMatches();renderPerception();renderVolumetry();renderValidation();
  $('#download').disabled=!run||invalid||['queued','running'].includes(run.status);
  $('#replay').disabled=!run||invalid||['queued','running'].includes(run.status);
  $('#final-review').disabled=!node('S16')||invalid;
}
function renderPipeline(){
  if(!state.run){$('#pipeline-content').innerHTML='Importe as quatro vistas para iniciar.';return;}
  $('#pipeline-content').innerHTML=state.run.stages.map(s=>{
    const quality=s.quality_status==='passed'
      ?`<span class="quality-proof pass">LIMITE ✓ · ${Math.round((s.quality_score||0)*100)}% · ${s.quality_attempts} tentativa(s)</span>`
      :s.quality_status==='retrain_candidate'
        ?`<span class="quality-proof failed">LIMITE ✕ · RETREINO CANDIDATO · ${s.quality_attempts} tentativa(s)</span>`
        :'';
    const trace=s.quality_trace_artifact_id
      ?`<a href="${artifactUrl(s.quality_trace_artifact_id)}" target="_blank" rel="noopener">Prova do limite ↗</a>`
      :'';
    const retrain=s.training_signal_artifact_id
      ?`<a href="${artifactUrl(s.training_signal_artifact_id)}" target="_blank" rel="noopener">Sinal de retreino ↗</a>`
      :'';
    return `<div class="pipeline-row"><span class="stage-number">${s.stage}</span><div>${stageNames[s.stage]||s.stage}<p>${s.stage==='S05'?esc(labels[state.run.views.find(v=>s.node_id.includes(v.view_id))?.label]||''):''} ${s.cached?'Reutilizado do cache · ':''}${s.output_artifact_id?.slice(0,8)||'Processando'}</p>${quality}${s.error?`<p>${esc(s.error)}</p>`:''}</div><span class="status ${s.invalidated?'warn':s.status==='failed'?'failed':''}">${s.invalidated?'↻ Reprocessar':states[s.status]}</span><div class="pipeline-links">${s.output_artifact_id?`<a href="${artifactUrl(s.output_artifact_id)}" target="_blank" rel="noopener">Contrato ↗</a>`:''}${trace}${retrain}</div></div>`;
  }).join('')||'Execução na fila…';
}
function renderObservations(){
  const segment=currentSegment();
  $('#observation-count').textContent=segment?.observations.length||0;
  if(segment&&!segment.observations.some(o=>o.observation_id===state.selected))state.selected=segment.observations[0]?.observation_id;
  $('#observations').innerHTML=segment?.observations.map((o,i)=>`<div class="observation ${o.observation_id===state.selected?'selected':''}" role="button" tabindex="0" data-observation="${o.observation_id}"><span class="swatch" style="background:${colors[i%colors.length]}"></span><div><strong>${esc(partName(o))}</strong><small>${states[o.review_state]}</small></div><span class="confidence">${Math.round(o.confidence*100)}%</span></div>`).join('')||'<p class="muted">Nenhuma proposta disponível.</p>';
  const o=selectedObservation();
  $('#part-details').innerHTML=o?`<label>Classe semântica<select id="part-class">${Object.entries(names).map(([k,v])=>`<option value="${k}" ${k===o.class?'selected':''}>${v}</option>`).join('')}</select></label><label>Lado do personagem<select id="part-side">${Object.entries(sides).map(([k,v])=>`<option value="${k}" ${k===o.side?'selected':''}>${v}</option>`).join('')}</select></label><button id="relabel" class="button secondary small full">Salvar rótulo</button><p>Origem: ${esc(o.provenance.type)}<br>${esc(o.provenance.note||o.provenance.source)}</p><div class="button-row"><button id="approve-part" class="button primary small">✓ Aprovar</button><button id="reject-part" class="button secondary small">Rejeitar</button></div>`:'';
  $('#save-mask').disabled=!state.dirty;
}
function imageLoad(url){return new Promise((resolve,reject)=>{const image=new Image();image.onload=()=>resolve(image);image.onerror=()=>reject(new Error('Não foi possível carregar a imagem'));image.src=url;});}
async function loadMask(){
  renderObservations();const o=selectedObservation(), view=(state.run?.views||state.views).find(v=>v.view_id===state.activeView);
  $('#mask-empty').hidden=!!o;
  const canvas=$('#mask-canvas');
  if(!o||!view){canvas.width=1;canvas.height=1;return;}
  const selected=o.observation_id;
  const [image,mask]=await Promise.all([imageLoad(artifactUrl(view.normalized_artifact_id)),imageLoad(artifactUrl(o.mask_artifact_id))]);
  if(state.selected!==selected)return;
  baseImage=image;canvas.width=image.width;canvas.height=image.height;
  const source=document.createElement('canvas');source.width=image.width;source.height=image.height;
  const sourceContext=source.getContext('2d',{willReadFrequently:true});sourceContext.drawImage(image,0,0);
  baseImageData=sourceContext.getImageData(0,0,image.width,image.height);
  edgeData=DFSeg.edgeMap(baseImageData);
  maskLayer.width=image.width;maskLayer.height=image.height;
  maskLayer.getContext('2d',{willReadFrequently:true}).drawImage(mask,0,0);
  undo=[];state.dirty=false;showEdges=false;$('#toggle-edges').classList.remove('active');
  $('.canvas-wrap').classList.remove('edge-mode');resetLasso();$('#save-mask').disabled=true;
  $('#mask-info').textContent=`${image.width} × ${image.height} · ${partName(o)} · ${Math.round(o.confidence*100)}% de confiança`;
  paintMask();
}
function paintMask(){
  if(!baseImage)return;
  const canvas=$('#mask-canvas'),ctx=canvas.getContext('2d');
  ctx.clearRect(0,0,canvas.width,canvas.height);ctx.drawImage(baseImage,0,0);
  if(showEdges&&edgeData){
    const edgeImage=ctx.createImageData(canvas.width,canvas.height);
    for(let i=0;i<edgeData.length;i++){
      const value=edgeData[i],p=i*4;
      edgeImage.data[p]=105;edgeImage.data[p+1]=225;edgeImage.data[p+2]=194;
      edgeImage.data[p+3]=Math.round(value*.58);
    }
    const edgeCanvas=document.createElement('canvas');edgeCanvas.width=canvas.width;edgeCanvas.height=canvas.height;
    edgeCanvas.getContext('2d').putImageData(edgeImage,0,0);ctx.drawImage(edgeCanvas,0,0);
  }
  const pixels=maskLayer.getContext('2d',{willReadFrequently:true}).getImageData(0,0,maskLayer.width,maskLayer.height);
  for(let i=0;i<pixels.data.length;i+=4){const visible=pixels.data[i]>127;pixels.data[i]=183;pixels.data[i+1]=226;pixels.data[i+2]=133;pixels.data[i+3]=visible?110:0;}
  const overlay=document.createElement('canvas');overlay.width=canvas.width;overlay.height=canvas.height;overlay.getContext('2d').putImageData(pixels,0,0);ctx.drawImage(overlay,0,0);
  if(lassoPath.length){
    ctx.save();ctx.strokeStyle='#d9f3b6';ctx.lineWidth=Math.max(1,canvas.width/500);ctx.setLineDash([6,4]);
    ctx.beginPath();ctx.moveTo(lassoPath[0][0],lassoPath[0][1]);
    for(const point of lassoPath.slice(1))ctx.lineTo(point[0],point[1]);
    ctx.stroke();ctx.setLineDash([]);
    for(const point of lassoPoints){ctx.beginPath();ctx.arc(point[0],point[1],Math.max(3,canvas.width/250),0,Math.PI*2);ctx.fillStyle='#c5e995';ctx.fill();}
    ctx.restore();
  }
}
function position(event,canvas){const box=canvas.getBoundingClientRect();return [(event.clientX-box.left)*canvas.width/box.width,(event.clientY-box.top)*canvas.height/box.height];}
function pushUndo(){
  if(!maskLayer.width||!maskLayer.height)return;
  undo.push(maskLayer.getContext('2d',{willReadFrequently:true}).getImageData(0,0,maskLayer.width,maskLayer.height));
  if(undo.length>12)undo.shift();
}
function markMaskDirty(){state.dirty=true;$('#save-mask').disabled=false;}
function currentBinaryMask(){
  const data=maskLayer.getContext('2d',{willReadFrequently:true}).getImageData(0,0,maskLayer.width,maskLayer.height).data;
  const binary=new Uint8Array(maskLayer.width*maskLayer.height);
  for(let i=0;i<binary.length;i++)binary[i]=data[i*4]>127?1:0;
  return binary;
}
function applyBinaryMask(binary,mode='replace'){
  const ctx=maskLayer.getContext('2d',{willReadFrequently:true});
  const image=ctx.getImageData(0,0,maskLayer.width,maskLayer.height);
  for(let i=0;i<binary.length;i++){
    const current=image.data[i*4]>127,selected=!!binary[i];
    const value=mode==='add'?(current||selected):mode==='subtract'?(current&&!selected):selected;
    const p=i*4,v=value?255:0;image.data[p]=v;image.data[p+1]=v;image.data[p+2]=v;image.data[p+3]=255;
  }
  ctx.putImageData(image,0,0);markMaskDirty();paintMask();
}
function resetLasso(){lassoPoints=[];lassoPath=[];updateLassoButtons();paintMask();}
function updateLassoButtons(){
  const finish=$('#finish-lasso'),cancel=$('#cancel-lasso');
  if(finish)finish.disabled=lassoPoints.length<3;if(cancel)cancel.disabled=lassoPoints.length===0;
}
function toolCursor(){
  const canvas=$('#mask-canvas');
  canvas.style.cursor=state.tool==='inspect'?'default':state.tool==='lasso'?'cell':'crosshair';
}
function smartSelect(event,kind){
  if(!baseImageData||!edgeData)return;
  const point=position(event,$('#mask-canvas'));
  const tolerance=Number($('#region-tolerance').value)+(kind==='wand'?-10:0);
  const sensitivity=Math.min(95,Number($('#edge-sensitivity').value)+(kind==='wand'?16:0));
  pushUndo();
  const region=DFSeg.smartRegion(baseImageData,edgeData,point[0],point[1],{
    tolerance:Math.max(8,tolerance),edgeSensitivity:sensitivity,adaptive:kind!=='wand'
  });
  applyBinaryMask(region,event.altKey?'subtract':'add');
}
function maskOperation(operation){
  if(!baseImageData||!edgeData)return;
  pushUndo();let mask=currentBinaryMask();
  if(operation==='refine'){
    const radius=Math.max(4,Math.round(Number($('#edge-sensitivity').value)/5));
    mask=DFSeg.refineToEdges(mask,edgeData,maskLayer.width,maskLayer.height,radius);
  }else if(operation==='fill'){
    mask=DFSeg.fillHoles(mask,maskLayer.width,maskLayer.height);
  }else{
    mask=DFSeg.morph(mask,maskLayer.width,maskLayer.height,operation,1);
  }
  applyBinaryMask(mask,'replace');
}
function addLassoPoint(event){
  if(!edgeData)return;
  const point=position(event,$('#mask-canvas')).map(Math.round);
  if(lassoPoints.length){
    const previous=lassoPoints[lassoPoints.length-1];
    const segment=DFSeg.magneticPath(edgeData,maskLayer.width,maskLayer.height,previous,point,
      Math.max(8,Math.round(Number($('#edge-sensitivity').value)/3)));
    lassoPath.push(...segment.slice(1));
  }else lassoPath=[point];
  lassoPoints.push(point);updateLassoButtons();paintMask();
}
function finishLasso(){
  if(lassoPoints.length<3)return;
  const first=lassoPoints[0],last=lassoPoints[lassoPoints.length-1];
  const close=DFSeg.magneticPath(edgeData,maskLayer.width,maskLayer.height,last,first,
    Math.max(8,Math.round(Number($('#edge-sensitivity').value)/3)));
  const polygon=[...lassoPath,...close.slice(1)];
  pushUndo();
  const ctx=maskLayer.getContext('2d');ctx.fillStyle='white';ctx.beginPath();ctx.moveTo(polygon[0][0],polygon[0][1]);
  for(const point of polygon.slice(1))ctx.lineTo(point[0],point[1]);ctx.closePath();ctx.fill();
  lassoPoints=[];lassoPath=[];updateLassoButtons();markMaskDirty();paintMask();
}
function paintPoint(event){
  const point=position(event,$('#mask-canvas')),ctx=maskLayer.getContext('2d');
  ctx.strokeStyle=ctx.fillStyle=state.tool==='erase'?'black':'white';ctx.lineWidth=Number($('#brush-size').value)*maskLayer.width/600;ctx.lineCap='round';ctx.lineJoin='round';
  ctx.beginPath();ctx.moveTo(...(lastPoint||point));ctx.lineTo(point[0]+.01,point[1]+.01);ctx.stroke();lastPoint=point;markMaskDirty();paintMask();
}
async function submitReview(artifact,actionName,extra={}){
  await api(`/runs/${state.run.run_id}/reviews`,{method:'POST',body:JSON.stringify({artifact_id:artifact,action:actionName,reviewer:'Revisor local',reason_code:actionName==='approve'?'review_passed':'human_correction',...extra})});
  await selectRun(await api('/runs/'+state.run.run_id));
  toast(actionName==='remask'
    ?'Correção salva. Vistas automáticas equivalentes foram recalculadas e continuam pendentes de revisão.'
    :'Revisão salva. A versão anterior foi preservada.');
}
function renderMatches(selectedId){
  $('#approve-matches').disabled=!state.matches;
  if(!state.matches){$('#matching-content').innerHTML='Execute ou atualize o pipeline para conectar as peças.';return;}
  const parts=state.matches.parts, selected=parts.find(p=>p.part_instance_id===selectedId)||parts[0];
  const observations=state.segments.flatMap(s=>s.observations);
  $('#matching-content').innerHTML=`<div class="part-chips">${parts.map(p=>`<button class="part-chip ${p===selected?'active':''}" data-match="${p.part_instance_id}">${esc(partName(p))}</button>`).join('')}</div><div class="match-grid">${selected.observation_ids.map(oid=>{
    const o=observations.find(o=>o.observation_id===oid);if(!o)return'';
    const v=state.run.views.find(v=>v.view_id===o.view_id);
    return `<div class="match-card"><img src="${artifactUrl(o.mask_artifact_id)}" alt="Máscara ${esc(partName(o))}"><label>${labels[v.label]||v.label} · ${Math.round(o.confidence*100)}%</label><select data-assignment="${oid}" aria-label="Identidade da observação ${esc(labels[v.label])}">${parts.map(p=>`<option value="${p.part_instance_id}" ${p===selected?'selected':''}>${esc(partName(p))}</option>`).join('')}<option value="new">＋ Separar como nova peça</option></select><small>${esc(o.provenance.type)}</small></div>`;
  }).join('')}</div><p class="check-intro">Confirme rótulos e lados antes de agrupar. Alterações invalidam o grafo e a geometria dependentes.</p>`;
}
function renderPerception(){
  const target=$('#perception-content');
  if(!target)return;
  const graph=state.perception;
  if(!graph){target.className='panel padded empty-text';target.innerHTML='Execute o pipeline para gerar o Perception Graph.';return;}
  target.className='panel padded';
  const evidenceLabels={observed:'Observado',observed_multiview:'Observado multi-view',human_confirmed:'Confirmado pelo humano',inferred:'Inferido',prior:'Prior'};
  const relationLabels={above:'acima de',below:'abaixo de',connected_to:'conectado a',attached_to:'anexado a',symmetric_to:'simétrico a'};
  const partsById=Object.fromEntries(graph.parts.map(p=>[p.part_instance_id,p]));
  const partLabel=id=>{const p=partsById[id];return p?partName(p):id.slice(0,8);};
  const partCards=graph.parts.map(p=>{
    const g=p.geometry;
    const evidence=evidenceLabels[g.evidence_kind]||g.evidence_kind;
    const depth=g.depth_norm==null?'—':g.depth_norm.toFixed(3);
    const width=g.frontal_width_norm==null?'—':g.frontal_width_norm.toFixed(3);
    return `<article class="perception-card"><div class="perception-card-head"><strong>${esc(partName(p))}</strong><span class="evidence ${esc(g.evidence_kind)}">${esc(evidence)}</span></div><p>${esc(g.shape_family)}</p><dl><div><dt>Altura relativa</dt><dd>${g.height_norm.toFixed(3)}</dd></div><div><dt>Largura frontal</dt><dd>${width}</dd></div><div><dt>Profundidade projetada</dt><dd>${depth}</dd></div><div><dt>Completude</dt><dd>${Math.round(g.completeness*100)}%</dd></div></dl><small>${p.observations.length} vista(s) · confiança ${Math.round(p.confidence*100)}%</small></article>`;
  }).join('');
  const relations=graph.relations.map(r=>`<div class="perception-row"><span>${esc(partLabel(r.subject_part_id))}</span><strong>${esc(relationLabels[r.predicate]||r.predicate)}</strong><span>${esc(partLabel(r.object_part_id))}</span><em class="evidence ${esc(r.evidence_kind)}">${esc(evidenceLabels[r.evidence_kind]||r.evidence_kind)}</em></div>`).join('')||'<p class="muted">Sem relações suficientes.</p>';
  const interfaces=graph.interfaces.map(i=>`<div class="interface-row"><div><strong>${esc(partLabel(i.part_a_id))} ↔ ${esc(partLabel(i.part_b_id))}</strong><small>${esc(i.candidate_joint_type||i.role)}</small></div><span class="evidence ${esc(i.evidence_kind)}">${esc(evidenceLabels[i.evidence_kind]||i.evidence_kind)}</span></div>`).join('')||'<p class="muted">Nenhuma interface candidata.</p>';
  target.innerHTML=`<div class="perception-summary"><div><small>OBJETO</small><strong>${esc(graph.object_type)}</strong></div><div><small>SIMETRIA</small><strong>${esc(graph.symmetry)}</strong></div><div><small>REGIÕES</small><strong>${graph.major_regions.map(esc).join(' · ')}</strong></div><div><small>ESTILO</small><strong>${esc(graph.style_family||'não informado')}</strong></div></div><h3>Peças percebidas</h3><div class="perception-grid">${partCards}</div><div class="perception-columns"><section><h3>Relações</h3>${relations}</section><section><h3>Interfaces candidatas</h3>${interfaces}</section></div><div class="perception-warnings">${graph.warnings.map(w=>`<p>△ ${esc(w)}</p>`).join('')}</div>`;
}

function renderVolumetry(){
  const target=$('#volumetry-content');
  if(!target)return;
  const result=state.volumetry;
  if(!result){target.className='panel padded empty-text';target.innerHTML='Execute o pipeline para gerar volumes calibrados.';return;}
  target.className='panel padded';
  const calibration=state.calibration?.cameras||[];
  const cameraRows=calibration.map(c=>{
    const scale=c.mm_per_pixel!=null?`${Number(c.mm_per_pixel).toFixed(4)} mm/px`:`${Number(c.world_units_per_pixel||0).toFixed(6)} u/px`;
    return `<div class="calibration-row"><span>${esc(labels[c.label]||c.label)}</span><strong>${c.calibrated?'✓ calibrada':'△ aproximação'}</strong><small>${scale}</small></div>`;
  }).join('');
  const cards=result.volumes.map(v=>{
    const ext=v.extents_xyz||[0,0,0], field=v.field||{}, metrics=v.reprojection_metrics||[];
    const metricRows=metrics.map(m=>`<div class="reprojection-row boundary-${m.hard_constraint?'hard':'soft'}"><span>${esc(labels[m.view_label]||m.view_label)} ${m.hard_constraint?'🔒':''}</span><strong class="${m.hard_boundary_compliant&&m.silhouette_iou>=.95?'good':'warn'}">IoU ${(m.silhouette_iou*100).toFixed(1)}%</strong><small>fora ${(m.outside_area_ratio*100).toFixed(2)}% · overshoot ${Number(m.max_overshoot_px||0).toFixed(2)}px</small></div>`).join('');
    const grid=field.grid_shape?field.grid_shape.join(' × '):'—';
    const voxel=field.voxel_size_mm!=null?`${Number(field.voxel_size_mm).toFixed(3)} mm`:`${Number(field.voxel_size_world||0).toFixed(5)} ${esc(field.unit||result.unit)}`;
    return `<article class="volume-card calibrated"><div class="volume-card-head"><strong>${esc(v.name)}</strong><span>IoU ${((v.mean_reprojection_iou||0)*100).toFixed(1)}%</span></div><dl><div><dt>Dimensões</dt><dd>${ext.map(x=>Number(x).toFixed(3)).join(' × ')}</dd></div><div><dt>Grid</dt><dd>${esc(grid)}</dd></div><div><dt>Voxel</dt><dd>${esc(voxel)}</dd></div><div><dt>Representação</dt><dd>${field.representation?'Visual Hull + SDF':'Visual Hull v1'}</dd></div><div><dt>Concavidades</dt><dd>${v.concavity_support?'suportadas':'não suportadas'}</dd></div></dl><h4>Reprojeção</h4><div class="reprojection-list">${metricRows||'<p class="muted">Sem métricas nesta versão.</p>'}</div></article>`;
  }).join('');
  target.innerHTML=`<div class="volume-summary"><div><small>MÉTODO</small><strong>${esc(result.method)}</strong></div><div><small>PEÇAS COM VOLUME</small><strong>${result.volumes.length}</strong></div><div><small>UNIDADE</small><strong>${esc(result.unit)}</strong></div></div><div class="volumetry-diagnostics"><section><h3>Calibração</h3>${cameraRows||'<p class="muted">Calibração indisponível.</p>'}</section><section><h3>Critério geométrico</h3><p class="check-intro">MESH VALID ≠ GEOMETRY ACCURATE. A fidelidade agora é medida reprojetando o volume nas referências.</p></section></div><div class="volume-grid">${cards}</div><div class="perception-warnings">${result.warnings.map(w=>`<p>△ ${esc(w)}</p>`).join('')}</div>`;
}

function renderValidation(){
  if(!state.report){$('#validation').innerHTML='<p class="muted">Nenhuma geometria gerada.</p>';return;}
  const checks=Object.groupBy?Object.groupBy(state.report.checks,c=>c.code):state.report.checks.reduce((a,c)=>((a[c.code]??=[]).push(c),a),{});
  const titles={manifold:'Malha fechada',normals:'Normais orientadas',nonzero_faces:'Faces não degeneradas',physical_scale:'Escala física',scale_calibration:'Calibração multi-view',multiview_consistency:'Consistência multi-view',wall_thickness:'Espessura de parede',clearance:'Folgas',collisions:'Colisões',joint_range:'Articulações',separability:'Separabilidade'};
  $('#validation').innerHTML='<p class="check-intro">Revisão necessária. Este baseline ainda não está validado para fabricação.</p>'+Object.entries(checks).map(([key,items])=>{
    const ok=items.every(c=>c.status==='pass'),fail=items.some(c=>c.status==='fail');
    return `<div class="check-row"><span>${titles[key]||key}</span><span class="${ok?'pass':'warn'}">${ok?'✓ Verificado':fail?'✕ Falhou':items[0].status==='not_evaluated'?'Não avaliado':'Revisar'}</span></div>`;
  }).join('');
  $('#geometry-unit').textContent=state.reconstruction?.unit==='mm'?'UNIDADES: MILÍMETROS':'ESCALA RELATIVA · ALTURA = 1';
}
async function renderHistory(){
  if(!state.project)return;
  const events=await api(`/projects/${state.project.project_id}/feedback`);
  $('#history-content').innerHTML=events.length?events.map(e=>`<article class="history-row"><time>${new Date(e.created_at).toLocaleString('pt-BR')}</time><div><strong>${esc(e.action)}</strong> <span class="muted">· ${esc(e.reviewer)}</span><p>${esc(e.reason_code)} · ${e.stage} · ${e.target.slice(0,8)}</p>${e.comment?`<p>${esc(e.comment)}</p>`:''}<a class="text-button" href="${artifactUrl(e.before_artifact_id)}" target="_blank" rel="noopener">Versão anterior ↗</a>${e.after_artifact_id?` <a class="text-button" href="${artifactUrl(e.after_artifact_id)}" target="_blank" rel="noopener">Nova versão ↗</a>`:''}</div></article>`).join(''):'Nenhuma revisão registrada. As correções aparecerão aqui.';
}

let rotation=.5, tilt=.12, zoom=1, dragging=null;
function draw3D(){
  const canvas=$('#geometry-canvas'),box=canvas.getBoundingClientRect();if(!box.width)return;
  const ratio=window.devicePixelRatio||1;canvas.width=box.width*ratio;canvas.height=box.height*ratio;
  const ctx=canvas.getContext('2d');ctx.scale(ratio,ratio);const w=box.width,h=box.height;
  const height=state.graph?.scale.canonical_height||1,scale=Math.min(w*.8,h*.76)/height*zoom;
  function project(v){let x=v[0]*Math.cos(rotation)-v[1]*Math.sin(rotation),y=v[0]*Math.sin(rotation)+v[1]*Math.cos(rotation),z=v[2];return [w/2+x*scale,h*.55-(z*Math.cos(tilt)-y*Math.sin(tilt))*scale,y*Math.cos(tilt)+z*Math.sin(tilt)];}
  ctx.lineWidth=1;ctx.strokeStyle='#344b3b';
  for(let i=-5;i<=5;i++)for(let axis=0;axis<2;axis++){const a=project(axis?[i*height/10,-height*.6,-height*.46]:[-height*.6,i*height/10,-height*.46]),b=project(axis?[i*height/10,height*.6,-height*.46]:[height*.6,i*height/10,-height*.46]);ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.stroke();}
  const faces=[];
  state.meshes.forEach((mesh,index)=>{const vertices=mesh.vertices.map(project);mesh.faces.forEach(f=>{const v=f.map(i=>vertices[i]);const normal=(v[1][0]-v[0][0])*(v[2][1]-v[0][1])-(v[1][1]-v[0][1])*(v[2][0]-v[0][0]);faces.push({v,depth:v.reduce((a,p)=>a+p[2],0)/3,index,shade:Math.min(1,Math.abs(normal)/40)});});});
  faces.sort((a,b)=>b.depth-a.depth);
  for(const f of faces){const base=colors[f.index%colors.length];ctx.fillStyle=base;ctx.beginPath();ctx.moveTo(f.v[0][0],f.v[0][1]);ctx.lineTo(f.v[1][0],f.v[1][1]);ctx.lineTo(f.v[2][0],f.v[2][1]);ctx.closePath();ctx.fill();ctx.fillStyle=`rgba(10,25,14,${.08+f.shade*.28})`;ctx.fill();ctx.strokeStyle='#17332025';ctx.lineWidth=.35;ctx.stroke();}
  if(!state.meshes.length){ctx.fillStyle='#93a78b';ctx.textAlign='center';ctx.font='12px Segoe UI';ctx.fillText('Execute o pipeline para construir sua primeira montagem.',w/2,h*.44);}
}

document.addEventListener('click',event=>{const target=event.target.closest('button,[data-observation]');if(!target)return;action(async()=>{
  if(target.dataset.tab)setTab(target.dataset.tab);
  if(target.dataset.project)await selectProject(target.dataset.project);
  if(target.dataset.upload){uploadLabel=target.dataset.upload;$('#file-upload').click();}
  if(target.id==='new-project'||target.id==='new-project-small')$('#project-dialog').showModal();
  if(target.id==='cancel-project')$('#project-dialog').close();
  if(target.id==='cancel-feedback')$('#feedback-dialog').close();
  if(target.id==='run-button'){
    target.disabled=true;const run=await api(`/projects/${state.project.project_id}/runs`,{method:'POST',body:JSON.stringify({})});state.runs.unshift(run);await selectRun(run);setTab('pipeline');
  }
  if(target.dataset.observation){state.selected=target.dataset.observation;await loadMask();}
  if(target.dataset.tool){
    if(state.tool==='lasso'&&target.dataset.tool!=='lasso'&&lassoPoints.length)resetLasso();
    state.tool=target.dataset.tool;$('[data-tool]').forEach(b=>b.classList.toggle('active',b===target));toolCursor();
  }
  if(target.id==='undo-mask'&&undo.length){maskLayer.getContext('2d').putImageData(undo.pop(),0,0);markMaskDirty();paintMask();}
  if(target.id==='toggle-edges'){showEdges=!showEdges;target.classList.toggle('active',showEdges);$('.canvas-wrap').classList.toggle('edge-mode',showEdges);paintMask();}
  if(target.id==='refine-edge')maskOperation('refine');
  if(target.id==='grow-mask')maskOperation('grow');
  if(target.id==='shrink-mask')maskOperation('shrink');
  if(target.id==='fill-holes')maskOperation('fill');
  if(target.id==='finish-lasso')finishLasso();
  if(target.id==='cancel-lasso')resetLasso();
  if(target.id==='save-mask')await submitReview(currentSegment().artifact_id,'remask',{target_id:state.selected,mask_png_base64:maskLayer.toDataURL('image/png').split(',')[1]});
  if(target.id==='relabel')await submitReview(currentSegment().artifact_id,'relabel',{target_id:state.selected,part_class:$('#part-class').value,side:$('#part-side').value});
  if(target.id==='approve-part'||target.id==='reject-part')await submitReview(currentSegment().artifact_id,target.id==='approve-part'?'approve':'reject',{target_id:state.selected});
  if(target.dataset.match)renderMatches(target.dataset.match);
  if(target.id==='approve-matches')await submitReview(state.matches.artifact_id,'approve',{dimension:'cross_view_consistency'});
  if(target.id==='download')window.location.href=`/api/runs/${state.run.run_id}/download`;
  if(target.id==='replay'){const run=await api(`/runs/${state.run.run_id}/replay`,{method:'POST'});state.runs.unshift(run);await selectRun(run);}
  if(target.id==='final-review')$('#feedback-dialog').showModal();
});});
$('#project-form').addEventListener('submit',event=>{event.preventDefault();action(async()=>{const data=Object.fromEntries(new FormData(event.target));data.known_height_mm=data.known_height_mm?Number(data.known_height_mm):null;data.style_family=data.style_family||null;const p=await api('/projects',{method:'POST',body:JSON.stringify(data)});$('#project-dialog').close();event.target.reset();await loadProjects(p.project_id);setTab('workspace');toast('Projeto criado. Adicione as quatro vistas para começar.');});});
$('#file-upload').addEventListener('change',event=>action(async()=>{const file=event.target.files[0];if(!file)return;const data=new FormData();data.append('label',uploadLabel);data.append('file',file);await api(`/projects/${state.project.project_id}/views`,{method:'POST',body:data});const id=state.project.project_id;await loadProjects(id);event.target.value='';toast('Vista importada. Original preservado.');}));
$('#seg-view').addEventListener('change',event=>action(async()=>{state.activeView=event.target.value;state.selected=null;await loadMask();}));
$('#run-select').addEventListener('change',event=>action(()=>selectRun(state.runs.find(r=>r.run_id===event.target.value))));
document.addEventListener('change',event=>{if(event.target.dataset.assignment)action(async()=>{const id=event.target.value==='new'?crypto.randomUUID():event.target.value;await submitReview(state.matches.artifact_id,'rematch',{assignments:{[event.target.dataset.assignment]:id},dimension:'cross_view_consistency'});});});
const canvas=$('#mask-canvas');
canvas.addEventListener('pointerdown',event=>{
  if(state.tool==='inspect'||!selectedObservation())return;
  if(state.tool==='region'||state.tool==='wand'){smartSelect(event,state.tool);return;}
  if(state.tool==='lasso'){addLassoPoint(event);return;}
  if(state.tool!=='add'&&state.tool!=='erase')return;
  paint=true;lastPoint=null;pushUndo();canvas.setPointerCapture(event.pointerId);paintPoint(event);
});
canvas.addEventListener('pointermove',event=>{if(paint&&(state.tool==='add'||state.tool==='erase'))paintPoint(event);});
canvas.addEventListener('pointerup',()=>{paint=false;lastPoint=null;});
canvas.addEventListener('pointercancel',()=>{paint=false;lastPoint=null;});
canvas.addEventListener('dblclick',event=>{if(state.tool==='lasso'){event.preventDefault();finishLasso();}});
const gc=$('#geometry-canvas');gc.addEventListener('pointerdown',e=>{dragging=[e.clientX,e.clientY];gc.setPointerCapture(e.pointerId);});gc.addEventListener('pointermove',e=>{if(dragging){rotation+=(e.clientX-dragging[0])*.01;tilt=Math.max(-1,Math.min(1,tilt+(e.clientY-dragging[1])*.008));dragging=[e.clientX,e.clientY];draw3D();}});gc.addEventListener('pointerup',()=>dragging=null);gc.addEventListener('pointercancel',()=>dragging=null);gc.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.max(.4,Math.min(3,zoom*Math.exp(-e.deltaY*.001)));draw3D();},{passive:false});window.addEventListener('resize',draw3D);
const dimensions={segmentation_accuracy:'Precisão da segmentação',cross_view_consistency:'Consistência entre vistas',shape_fidelity:'Fidelidade da forma',style_fidelity:'Fidelidade do estilo',joint_correctness:'Articulações',connector_correctness:'Encaixes',assembly_quality:'Qualidade da montagem',printability:'Imprimibilidade',editability:'Editabilidade'};
$('#score-fields').innerHTML=Object.entries(dimensions).map(([key,label])=>`<label>${label}<input type="number" min="0" max="1" step=".05" name="${key}" required placeholder="0 a 1"></label>`).join('');
$('#feedback-form').addEventListener('submit',event=>{event.preventDefault();action(async()=>{const data=Object.fromEntries(new FormData(event.target)),scores=Object.fromEntries(Object.keys(dimensions).map(k=>[k,Number(data[k])]));await submitReview(node('S16').output_artifact_id,'final_evaluation',{reviewer:data.reviewer,comment:data.comment,scores,dimension:'multidimensional',reason_code:'final_evaluation'});$('#feedback-dialog').close();});});
init();
