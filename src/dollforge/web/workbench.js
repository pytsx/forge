import * as THREE from 'three';
import {OrbitControls} from './vendor/OrbitControls.js';
import {viewFrame, pixelToWorld, maskBoundarySegments} from './scene-math.mjs';

const $ = selector => document.querySelector(selector);
const palette = [0x9ad6bd, 0x89bddd, 0xdac18a, 0xc8a2d0, 0x7fc6c9, 0xda9c89, 0xb2c68f];
const viewNames = {front: 'FRENTE · −Y', back: 'COSTAS · +Y', left: 'ESQUERDA · +X', right: 'DIREITA · −X'};
const html = text => String(text ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'}[c]));
const layers = {images:true, contours:true, masks:false, bounds:true, meshes:true, wireframe:false, grid:true};
let viewport;

class Workbench {
  constructor() {
    this.canvas = $('#geometry-canvas');
    this.renderer = new THREE.WebGLRenderer({canvas:this.canvas, antialias:true, alpha:true});
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.setClearColor(0x20262e, 0);
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.scene = new THREE.Scene();
    this.camera = new THREE.OrthographicCamera(-1, 1, 1, -1, .001, 10000);
    this.camera.up.set(0, 0, 1);
    this.controls = new OrbitControls(this.camera, this.canvas);
    this.controls.enableDamping = false;
    this.controls.screenSpacePanning = true;
    this.controls.addEventListener('change', () => this.draw());
    this.scene.add(new THREE.HemisphereLight(0xe5efff, 0x485563, 2.1));
    const light = new THREE.DirectionalLight(0xffffff, 2.6);
    light.position.set(2, -3, 5); this.scene.add(light);
    this.root = new THREE.Group(); this.scene.add(this.root);
    this.labels = []; this.images = new Map(); this.hiddenViews = new Set(); this.hiddenParts = new Set();
    this.height = 1; this.epoch = 0; this.key = ''; this.project = null; this.selected = null;
    this.spread = .65; this.opacity = .55; this.meshObjects = [];
    new ResizeObserver(() => this.resize()).observe($('#scene-host'));
    let down;
    this.canvas.addEventListener('pointerdown', e => down = [e.clientX, e.clientY]);
    this.canvas.addEventListener('pointerup', e => {
      if (down && e.button === 0 && Math.hypot(e.clientX-down[0], e.clientY-down[1]) < 5) this.pick(e);
      down = null;
    });
    this.canvas.addEventListener('keydown', e => {if(e.key.toLowerCase()==='f'){e.preventDefault();this.frame();}});
    this.canvas.addEventListener('webglcontextlost', e => {
      e.preventDefault(); this.error('A aceleração 3D foi interrompida. Recarregue a página para restaurar a cena.');
    });
    this.setCamera('orbit');
    this.resize();
  }

  error(message) {$('#viewport-error').textContent=message;$('#viewport-error').hidden=false;}
  resize() {
    const w = this.canvas.clientWidth, h = this.canvas.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    const extent = this.height * .9, aspect = w / h;
    this.camera.left = -extent * aspect; this.camera.right = extent * aspect;
    this.camera.top = extent; this.camera.bottom = -extent;
    this.camera.updateProjectionMatrix(); this.draw();
  }
  draw() {
    if (this.framePending) return;
    this.framePending = true;
    requestAnimationFrame(() => {
      this.framePending = false;
      this.renderer.render(this.scene, this.camera);
      const w = this.canvas.clientWidth, h = this.canvas.clientHeight;
      for (const label of this.labels) {
        const p = label.position.clone().addScaledVector(label.normal, this.spread*this.height).project(this.camera);
        label.element.style.left = `${(p.x*.5+.5)*w}px`;
        label.element.style.top = `${(-p.y*.5+.5)*h}px`;
        label.element.hidden = this.hiddenViews.has(label.view) || Math.abs(p.x)>1 || Math.abs(p.y)>1 || Math.abs(p.z)>1;
      }
    });
  }
  setCamera(name) {
    const directions = {orbit:[1.5,-2,1.15],front:[0,-2,0],back:[0,2,0],left:[2,0,0],right:[-2,0,0],top:[0,0,2]};
    const d = new THREE.Vector3(...directions[name]);
    this.camera.up.set(...(name==='top' ? [0,1,0] : [0,0,1]));
    this.camera.position.copy(this.controls.target).addScaledVector(d, this.height*2);
    this.camera.near = this.height*.001; this.camera.far = this.height*100;
    this.camera.updateProjectionMatrix(); this.controls.update(); this.draw();
    document.querySelectorAll('[data-camera]').forEach(b=>b.classList.toggle('active',b.dataset.camera===name));
  }
  frame() {this.controls.target.set(0,0,0);this.camera.zoom=1;this.setCamera('orbit');this.resize();}
  disposeRoot() {
    this.root.traverse(obj => {
      obj.geometry?.dispose();
      const materials = Array.isArray(obj.material) ? obj.material : obj.material ? [obj.material] : [];
      materials.forEach(m=>{m.map?.dispose();m.dispose();});
    });
    this.root.clear(); this.labels=[]; this.meshObjects=[]; $('#scene-labels').replaceChildren();
  }
  async raster(identifier, maxSize=512) {
    const key = `${identifier}:${maxSize}`;
    if (!this.images.has(key)) {
      const promise = new Promise((resolve,reject)=>{
        const image = new Image();
        image.onload=()=>{
          const scale=Math.min(1,maxSize/Math.max(image.width,image.height));
          const canvas=document.createElement('canvas');
          canvas.width=Math.max(1,Math.round(image.width*scale));canvas.height=Math.max(1,Math.round(image.height*scale));
          const ctx=canvas.getContext('2d',{willReadFrequently:true});
          ctx.imageSmoothingEnabled=maxSize>512;ctx.drawImage(image,0,0,canvas.width,canvas.height);resolve(canvas);
        };
        image.onerror=()=>reject(new Error('Não foi possível carregar uma referência ou máscara.'));
        image.src=`/api/artifacts/${identifier}/content`;
      }).catch(error=>{this.images.delete(key);throw error;});
      this.images.set(key,promise);
      if(this.images.size>80)this.images.delete(this.images.keys().next().value);
    }
    return this.images.get(key);
  }
  plane(frame, width, height, material) {
    const points=[[0,height],[width,height],[width,0],[0,0]].flatMap(([u,v])=>pixelToWorld(u,v,frame));
    const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.Float32BufferAttribute(points,3));
    geometry.setAttribute('uv',new THREE.Float32BufferAttribute([0,0,1,0,1,1,0,1],2));
    geometry.setIndex([0,1,2,0,2,3]); geometry.computeVertexNormals();
    return new THREE.Mesh(geometry,material);
  }
  lines(points, color, loop=false) {
    const geometry=new THREE.BufferGeometry().setAttribute('position',new THREE.Float32BufferAttribute(points.flat(),3));
    const material=new THREE.LineBasicMaterial({color,transparent:true,opacity:.85,depthTest:false});
    return loop ? new THREE.LineLoop(geometry,material) : new THREE.LineSegments(geometry,material);
  }
  applyLayers() {
    this.root.traverse(obj=>{
      if(obj.userData.layer)obj.visible=layers[obj.userData.layer];
      if(obj.userData.viewRoot)obj.visible=!this.hiddenViews.has(obj.userData.viewRoot);
      if(obj.userData.partRoot)obj.visible=!this.hiddenParts.has(obj.userData.partRoot);
      if(obj.userData.referenceImage)obj.material.opacity=this.opacity;
      if(obj.userData.viewRoot)obj.position.fromArray(obj.userData.normal).multiplyScalar(this.spread*this.height);
      if(obj.userData.surface){
        obj.material.wireframe=layers.wireframe;
        obj.material.emissive.setHex(obj.userData.partId===this.selected ? 0x234d42 : 0);
      }
    });this.draw();
  }
  async sync(data) {
    this.data=data;
    const views = data.run?.views?.length ? data.run.views : data.views;
    const key=JSON.stringify([data.project?.project_id,views.map(v=>v.normalized_artifact_id),data.segments.map(s=>s.artifact_id),data.calibration?.artifact_id,data.volumetry?.artifact_id,data.reconstruction?.meshes.map(m=>m.preview_artifact_id)]);
    if(key===this.key){this.updateStatus();return;}
    this.key=key;const epoch=++this.epoch;
    const newProject=this.project!==data.project?.project_id;
    if(newProject){this.project=data.project?.project_id;this.images.clear();this.hiddenViews.clear();this.hiddenParts.clear();this.selected=null;}
    const previousHeight=this.height;
    this.height=data.graph?.scale.canonical_height||data.project?.known_height_mm||1;
    this.disposeRoot();
    $('#viewport-error').hidden=true;$('#scene-loading').hidden=!views.length;
    const gridGroup=new THREE.Group();gridGroup.userData.layer='grid';this.root.add(gridGroup);
    const grid=new THREE.GridHelper(this.height*2.4,24,0x657585,0x394651);grid.rotation.x=Math.PI/2;grid.position.z=-this.height*.51;gridGroup.add(grid);
    const axes=new THREE.AxesHelper(this.height*.22);axes.position.set(-this.height*.8,-this.height*.8,-this.height*.5);gridGroup.add(axes);
    const meshes=data.meshes.length ? data.meshes : data.volumetry?.volumes||[];
    meshes.forEach((mesh,i)=>{
      if(!mesh.vertices?.length||!mesh.faces?.length)return;
      const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.Float32BufferAttribute(mesh.vertices.flat(),3));geometry.setIndex(mesh.faces.flat());geometry.computeVertexNormals();
      const material=new THREE.MeshStandardMaterial({color:palette[i%palette.length],roughness:.68,metalness:.05,side:THREE.DoubleSide});
      const surface=new THREE.Mesh(geometry,material);surface.userData={surface:true,partId:mesh.part_instance_id,layer:'meshes'};
      const group=new THREE.Group();group.userData.partRoot=mesh.part_instance_id;group.add(surface);this.root.add(group);this.meshObjects.push(surface);
      const bounds=new THREE.Box3Helper(new THREE.Box3().setFromObject(surface),0xe8b677);bounds.userData.layer='bounds';bounds.material.transparent=true;bounds.material.opacity=.5;group.add(bounds);
    });
    this.updateObjects(meshes);this.updateStatus();
    if(newProject||previousHeight!==this.height)this.frame();else this.resize();
    this.applyLayers();
    try {
      await Promise.all(views.map(async view=>{
        const observations=data.segments.find(s=>s.view_id===view.view_id)?.observations||[];
        const camera=data.calibration?.cameras.find(c=>c.view_id===view.view_id);
        const frame=viewFrame(view,camera,this.height,observations);
        const image=await this.raster(view.normalized_artifact_id,1024);
        if(epoch!==this.epoch)return;
        const group=new THREE.Group();group.userData={viewRoot:view.view_id,normal:frame.normal};this.root.add(group);
        const texture=new THREE.CanvasTexture(image);texture.colorSpace=THREE.SRGBColorSpace;
        const plane=this.plane(frame,view.qa.width,view.qa.height,new THREE.MeshBasicMaterial({map:texture,side:THREE.DoubleSide,transparent:true,opacity:this.opacity,depthWrite:false}));
        plane.userData={layer:'images',referenceImage:true,viewId:view.view_id};plane.renderOrder=-2;group.add(plane);
        const corners=[[0,0],[view.qa.width,0],[view.qa.width,view.qa.height],[0,view.qa.height]].map(p=>pixelToWorld(...p,frame));
        const border=this.lines(corners,0x708b9c,true);border.userData.layer='images';group.add(border);
        const label=document.createElement('span');label.className='scene-label';label.textContent=viewNames[view.label]||view.label;$('#scene-labels').append(label);
        this.labels.push({element:label,position:new THREE.Vector3(...pixelToWorld(view.qa.width/2,-16,frame)),normal:new THREE.Vector3(...frame.normal),view:view.view_id});
        await Promise.all(observations.filter(o=>o.review_state!=='rejected').map(async (observation,i)=>{
          const mask=await this.raster(observation.mask_artifact_id);
          if(epoch!==this.epoch)return;
          const pixels=mask.getContext('2d').getImageData(0,0,mask.width,mask.height);
          const edges=maskBoundarySegments(pixels.data,mask.width,mask.height),points=[];
          for(let n=0;n<edges.length;n+=2)points.push(pixelToWorld(edges[n]*view.qa.width/mask.width,edges[n+1]*view.qa.height/mask.height,frame,.0002*this.height));
          const contour=this.lines(points,palette[i%palette.length]);contour.userData.layer='contours';group.add(contour);
          const [x0,y0,x1,y1]=observation.bbox_xyxy;
          const box=this.lines([[x0,y0],[x1,y0],[x1,y1],[x0,y1]].map(p=>pixelToWorld(...p,frame,.0003*this.height)),0xe8b677,true);box.userData.layer='bounds';box.material.opacity=.45;group.add(box);
          const rgba=document.createElement('canvas');rgba.width=mask.width;rgba.height=mask.height;const ctx=rgba.getContext('2d');const color=new THREE.Color(palette[i%palette.length]);
          for(let n=0;n<pixels.data.length;n+=4){const on=pixels.data[n]>127;pixels.data[n]=color.r*255;pixels.data[n+1]=color.g*255;pixels.data[n+2]=color.b*255;pixels.data[n+3]=on?110:0;}ctx.putImageData(pixels,0,0);
          const fillTexture=new THREE.CanvasTexture(rgba);fillTexture.colorSpace=THREE.SRGBColorSpace;
          const fill=this.plane(frame,view.qa.width,view.qa.height,new THREE.MeshBasicMaterial({map:fillTexture,transparent:true,depthWrite:false,side:THREE.DoubleSide,depthTest:false}));fill.userData.layer='masks';fill.renderOrder=2;group.add(fill);
        }));
        if(epoch===this.epoch)this.applyLayers();
      }));
    } catch(error) {if(epoch===this.epoch)this.error(error.message);}
    finally {if(epoch===this.epoch){$('#scene-loading').hidden=true;this.applyLayers();}}
  }
  updateStatus() {
    const data=this.data;if(!data)return;
    const count=data.meshes.length||data.volumetry?.volumes?.length||0;
    $('#scene-empty').hidden=!!data.project;
    $('#scene-mode').textContent=count?(data.reconstruction?'MODELO 3D · EM REVISÃO':'VOLUME INTERMEDIÁRIO · EM REVISÃO'):'REFERÊNCIAS E CONTORNOS';
    const calibrated=data.calibration?.cameras.filter(c=>c.calibrated).length||0;
    $('#scene-registration').textContent=calibrated?`${calibrated} vistas alinhadas · projeção ortográfica`:'Posicionamento estimado · aguardando calibração';
    const meshes=data.meshes.length?data.meshes:data.volumetry?.volumes||[];
    $('#scene-stats').textContent=`${count} peças · ${meshes.reduce((n,m)=>n+(m.faces?.length||0),0).toLocaleString('pt-BR')} triângulos`;
    $('#geometry-unit').textContent=data.graph?.scale.unit==='mm'||(!data.graph&&data.project?.known_height_mm)?'MILÍMETROS':'ESCALA RELATIVA';
  }
  updateObjects(meshes) {
    $('#scene-parts').innerHTML=meshes.length?meshes.map((m,i)=>`<div class="part-row ${this.selected===m.part_instance_id?'selected':''}"><input type="checkbox" data-part-visible="${m.part_instance_id}" aria-label="Mostrar ${html(m.name)}" ${this.hiddenParts.has(m.part_instance_id)?'':'checked'}><span class="swatch" style="background:#${palette[i%palette.length].toString(16)}"></span><button data-scene-part="${m.part_instance_id}">${html(m.name)}</button><small>${Math.round(m.confidence*100)}%</small></div>`).join(''):'<p class="muted">As peças aparecerão durante a reconstrução.</p>';
  }
  selectPart(id) {
    this.selected=id;const meshes=this.data.meshes.length?this.data.meshes:this.data.volumetry?.volumes||[];
    const part=meshes.find(m=>m.part_instance_id===id);if(!part)return;
    $('#selection-info').innerHTML=`<h3>Peça selecionada</h3><strong>${html(part.name)}</strong><p>${part.faces.length.toLocaleString('pt-BR')} triângulos · confiança ${Math.round(part.confidence*100)}%</p><p>${html(part.provenance?.note||part.provenance?.source)}</p>`;
    this.updateObjects(meshes);this.applyLayers();
  }
  selectView(id) {
    const views=this.data.run?.views?.length?this.data.run.views:this.data.views;
    const view=views.find(v=>v.view_id===id);if(!view)return;
    $('#selection-info').innerHTML=`<h3>Referência selecionada</h3><strong>${html(viewNames[view.label])}</strong><p>${view.qa.width} × ${view.qa.height} pixels</p><p>Os contornos mostram as máscaras segmentadas. Os retângulos mostram seus limites.</p><button data-tab="segmentation" class="button secondary small">Editar esta vista</button>`;
    window.dispatchEvent(new CustomEvent('forge-view',{detail:id}));
  }
  pick(event) {
    const box=this.canvas.getBoundingClientRect();const pointer=new THREE.Vector2((event.clientX-box.left)/box.width*2-1,-(event.clientY-box.top)/box.height*2+1);
    const ray=new THREE.Raycaster();ray.setFromCamera(pointer,this.camera);
    const surfaces=this.meshObjects.filter(m=>m.visible&&m.parent.visible);
    const hit=ray.intersectObjects(surfaces,false)[0];
    if(hit){this.selectPart(hit.object.userData.partId);return;}
    const planes=[];this.root.traverse(o=>{if(o.userData.referenceImage&&o.visible&&o.parent.visible)planes.push(o);});
    const viewHit=ray.intersectObjects(planes,false)[0];if(viewHit)this.selectView(viewHit.object.userData.viewId);
  }
}

try {
  viewport=new Workbench();window.forgeViewport=viewport;
  window.dispatchEvent(new Event('forge-viewport-ready'));
} catch(error) {
  $('#viewport-error').hidden=false;
  $('#viewport-error').textContent='Não foi possível iniciar o espaço 3D. Ative a aceleração gráfica do navegador. '+error.message;
}
document.addEventListener('click',event=>{
  if(!viewport)return;
  const button=event.target.closest('button');if(!button)return;
  if(button.dataset.camera)viewport.setCamera(button.dataset.camera);
  if(button.id==='frame-scene')viewport.frame();
  if(button.dataset.scenePart)viewport.selectPart(button.dataset.scenePart);
  if(button.dataset.sceneView)viewport.selectView(button.dataset.sceneView);
});
document.addEventListener('change',event=>{
  if(!viewport)return;const target=event.target;
  if(target.dataset.layer)layers[target.dataset.layer]=target.checked;
  if(target.dataset.viewVisible){if(target.checked)viewport.hiddenViews.delete(target.dataset.viewVisible);else viewport.hiddenViews.add(target.dataset.viewVisible);}
  if(target.dataset.partVisible){if(target.checked)viewport.hiddenParts.delete(target.dataset.partVisible);else viewport.hiddenParts.add(target.dataset.partVisible);}
  viewport.applyLayers();
});
$('#reference-opacity').addEventListener('input',e=>{if(viewport){viewport.opacity=Number(e.target.value)/100;viewport.applyLayers();}});
$('#reference-distance').addEventListener('input',e=>{if(viewport){viewport.spread=Number(e.target.value)/100;viewport.applyLayers();}});
