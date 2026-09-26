'use strict';

(function(global){
  function clamp(value,min,max){return Math.max(min,Math.min(max,value));}

  function edgeMap(imageData){
    const {width,height,data}=imageData;
    const gray=new Float32Array(width*height);
    for(let i=0,p=0;i<data.length;i+=4,p++){
      gray[p]=data[i]*.2126+data[i+1]*.7152+data[i+2]*.0722;
    }
    const magnitude=new Float32Array(width*height);
    let max=1;
    for(let y=1;y<height-1;y++){
      for(let x=1;x<width-1;x++){
        const i=y*width+x;
        const tl=gray[i-width-1],tc=gray[i-width],tr=gray[i-width+1];
        const ml=gray[i-1],mr=gray[i+1];
        const bl=gray[i+width-1],bc=gray[i+width],br=gray[i+width+1];
        const gx=-tl+tr-2*ml+2*mr-bl+br;
        const gy=-tl-2*tc-tr+bl+2*bc+br;
        const value=Math.sqrt(gx*gx+gy*gy);
        magnitude[i]=value;max=Math.max(max,value);
      }
    }
    // Robust normalization: estimate a high percentile from a compact histogram.
    const bins=new Uint32Array(256);
    for(let i=0;i<magnitude.length;i++){
      bins[Math.min(255,Math.floor(magnitude[i]/max*255))]++;
    }
    const target=magnitude.length*.98;let cumulative=0,p98=255;
    for(let b=0;b<256;b++){cumulative+=bins[b];if(cumulative>=target){p98=Math.max(1,b);break;}}
    const scale=max*p98/255||1;
    const out=new Uint8Array(width*height);
    for(let i=0;i<out.length;i++)out[i]=Math.round(clamp(magnitude[i]/scale,0,1)*255);
    return out;
  }

  function colorDistance(data,a,b){
    const ia=a*4,ib=b*4;
    const dr=data[ia]-data[ib],dg=data[ia+1]-data[ib+1],db=data[ia+2]-data[ib+2];
    return Math.sqrt(dr*dr+dg*dg+db*db)/441.673;
  }

  function smartRegion(imageData,edges,seedX,seedY,options={}){
    const {width,height,data}=imageData,total=width*height;
    const x=clamp(Math.round(seedX),0,width-1),y=clamp(Math.round(seedY),0,height-1);
    const seed=y*width+x;
    const tolerance=(options.tolerance??40)/100;
    const sensitivity=(options.edgeSensitivity??60)/100;
    const edgeBarrier=clamp(245-sensitivity*185,45,240);
    const adaptive=options.adaptive!==false;
    const selected=new Uint8Array(total),visited=new Uint8Array(total),queue=new Int32Array(total);
    let head=0,tail=0;queue[tail++]=seed;visited[seed]=1;selected[seed]=1;
    while(head<tail){
      const current=queue[head++],cx=current%width,cy=(current/width)|0;
      const neighbors=[current-1,current+1,current-width,current+width];
      for(const next of neighbors){
        if(next<0||next>=total||visited[next])continue;
        const nx=next%width,ny=(next/width)|0;
        if(Math.abs(nx-cx)+Math.abs(ny-cy)!==1)continue;
        visited[next]=1;
        const edge=Math.max(edges[current],edges[next]);
        if(edge>edgeBarrier)continue;
        const seedDist=colorDistance(data,seed,next);
        const localDist=colorDistance(data,current,next);
        const score=adaptive?.68*seedDist+.32*localDist:seedDist;
        if(score>tolerance)continue;
        selected[next]=1;queue[tail++]=next;
      }
    }
    return selected;
  }

  function morph(mask,width,height,operation,iterations=1){
    let src=Uint8Array.from(mask);
    const offsets=[[-1,-1],[0,-1],[1,-1],[-1,0],[1,0],[-1,1],[0,1],[1,1]];
    for(let step=0;step<iterations;step++){
      const dst=new Uint8Array(src.length);
      for(let y=0;y<height;y++)for(let x=0;x<width;x++){
        const i=y*width+x;
        if(operation==='grow'){
          if(src[i]){dst[i]=1;continue;}
          dst[i]=offsets.some(([dx,dy])=>{
            const nx=x+dx,ny=y+dy;return nx>=0&&ny>=0&&nx<width&&ny<height&&src[ny*width+nx];
          })?1:0;
        }else{
          if(!src[i])continue;
          dst[i]=offsets.every(([dx,dy])=>{
            const nx=x+dx,ny=y+dy;return nx>=0&&ny>=0&&nx<width&&ny<height&&src[ny*width+nx];
          })?1:0;
        }
      }
      src=dst;
    }
    return src;
  }

  function fillHoles(mask,width,height){
    const outside=new Uint8Array(mask.length),queue=new Int32Array(mask.length);
    let head=0,tail=0;
    function seed(i){if(!mask[i]&&!outside[i]){outside[i]=1;queue[tail++]=i;}}
    for(let x=0;x<width;x++){seed(x);seed((height-1)*width+x);}
    for(let y=0;y<height;y++){seed(y*width);seed(y*width+width-1);}
    while(head<tail){
      const i=queue[head++],x=i%width,y=(i/width)|0;
      for(const [nx,ny] of [[x-1,y],[x+1,y],[x,y-1],[x,y+1]]){
        if(nx<0||ny<0||nx>=width||ny>=height)continue;
        const n=ny*width+nx;
        if(!mask[n]&&!outside[n]){outside[n]=1;queue[tail++]=n;}
      }
    }
    const out=Uint8Array.from(mask);
    for(let i=0;i<out.length;i++)if(!mask[i]&&!outside[i])out[i]=1;
    return out;
  }

  function strongest(edges,width,height,x0,y0,x1,y1,fallbackX,fallbackY){
    let best=-1,bx=fallbackX,by=fallbackY;
    for(let y=clamp(y0,0,height-1);y<=clamp(y1,0,height-1);y++){
      for(let x=clamp(x0,0,width-1);x<=clamp(x1,0,width-1);x++){
        const value=edges[y*width+x];
        if(value>best){best=value;bx=x;by=y;}
      }
    }
    return [bx,by,best];
  }

  function refineToEdges(mask,edges,width,height,radius=10){
    const rowMask=new Uint8Array(mask.length),colMask=new Uint8Array(mask.length);
    for(let y=0;y<height;y++){
      let left=-1,right=-1;
      for(let x=0;x<width;x++)if(mask[y*width+x]){left=x;break;}
      for(let x=width-1;x>=0;x--)if(mask[y*width+x]){right=x;break;}
      if(left<0)continue;
      const [sl,,le]=strongest(edges,width,height,left-radius,y,left+radius,y,left,y);
      const [sr,,re]=strongest(edges,width,height,right-radius,y,right+radius,y,right,y);
      const l=le>35?sl:left,r=re>35?sr:right;
      for(let x=Math.min(l,r);x<=Math.max(l,r);x++)rowMask[y*width+x]=1;
    }
    for(let x=0;x<width;x++){
      let top=-1,bottom=-1;
      for(let y=0;y<height;y++)if(mask[y*width+x]){top=y;break;}
      for(let y=height-1;y>=0;y--)if(mask[y*width+x]){bottom=y;break;}
      if(top<0)continue;
      const [,st,te]=strongest(edges,width,height,x,top-radius,x,top+radius,x,top);
      const [,sb,be]=strongest(edges,width,height,x,bottom-radius,x,bottom+radius,x,bottom);
      const t=te>35?st:top,b=be>35?sb:bottom;
      for(let y=Math.min(t,b);y<=Math.max(t,b);y++)colMask[y*width+x]=1;
    }
    const out=new Uint8Array(mask.length);
    for(let i=0;i<out.length;i++){
      const votes=(mask[i]?1:0)+(rowMask[i]?1:0)+(colMask[i]?1:0);
      out[i]=votes>=2?1:0;
    }
    return out;
  }

  function magneticPath(edges,width,height,start,end,snapRadius=18){
    const dx=end[0]-start[0],dy=end[1]-start[1],length=Math.max(2,Math.ceil(Math.hypot(dx,dy)));
    const nx=-dy/length,ny=dx/length,points=[];
    for(let step=0;step<=length;step++){
      const t=step/length,cx=start[0]+dx*t,cy=start[1]+dy*t;
      let bestScore=-Infinity,best=[cx,cy];
      for(let offset=-snapRadius;offset<=snapRadius;offset++){
        const x=Math.round(cx+nx*offset),y=Math.round(cy+ny*offset);
        if(x<0||y<0||x>=width||y>=height)continue;
        const edge=edges[y*width+x]/255;
        const score=edge-Math.abs(offset)/Math.max(1,snapRadius)*.22;
        if(score>bestScore){bestScore=score;best=[x,y];}
      }
      if(step===0)best=[Math.round(start[0]),Math.round(start[1])];
      if(step===length)best=[Math.round(end[0]),Math.round(end[1])];
      points.push(best);
    }
    const smooth=points.map((point,i)=>{
      if(i===0||i===points.length-1)return point;
      return [
        Math.round((points[i-1][0]+point[0]+points[i+1][0])/3),
        Math.round((points[i-1][1]+point[1]+points[i+1][1])/3),
      ];
    });
    return smooth;
  }

  global.DFSeg={edgeMap,smartRegion,morph,fillHoles,refineToEdges,magneticPath};
})(window);
