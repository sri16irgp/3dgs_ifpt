(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const viewport = $('viewport'), frame = $('frame'), status = $('status');
  const add = (a,b) => a.map((x,i)=>x+b[i]);
  const sub = (a,b) => a.map((x,i)=>x-b[i]);
  const mul = (a,s) => a.map(x=>x*s);
  const dot = (a,b) => a.reduce((n,x,i)=>n+x*b[i],0);
  const cross = (a,b) => [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
  const length = v => Math.hypot(...v);
  const norm = v => mul(v,1/Math.max(length(v),1e-9));
  const rotate = (v,axis,angle) => add(add(mul(v,Math.cos(angle)),mul(cross(axis,v),Math.sin(angle))),mul(axis,dot(axis,v)*(1-Math.cos(angle))));
  const state = {scene:null, socket:null, camera:null, target:null, up:null, selected:null, isolate:false, dragging:null,
    sequence:0, sent:0, displayed:0, pending:null, awaiting:false, lastSend:0, lastFrame:0, queryId:0, reconnect:1000};
  let imageUrl = null, timer = null;
  function message(value,error=false){status.textContent=value;status.classList.toggle('error',error)}
  function connection(value){const element=$('connection');if(element)element.textContent=value}
  function orientation(){
    const forward=norm(sub(state.target,state.camera));
    let right=norm(cross(forward,state.up));
    if(length(right)<0.01)right=[1,0,0];
    const down=norm(cross(forward,right));
    return [[right[0],down[0],forward[0]],
      [right[1],down[1],forward[1]],[right[2],down[2],forward[2]]];
  }
  function schedule(moving=false){
    if(!state.scene)return;
    state.pending={id:++state.sequence,position:state.camera,rotation:orientation(),
      width:Math.max(1,viewport.clientWidth),height:Math.max(1,viewport.clientHeight),
      fov_y:state.scene.initial.fov_y,instance:state.selected?.id ?? null,isolate:state.isolate,moving};
    flush();
    if(moving){clearTimeout(timer);timer=setTimeout(()=>schedule(false),180)}
  }
  function flush(){
    if(!state.pending||state.awaiting||state.socket?.readyState!==WebSocket.OPEN)return;
    const wait=Math.max(0,1000/15-(performance.now()-state.lastSend));
    if(wait>0){setTimeout(flush,wait);return}
    const request=state.pending;state.pending=null;state.awaiting=true;state.sent=request.id;
    state.lastSend=performance.now();state.socket.send(JSON.stringify(request));
  }
  function connect(){
    const socket=new WebSocket(`${location.protocol==='https:'?'wss':'ws'}://${location.host}/ws`);
    socket.binaryType='blob';state.socket=socket;
    connection('Connecting…');
    socket.onopen=()=>{connection('Connected to Thor');state.reconnect=1000;state.awaiting=false;schedule(false)};
    socket.onmessage=event=>{
      if(typeof event.data==='string'){
        const info=JSON.parse(event.data);
        if(info.type==='error'){state.awaiting=false;message(info.message,true);flush();return}
        state.frameInfo=info;return;
      }
      const info=state.frameInfo;state.frameInfo=null;state.awaiting=false;
      if(info && info.id>=state.displayed && info.id===state.sent){
        const next=URL.createObjectURL(event.data),old=imageUrl;imageUrl=next;
        frame.onload=()=>{if(old)URL.revokeObjectURL(old)};frame.src=next;
        state.displayed=info.id;$('busy').hidden=true;
        const now=performance.now(),fps=state.lastFrame?1000/(now-state.lastFrame):0;state.lastFrame=now;
        $('stats').textContent=`${info.width} × ${info.height} · ${info.render_ms} ms · ${fps.toFixed(1)} fps · GPU ${info.gpu_mb} MB`;
      }
      flush();
    };
    socket.onclose=()=>{if(state.socket!==socket)return;connection('Disconnected · reconnecting');state.awaiting=false;state.pending=null;setTimeout(connect,state.reconnect);state.reconnect=Math.min(state.reconnect*2,10000)};
    socket.onerror=()=>socket.close();
  }
  function setSelection(result){
    state.selected=result;state.isolate=false;
    $('selection').hidden=!result;
    for(const id of ['isolate','focus','clear'])$(id).disabled=!result;
    $('isolate').textContent='Isolate';
    $('download').classList.toggle('disabled',!result);
    $('download').setAttribute('aria-disabled',String(!result));
    if(result){$('object-id').textContent=`Object ${result.id}`;$('count').textContent=`${result.gaussians.toLocaleString()} Gaussians`;
      $('score').textContent=`Similarity ${result.cosine.toFixed(4)}`;$('download').href=`/api/export/${result.id}`}
    else $('download').removeAttribute('href');
    schedule(false);
  }
  $('search-form').addEventListener('submit',async e=>{
    e.preventDefault();const text=$('prompt').value.trim();if(!text)return;
    const id=++state.queryId;$('search').disabled=true;message('Searching objects on Thor…');
    try{const response=await fetch('/api/query',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text})});
      const payload=await response.json();if(id!==state.queryId)return;
      if(!response.ok)throw Error(payload.detail||'Search failed');
      setSelection(payload.result);message(payload.result?`Top match for “${text}” selected`:'No objects available');
    }catch(error){if(id===state.queryId)message(error.message,true)}
    finally{if(id===state.queryId)$('search').disabled=false}
  });
  $('isolate').onclick=()=>{state.isolate=!state.isolate;$('isolate').textContent=state.isolate?'Show context':'Isolate';schedule(false)};
  $('clear').onclick=()=>{state.queryId++;$('search').disabled=false;setSelection(null);message('Selection cleared')};
  $('reset').onclick=()=>{resetCamera();schedule(false)};
  $('focus').onclick=()=>{if(!state.selected)return;fetch('/api/focus/'+state.selected.id).then(r=>r.json()).then(data=>{
    if(!data.center)throw Error(data.detail||'Unable to focus');
    const forward=norm(sub(state.target,state.camera));state.target=data.center;
    state.camera=sub(data.center,mul(forward,Math.max(data.radius*3,0.2)));schedule(false);
  }).catch(e=>message(e.message,true))};
  viewport.addEventListener('contextmenu',e=>e.preventDefault());
  viewport.addEventListener('pointerdown',e=>{viewport.setPointerCapture(e.pointerId);state.dragging={x:e.clientX,y:e.clientY,pan:e.button===2};viewport.classList.add('dragging')});
  viewport.addEventListener('pointerup',()=>{state.dragging=null;viewport.classList.remove('dragging');schedule(false)});
  viewport.addEventListener('pointermove',e=>{
    if(!state.dragging)return;const dx=e.clientX-state.dragging.x,dy=e.clientY-state.dragging.y;
    state.dragging.x=e.clientX;state.dragging.y=e.clientY;
    const distance=length(sub(state.camera,state.target));
    if(state.dragging.pan){const forward=norm(sub(state.target,state.camera)),right=norm(cross(forward,state.up));
      const shift=add(mul(right,-dx*distance/viewport.clientHeight),mul(state.up,dy*distance/viewport.clientHeight));
      state.camera=add(state.camera,shift);state.target=add(state.target,shift)
    }else{let offset=sub(state.camera,state.target);offset=rotate(offset,state.up,-dx*.005);
      const right=norm(cross(norm(mul(offset,-1)),state.up));offset=rotate(offset,right,-dy*.005);
      state.camera=add(state.target,offset)}
    schedule(true);
  });
  viewport.addEventListener('wheel',e=>{e.preventDefault();const offset=sub(state.camera,state.target);
    state.camera=add(state.target,mul(offset,Math.exp(Math.max(-100,Math.min(100,e.deltaY))*.0015)));schedule(true)
  },{passive:false});
  new ResizeObserver(()=>schedule(false)).observe(viewport);
  function resetCamera(){const initial=state.scene.initial;state.camera=[...initial.position];
    const rot=initial.rotation,forward=[rot[0][2],rot[1][2],rot[2][2]];
    state.up=norm([-rot[0][1],-rot[1][1],-rot[2][1]]);
    state.target=add(state.camera,mul(forward,state.scene.radius))}
  fetch('/api/scene').then(r=>{if(!r.ok)throw Error('Unable to load scene');return r.json()}).then(scene=>{
    state.scene=scene;$('scene-name').textContent=scene.scene;resetCamera();message(`${scene.objects} objects loaded. Enter a prompt to select one.`);connect()
  }).catch(e=>{console.error('LBG viewer initialization failed:',e);message(e.message,true);$('busy').textContent=`Scene unavailable: ${e.message}`});
})();
