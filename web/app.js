'use strict';
const $=id=>document.getElementById(id);
const video=$('video'),overlay=$('overlay'),work=$('work'),resultsEl=$('results');
const workflow=new AssistiveWorkflow.Workflow();
const {DetectionMemory,LatestLane,describe,mapsUrl,locate}=AssistiveRealtime;
const options={frame_interval:350,ocr_interval_ms:2000,tts_cooldown_ms:650,detection_ttl_ms:6500,
  max_missed_frames:3,max_result_age_ms:4500,location_timeout_ms:8000,tts_rate:1.05,language:'en-IN'};
const state={phase:'IDLE',camera:'off',audio:{microphone:'off',tts:'idle',enabled:false},
  ready:false,live:false,muted:false,hazOnly:false,objectStatus:'idle',textStatus:'idle',
  lastSpeech:'',cache:{objects:null,text:null},searchStarted:0,searchMissSpoken:false,
  summary:null,navStatus:'',mapUrl:null,closed:false};
let session='',devicePitch=0,viewShape='',cameraPromise=null,healthPromise=null;
let VFOV=Number((()=>{try{return localStorage.getItem('av_vfov');}catch{return 0;}})())||0;
const memory=new DetectionMemory();
const notices=new Map();
const speech=new AssistiveSpeech.SpeechManager({
  Recognition:window.SpeechRecognition||window.webkitSpeechRecognition,
  synthesis:window.speechSynthesis,Utterance:window.SpeechSynthesisUtterance,
  onCommand:handleCommand,
  onState:value=>{state.audio=value;updateStatus();},
  onText:text=>{state.lastSpeech=text;$('transcript').textContent=text;},
  onError:(text,source)=>{$('transcript').textContent=text;if($('gate'))$('gateHint').textContent=text+' Use Start scanning to continue.';setStatus(text,'err');
    if(source==='recognition'&&!state.muted)speech.enqueue({text,key:'microphone-error',priority:1});},
});
function say(text,opts={}){
  if(state.muted){state.lastSpeech=text;$('transcript').textContent=text;opts.onDone?.(false);return;}
  speech.say(text,opts);
}
function notice(key,text,period=30000){
  if(Date.now()-(notices.get(key)||0)<period)return;
  notices.set(key,Date.now());
  if(state.muted){$('transcript').textContent=text;return;}
  speech.enqueue({key:'notice-'+key,text,priority:1,expiresAt:Date.now()+5000});
}
function setStatus(text,kind=''){const el=$('status');el.className=kind;el.querySelector('span').textContent=text;}
function updateStatus(){
  const cameraOn=!!video.srcObject?.getTracks().some(t=>t.readyState==='live');
  state.camera=cameraOn?'on':state.phase==='STARTING_SCAN'?'starting':'off';
  $('componentStatus').textContent=`Camera: ${state.camera}. Microphone: ${state.audio.microphone}. Detection: ${state.objectStatus}. OCR: ${state.textStatus}. TTS: ${state.audio.tts}.`;
  $('enableVoice').textContent=state.audio.enabled?'Disable voice commands':'Enable voice commands';
  $('voiceBtn').querySelector('span').textContent=state.audio.enabled?'Voice on':'Enable voice';
  $('voiceBtn').classList.toggle('listening',state.audio.microphone==='listening');
  $('liveBtn').querySelector('span').textContent=state.live?'Stop scanning':'Start scanning';
  $('liveBtn').classList.toggle('on',state.live);
  if(state.phase==='STARTING_SCAN')$('modeStatus').textContent='Starting camera';
  else if(state.phase==='ERROR')$('modeStatus').textContent='Visual assistance unavailable';
  else if(state.phase==='ENVIRONMENT_SCAN')$('modeStatus').textContent='Environmental scan';
  else if(state.live)$('modeStatus').textContent=workflow.mode==='SEARCH'?'Keyword search':'Live monitoring';
}
const configReady=fetch('/api/config',{signal:AbortSignal.timeout(5000)}).then(r=>r.json()).then(c=>{
  Object.assign(options,c);speech.cooldown=options.tts_cooldown_ms;speech.rate=options.tts_rate;speech.language=options.language;
  memory.ttl=options.detection_ttl_ms;memory.maxMisses=options.max_missed_frames;
}).catch(()=>{});
function reveal(){
  $('gate')?.remove();$('app').style.visibility='visible';state.closed=false;
}
function clearView(lane){
  if(lane){state.cache[lane]=null;memory.hide(lane);}else{state.cache={objects:null,text:null};memory.hide('objects');memory.hide('text');resultsEl.replaceChildren();overlay.getContext('2d').clearRect(0,0,overlay.width,overlay.height);}
  drawCurrent();
}
function drawCurrent(){
  const now=Date.now();memory.expire(now);speech.prune();
  for(const lane of ['objects','text'])if(state.cache[lane]&&now-state.cache[lane].captured>options.detection_ttl_ms){state.cache[lane]=null;memory.hide(lane);}
  const observations=Object.values(state.cache).filter(Boolean);
  if(!observations.length){resultsEl.replaceChildren();overlay.getContext('2d').clearRect(0,0,overlay.width,overlay.height);return;}
  const frame=observations.at(-1).frame;
  const current={frame,objects:[],hazards:[],texts:[],symbols:[],match:null};
  for(const observation of observations){
    for(const t of observation.tracks){
      if(!memory.valid(t,t.version,now))continue;
      const it={...t.item,confirmed:t.stable};
      current[it.kind==='text'?'texts':it.kind==='hazard'?'hazards':'objects'].push(it);
      if(t.searchMatch===workflow.query&&workflow.mode==='SEARCH')current.match=it;
    }
  }
  render(current);
}
async function startCamera(generation){
  if(video.srcObject?.getTracks().some(t=>t.readyState==='live'))return true;
  // Reuse a pending permission request. Obsolete streams are stopped before use.
  if(cameraPromise){await cameraPromise;if(generation!==workflow.generation)return false;return startCamera(generation);}
  cameraPromise=(async()=>{
    if(!navigator.mediaDevices?.getUserMedia)throw new Error('Camera access requires HTTPS or localhost.');
    const stream=await navigator.mediaDevices.getUserMedia({video:{facingMode:{ideal:'environment'},width:{ideal:1280},height:{ideal:720}},audio:false});
    if(generation!==workflow.generation){stream.getTracks().forEach(t=>t.stop());return false;}
    video.srcObject=stream;
    await new Promise((resolve,reject)=>{
      const done=()=>{clearTimeout(timer);video.removeEventListener('loadedmetadata',done);resolve();};
      const timer=setTimeout(()=>{video.removeEventListener('loadedmetadata',done);reject(new Error('Camera startup timed out.'));},8000);
      video.addEventListener('loadedmetadata',done,{once:true});if(video.readyState>=1)done();
    });
    if(generation!==workflow.generation){stream.getTracks().forEach(t=>t.stop());return false;}
    await video.play();
    for(const track of stream.getTracks())track.onended=()=>{
      if(state.live){stopResources();state.phase='ERROR';notice('camera-ended','Camera disconnected. Say start scanning to retry.');updateStatus();}
    };
    return true;
  })();
  try{return await cameraPromise;}finally{cameraPromise=null;}
}
function stopResources(){
  state.live=false;objectLane.stop();textLane.stop();state.summary=null;
  video.srcObject?.getTracks().forEach(t=>{t.onended=null;t.stop();});video.srcObject=null;
  state.objectStatus='idle';state.textStatus='idle';memory.clear();clearView();
}
async function checkHealth(){
  if(healthPromise)return healthPromise;
  healthPromise=(async()=>{
    try{
      const response=await fetch('/api/health',{signal:AbortSignal.timeout(5000)});const h=await response.json();
      state.ready=!!h.ready;state.schema=h.object_schema;
      if(!state.ready){notice('models',h.error?'Vision models unavailable. Retrying automatically.':'Vision models are loading. Scanning will resume automatically.');return false;}
      return true;
    }catch(e){state.ready=false;notice('server','Vision server unavailable. Retrying automatically.');return false;}
    finally{healthPromise=null;}
  })();return healthPromise;
}
// Separate capture surfaces: simultaneous toBlob calls never redraw the same canvas.
const canvases={objects:document.createElement('canvas'),text:document.createElement('canvas')};
async function capture(lane){
  const w=video.videoWidth,h=video.videoHeight;if(!w||!h)throw new Error('Camera not ready');
  const shape=w+':'+h;
  if(viewShape&&shape!==viewShape){memory.clear();clearView();}
  viewShape=shape;
  const canvas=canvases[lane],scale=Math.min(1,960/Math.max(w,h));
  canvas.width=Math.round(w*scale);canvas.height=Math.round(h*scale);
  canvas.getContext('2d').drawImage(video,0,0,canvas.width,canvas.height);
  const captured=Date.now();
  const blob=await new Promise((resolve,reject)=>canvas.toBlob(b=>b?resolve(b):reject(new Error('Frame capture failed')),'image/jpeg',.85));
  return {blob,captured,shape};
}
async function analyzeLane(lane,{signal,current}){
  const generation=workflow.generation;
  const status=lane==='objects'?'objectStatus':'textStatus';
  if(!state.live||document.hidden)return;
  if(!state.ready&&!await checkHealth()){state[status]='unavailable';updateStatus();return;}
  if(!current()||generation!==workflow.generation)return;
  const frame=await capture(lane);if(!current()||generation!==workflow.generation)return;
  state[status]='processing';updateStatus();
  const fd=new FormData();fd.append('frame',frame.blob,'frame.jpg');fd.append('component',lane);
  fd.append('keyword',workflow.query);fd.append('mode','single');fd.append('session',session);
  fd.append('pitch',String(devicePitch));if(VFOV)fd.append('vfov',String(VFOV));
  const controller=new AbortController();const abort=()=>controller.abort();signal.addEventListener('abort',abort,{once:true});
  const timeout=setTimeout(abort,15000);
  try{
    const response=await fetch('/api/detect/'+lane,{method:'POST',body:fd,signal:controller.signal});
    const d=await response.json();
    if(!current()||generation!==workflow.generation||document.hidden)return;
    if(!response.ok){if(response.status===503&&!d.busy)state.ready=false;throw new Error(d.error||'Vision request failed');}
    if(Date.now()-frame.captured>options.max_result_age_ms||frame.shape!==viewShape){clearView(lane);state[status]='waiting for current frame';return;}
    if(d.component_errors?.length){clearView(lane);state[status]='unavailable';notice('component-'+lane,d.component_errors.join(' '));return;}
    const items=lane==='objects'?[...d.objects,...d.hazards]:d.texts;
    const tracks=memory.update(lane,items,d.frame,Date.now());
    state.cache[lane]={tracks,frame:d.frame,captured:frame.captured};state[status]='active';
    for(const t of tracks){
      if(d.match&&t.item.label===d.match.label&&t.item.box.every((x,i)=>x===d.match.box[i]))t.searchMatch=workflow.query;
      else if(workflow.mode==='SEARCH')t.searchMatch=null;
      announceTrack(t);
    }
    if(lane==='text'&&workflow.mode==='SEARCH'&&workflow.query&&!state.searchMissSpoken&&
       Date.now()-state.searchStarted>8000&&!memory.fresh(Date.now()).some(t=>t.searchMatch===workflow.query)){
      state.searchMissSpoken=true;notice('search-missing-'+workflow.query,`I have not found ${workflow.query} yet. Search remains active.`);
    }
    if(state.summary&&frame.captured>=state.summary.started){state.summary.done.add(lane);finishSummary();}
    drawCurrent();setStatus(d.frame_quality==='limited'?'Monitoring · limited image quality':'Monitoring','ok');
  }catch(e){
    if(current()&&generation===workflow.generation){clearView(lane);state[status]='unavailable';notice('request-'+lane,'Visual '+(lane==='text'?'text recognition':'object detection')+' temporarily unavailable. Retrying.');}
  }finally{clearTimeout(timeout);signal.removeEventListener('abort',abort);if(current())updateStatus();}
}
const objectLane=new LatestLane(ctx=>analyzeLane('objects',ctx),{interval:options.frame_interval,onError:()=>{clearView('objects');state.objectStatus='unavailable';updateStatus();}});
const textLane=new LatestLane(ctx=>analyzeLane('text',ctx),{interval:options.ocr_interval_ms,onError:()=>{clearView('text');state.textStatus='unavailable';updateStatus();}});
function announceTrack(t){
  if(!t.stable||state.muted)return;
  const searching=workflow.mode==='SEARCH'&&t.searchMatch===workflow.query;
  if(state.summary&&!t.item.critical)return;
  if(!searching&&(t.spoken||state.hazOnly&&!t.item.critical))return;
  if(searching&&t.searchSpoken===workflow.query)return;
  const version=t.version,query=workflow.query,generation=workflow.generation;
  const priority=t.item.critical?0:searching?1:t.item.kind==='text'?2:3;
  const text=t.item.critical?'Warning. '+describe(t.item):searching?`I found ${query}. `+describe(t.item):describe(t.item);
  speech.enqueue({key:`track-${t.id}-${version}-${searching?query:''}`,text,priority,
    expiresAt:Date.now()+options.max_result_age_ms,
    valid:()=>generation===workflow.generation&&memory.valid(t,version,Date.now()),
    onStart:()=>{memory.markSpoken(t,Date.now());if(searching)t.searchSpoken=query;if(t.item.critical)flashHazard();}});
}
function flashHazard(){
  try{navigator.vibrate?.([120,60,120]);}catch{}
  const el=$('hazardFlash');el.style.display='block';setTimeout(()=>el.style.display='none',1200);
}
function finishSummary(force=false){
  const summary=state.summary;if(!summary||!force&&summary.done.size<2)return;
  state.summary=null;state.phase=workflow.mode==='SEARCH'?'SEARCHING':'SCANNING';
  const tracks=Object.values(state.cache).filter(Boolean).flatMap(c=>c.tracks).filter(t=>!t.misses&&Date.now()-t.lastSeen<memory.ttl)
    .sort((a,b)=>Number(!!b.item.critical)-Number(!!a.item.critical)||(a.item.steps||99)-(b.item.steps||99)).slice(0,3);
  const text='Environmental scan complete. '+(tracks.length?tracks.map(t=>describe(t.item)).join(' '):'No reliable objects or text identified. Monitoring continues.');
  speech.enqueue({key:'summary-'+summary.started,text,priority:2,expiresAt:Date.now()+options.max_result_age_ms,
    valid:()=>state.live&&tracks.every(t=>memory.valid(t,t.version,Date.now())),
    onStart:()=>tracks.forEach(t=>memory.markSpoken(t,Date.now()))});
  updateStatus();
}
async function environmentScan(){
  if(!state.live)await selectMode('ENVIRONMENT');
  if(!state.live)return;
  state.phase='ENVIRONMENT_SCAN';state.summary={started:Date.now(),done:new Set()};
  objectLane.kick();textLane.kick();updateStatus();
}
function deadline(promise,ms){
  let timer;return Promise.race([promise,new Promise((_,reject)=>{timer=setTimeout(()=>reject(new Error('Operation timed out')),ms);})]).finally(()=>clearTimeout(timer));
}
async function selectMode(mode,query=''){
  if(mode==='ENVIRONMENT_SCAN'){await environmentScan();return;}
  reveal();workflow.transition(mode,query);const generation=workflow.generation;
  speech.cancel();state.summary=null;state.mapUrl=null;$('mapLink').hidden=true;
  $('modeChoices').hidden=mode!=='MAIN_MENU';$('queryForm').hidden=!['SEARCH','NAVIGATION'].includes(mode);
  $('queryInput').value=query;$('queryLabel').textContent=mode==='NAVIGATION'?'Where would you like to go?':'What would you like me to find?';
  const visual=['ENVIRONMENT','SEARCH'].includes(mode);$('stage').hidden=!visual;$('controls').hidden=!visual;
  $('modeStatus').textContent={MAIN_MENU:'Choose assistance',ENVIRONMENT:'Live monitoring',SEARCH:'Keyword search',NAVIGATION:'Navigation',STOPPED:'Assistance stopped'}[mode];
  if(!visual)stopResources();
  if(mode==='STOPPED'){state.phase='IDLE';speech.stop();say('Stopped. Camera and microphone are off.');updateStatus();return;}
  if(mode==='MAIN_MENU'){state.phase='IDLE';say('Say start scanning, two for keyword search, or three for navigation.');updateStatus();return;}
  if(mode==='NAVIGATION'){
    state.phase='NAVIGATING';if(query)await navigateTo(query);else{say('Where would you like to go?');$('queryInput').focus();}updateStatus();return;
  }
  if(mode==='SEARCH'){state.searchStarted=Date.now();state.searchMissSpoken=false;
    if(!query){say('What would you like me to find?');$('queryInput').focus();updateStatus();return;}}
  if(state.live){state.phase=mode==='SEARCH'?'SEARCHING':'SCANNING';say(mode==='SEARCH'?'Searching for '+query:'Monitoring continues.');objectLane.kick();textLane.kick();updateStatus();return;}
  state.phase='STARTING_SCAN';say('Starting the camera and live monitoring.');updateStatus();
  try{
    await configReady;
    if(!await deadline(startCamera(generation),12000)||generation!==workflow.generation)return;
    session=crypto.randomUUID?.()||Math.random().toString(36).slice(2);memory.clear();clearView();
    state.live=true;state.phase=mode==='SEARCH'?'SEARCHING':'SCANNING';
    say(mode==='SEARCH'?'Searching for '+query:'Live monitoring started.');
    objectLane.interval=options.frame_interval;textLane.interval=options.ocr_interval_ms;
    objectLane.start();textLane.start();
  }catch(e){
    if(generation!==workflow.generation)return;
    workflow.generation++;stopResources();state.phase='ERROR';say('Camera unavailable. Allow camera access in browser settings and use HTTPS, then say start scanning to retry.');
  }finally{updateStatus();}
}
async function navigateTo(destination){
  if(!destination){say('Please say navigate to, followed by a destination.');return;}
  workflow.destination=destination;const generation=workflow.generation;
  state.navStatus='Getting location';setStatus(state.navStatus,'busy');say('Getting your location for '+destination+'.');
  const locationResult=await locate(navigator.geolocation,{timeout:options.location_timeout_ms});
  if(generation!==workflow.generation)return;
  state.mapUrl=mapsUrl(destination,locationResult.coords);$('mapLink').href=state.mapUrl;$('mapLink').hidden=false;
  state.navStatus='Directions ready';setStatus(state.navStatus,'ok');
  const indoor=state.schema&&state.schema!=='coco'?'Your object model does not detect vehicles. ':'';
  const text=(locationResult.error?locationResult.error+' Google Maps can choose your starting point. ':'')+
    indoor+'Opening Google Maps for directions, distance and travel time to '+destination+'.';
  // Same-tab navigation works from voice callbacks without popup permissions.
  say(text,{onDone:()=>{if(generation===workflow.generation)openMaps();}});
}
function openMaps(){
  if(!state.mapUrl)return;
  try{window.location.assign(state.mapUrl);}
  catch(e){setStatus('Google Maps could not open','err');say('Google Maps could not open. Use Open Google Maps directions to retry.');}
}
function handleCommand(raw){
  const command=workflow.command(raw);if(!command)return;
  const [intent,target]=command;
  if(state.phase==='STARTING_SCAN'&&intent==='ENVIRONMENT')return;
  if(intent==='REPEAT'){if(state.lastSpeech)say(state.lastSpeech);return;}
  if(intent==='NAV_INFO'){if(workflow.destination)selectMode('NAVIGATION',workflow.destination);else selectMode('NAVIGATION');return;}
  if(intent==='HELP'){say('Say start scanning, scan environment, find followed by a sign, navigate to a destination, distance and time, stop, back, or repeat.');return;}
  if(intent==='MUTE'||intent==='UNMUTE'){setMuted(intent==='MUTE');return;}
  if(intent==='HAZARDS'){$('hazOnly').click();return;}
  if(intent==='CALIBRATE'){calibrate();return;}
  if(intent==='CLOSE'){selectMode('STOPPED');return;}
  selectMode(intent,target||'');
}
function setMuted(value){state.muted=value;speech.cancel();$('muteBtn').setAttribute('aria-pressed',value);$('muteBtn').setAttribute('aria-label',value?'Unmute voice output':'Mute voice output');$('muteBtn').textContent=value?'Unmute':'Mute';if(!value)say('Sound on.');}
function toggleVoice(){speech.listen(!speech.enabled);if(speech.enabled)say('Voice commands enabled.');}
function calibrate(){
  // Keep the existing optional measured calibration without using camera-steering prompts.
  const value=window.prompt('Optional measured vertical camera field of view in degrees (25–90). Leave blank to keep the current value.',String(VFOV||50));
  if(value===null||!value.trim())return;
  const fov=Number(value);if(!Number.isFinite(fov)||fov<25||fov>90){say('Calibration value must be between 25 and 90 degrees.');return;}
  VFOV=fov;try{localStorage.setItem('av_vfov',String(fov));}catch{}
  say('Camera calibration saved. Distances remain approximate.');
}
$('startBtn').onclick=()=>{speech.listen(true);selectMode('ENVIRONMENT');};
$('enableVoice').onclick=toggleVoice;$('voiceBtn').onclick=toggleVoice;
$('menuBtn').onclick=()=>selectMode('MAIN_MENU');$('stopAll').onclick=()=>selectMode('STOPPED');
$('repeatBtn').onclick=()=>state.lastSpeech&&say(state.lastSpeech);
$('liveBtn').onclick=()=>selectMode(state.live?'STOPPED':'ENVIRONMENT');
$('scanOnce').onclick=environmentScan;$('navBtn').onclick=()=>selectMode('NAVIGATION');
$('closeBtn').onclick=()=>selectMode('STOPPED');$('calibChip').onclick=calibrate;
$('muteBtn').onclick=()=>setMuted(!state.muted);
$('hazOnly').onclick=()=>{state.hazOnly=!state.hazOnly;$('hazOnly').classList.toggle('tog',state.hazOnly);say(state.hazOnly?'Hazard announcements only.':'All new information enabled.');};
document.querySelectorAll('[data-mode]').forEach(b=>b.onclick=()=>selectMode(b.dataset.mode==='ENVIRONMENT'?'ENVIRONMENT_SCAN':b.dataset.mode));
document.querySelectorAll('.chip[data-k]').forEach(b=>b.onclick=()=>selectMode('SEARCH',b.dataset.k));
$('queryForm').onsubmit=e=>{e.preventDefault();const target=$('queryInput').value.trim();if(target)selectMode(workflow.mode,target);};
$('navClose').onclick=()=>$('navSheet').classList.remove('open');
$('navGo').onclick=()=>selectMode('NAVIGATION',$('dest').value.trim());
window.addEventListener('deviceorientation',e=>{
  devicePitch=e.beta!=null&&Math.abs(screen.orientation?.angle||0)%180===0?Math.max(-10,Math.min(70,90-e.beta)):0;
});
document.addEventListener('visibilitychange',()=>{
  // Visibility is not a Stop command. Keep the voice session's listening intent.
  if(document.hidden){clearView('objects');clearView('text');}
});
window.addEventListener('pagehide',()=>{workflow.generation++;stopResources();speech.stop();});
const expiryTimer=setInterval(()=>{
  drawCurrent();if(state.summary&&Date.now()-state.summary.started>12000)finishSummary(true);
  // Retry any eligible unspoken current items after the previous utterance finishes.
  if(state.live)for(const t of memory.fresh(Date.now()))announceTrack(t);
  updateStatus();
},250);
window.addEventListener('load',()=>{
  say('Welcome to Assistive Vision System. Please say start scanning to begin.');
  speech.listen(true); // Browsers that require a gesture expose the Start/voice fallback.
},{once:true});
const COLORS={hazard:'#ff453a',text:'#ffd60a',object:'#0a84ff',symbol:'#30d158'};
function render(d){
  const w=d.frame.w,h=d.frame.h;
  overlay.width=w;overlay.height=h;   // CSS object-fit:cover matches video crop
  const ctx=overlay.getContext('2d');
  ctx.clearRect(0,0,w,h);
  ctx.lineWidth=Math.max(3,w/260);
  ctx.font=`800 ${Math.max(14,Math.round(w/42))}px Inter,sans-serif`;
  const draw=(items,color)=>items.forEach(it=>{
    const [x1,y1,x2,y2]=it.box;
    ctx.strokeStyle=color;
    if(it.confirmed===false)ctx.setLineDash([9,7]);
    if(it.quad){ctx.beginPath();it.quad.forEach(([x,y],i)=>i?ctx.lineTo(x,y):ctx.moveTo(x,y));ctx.closePath();ctx.stroke();}
    else ctx.strokeRect(x1,y1,x2-x1,y2-y1);ctx.setLineDash([]);
    const label=String(it.label).slice(0,22);
    const tw=ctx.measureText(label).width+14,th=Math.max(20,Math.round(w/34));
    const ly=Math.max(0,y1-th);
    ctx.fillStyle=color;ctx.fillRect(x1,ly,tw,th);
    ctx.fillStyle='#000';ctx.fillText(label,x1+7,ly+th-6);
  });
  draw(d.objects,COLORS.object);draw(d.texts,COLORS.text);
  draw(d.symbols,COLORS.symbol);draw(d.hazards,COLORS.hazard);
  if(d.match){const [x1,y1,x2,y2]=d.match.box;
    ctx.strokeStyle='#fff';ctx.setLineDash([12,7]);
    ctx.strokeRect(x1-7,y1-7,x2-x1+14,y2-y1+14);ctx.setLineDash([]);}

  resultsEl.innerHTML='';
  const short=dir=>dir.includes('left')?'← LEFT':dir.includes('right')?'RIGHT →':'↑ AHEAD';
  const add=(it,cls)=>{
    const div=document.createElement('div');div.className='hit '+cls;
    if(it.confirmed===false){div.style.opacity='.45';div.style.borderStyle='dashed';}
    div.innerHTML='<span class="t"></span><span class="d"></span>';
    div.querySelector('.t').textContent=it.label;
    div.querySelector('.d').textContent=(it.confirmed===false?'CONFIRMING… ':'')+short(it.direction)+
      (it.steps?(it.steps>30?' · FAR':' · ≈'+it.steps+' STEP'+(it.steps>1?'S':'')):
       it.proximity==='very close'?' · CLOSE':it.proximity==='nearby'?' · NEAR':'');
    resultsEl.appendChild(div);
  };
  d.hazards.forEach(x=>add(x,'hazard'));
  if(d.match)add(d.match,'match');
  d.symbols.slice(0,4).forEach(x=>{if(!d.match||x.label!==d.match.label)add(x,'symbol');});
  d.texts.slice(0,5).forEach(x=>add(x,'text'));
  d.objects.slice(0,5).forEach(x=>add(x,'object'));
}

