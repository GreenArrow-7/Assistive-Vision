'use strict';
const $=id=>document.getElementById(id);
const video=$('video'),overlay=$('overlay'),resultsEl=$('results');
const workflow=new AssistiveWorkflow.Workflow();
const {DetectionMemory,LatestLane,describe,textIsImportant,mapsUrl,locate}=AssistiveRealtime;
const options={frame_interval:350,ocr_interval_ms:2000,tts_cooldown_ms:650,detection_ttl_ms:6500,
  max_missed_frames:3,max_result_age_ms:4500,location_timeout_ms:8000,tts_rate:1.05,language:'en-IN'};
const state={phase:'IDLE',camera:'off',audio:{microphone:'off',tts:'idle',enabled:false},
  ready:false,live:false,muted:false,hazOnly:false,objectStatus:'idle',textStatus:'idle',
  lastSpeech:'',cache:{objects:null,text:null},searchStarted:0,searchMissSpoken:false,searchFound:false,speechFailed:false,
  summary:null,routeSpeech:'',mapUrl:null,mapsAsked:0,waiting:false,closed:false,
  // Demo mode (server DEMO_MODE=true): recorded scenes replace the camera and are always labelled.
  demo:false,scenes:[],demoStarted:0,sceneShown:-1};
let session='',devicePitch=0,viewShape='',cameraPromise=null,healthPromise=null;
let VFOV=Number((()=>{try{return localStorage.getItem('av_vfov');}catch{return 0;}})())||0;
const memory=new DetectionMemory();
const notices=new Map();
const speech=new AssistiveSpeech.SpeechManager({
  Recognition:window.SpeechRecognition||window.webkitSpeechRecognition,
  synthesis:window.speechSynthesis,Utterance:window.SpeechSynthesisUtterance,
  onCommand:handleCommand,
  onState:value=>{state.audio=value;updateStatus();},
  onText:text=>{state.lastSpeech=text;show(text,false);},
  // When speech output fails, the message itself (not the error) goes to the screen-reader region.
  // A microphone problem is voiced by the app, so it is not sent to the screen reader as well unless muted.
  onError:(text,source)=>{const mic=source==='recognition';if(!mic)state.speechFailed=true;
    show(mic?text:state.lastSpeech,!mic||state.muted);setStatus(text,'err');
    if($('gate'))$('gateHint').textContent=(mic?'':state.lastSpeech+' ')+text+' Use Start scanning to continue.';
    if(mic&&!state.muted)speech.enqueue({text,key:'microphone-error',priority:1});},
});
// #transcript is plain visible text. #srLive is the one live region and is filled only when the app
// is not voicing the message itself (muted, or speech output failed), so nothing is announced twice.
function show(text,live){$('transcript').textContent=text;$('srLive').textContent=live?text:'';}
function say(text,opts={}){
  if(state.muted){state.lastSpeech=text;show(text,true);opts.onDone?.(false);return;}
  speech.say(text,opts);
}
function notice(key,text,period=30000){
  // A condition that has not changed earns less and less airtime: a server that stays down
  // repeated itself every 30s forever, which is most of what made the assistant feel noisy.
  const seen=notices.get(key);
  if(seen&&Date.now()-seen.at<Math.min(seen.period*2,240000))return;
  notices.set(key,{at:Date.now(),period:seen?Math.min(seen.period*2,240000):period});
  if(state.muted){show(text,true);return;}
  speech.enqueue({key:'notice-'+key,text,priority:1,expiresAt:Date.now()+5000});
}
function setStatus(text,kind=''){const el=$('status');el.className=kind;el.querySelector('span').textContent=text;}
function updateStatus(){
  const cameraOn=!!video.srcObject?.getTracks().some(t=>t.readyState==='live');
  state.camera=state.demo&&state.live?'not used (demo, recorded scenes)':cameraOn?'on':state.phase==='STARTING_SCAN'?'starting':'off';
  // Written only on change: the 250 ms refresh must not replace text under a screen-reader cursor.
  const put=(el,text)=>{if(el.textContent!==text)el.textContent=text;};
  put($('componentStatus'),`Camera: ${state.camera}. Microphone: ${state.audio.microphone}. Detection: ${state.objectStatus}. OCR: ${state.textStatus}. TTS: ${state.audio.tts}.`);
  put($('enableVoice'),state.audio.enabled?'Disable voice commands':'Enable voice commands');
  put($('voiceBtn').querySelector('span'),state.audio.enabled?'Voice on':'Enable voice');
  $('voiceBtn').classList.toggle('listening',state.audio.microphone==='listening');
  put($('liveBtn').querySelector('span'),state.live?'Stop scanning':'Start scanning');
  $('liveBtn').classList.toggle('on',state.live);
  const mode=state.phase==='STARTING_SCAN'?'Starting camera':state.phase==='ERROR'?'Visual assistance unavailable':state.phase==='ENVIRONMENT_SCAN'?'Environmental scan':
    state.live?(state.demo?'Demo mode · ':'')+(workflow.mode==='SEARCH'?'Keyword search':state.demo?'recorded scenes':'Live monitoring'):'';
  if(mode)put($('modeStatus'),mode);
}
const sceneNow=()=>state.scenes[state.sceneShown];
const simulatedPrefix=()=>state.demo&&sceneNow()?.simulated?'Simulated. ':'';
async function loadDemo(){
  const r=await fetch('/api/demo/scenes',{signal:AbortSignal.timeout(5000)});const d=await r.json();
  // Preload the pictures so the first frame of each scene lines up with its overlay.
  state.scenes=(d.scenes||[]).map(s=>({...s,img:Object.assign(document.createElement('img'),{src:s.image})}));
  state.demo=state.scenes.length>0;
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
    current.symbols.push(...observation.symbols); // the server's sign reading of this frame; drawn, not spoken
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
    // The permission prompt waits on the person, so it has no deadline; a spoken hint explains the silence.
    const hint=setTimeout(()=>{if(generation===workflow.generation)notice('camera-permission','Waiting for camera permission. Choose Allow in the browser prompt.');},6000);
    let stream;
    try{stream=await navigator.mediaDevices.getUserMedia({video:{facingMode:{ideal:'environment'},width:{ideal:1280},height:{ideal:720}},audio:false});}
    finally{clearTimeout(hint);}
    if(generation!==workflow.generation){stream.getTracks().forEach(t=>t.stop());return false;}
    video.srcObject=stream;
    await new Promise((resolve,reject)=>{
      const done=()=>{clearTimeout(timer);video.removeEventListener('loadedmetadata',done);resolve();};
      const timer=setTimeout(()=>{video.removeEventListener('loadedmetadata',done);reject(new Error('Camera startup timed out.'));},8000);
      video.addEventListener('loadedmetadata',done,{once:true});if(video.readyState>=1)done();
    });
    if(generation!==workflow.generation){stream.getTracks().forEach(t=>t.stop());return false;}
    await deadline(video.play(),4000);
    for(const track of stream.getTracks())track.onended=()=>{
      if(state.live){stopResources();state.phase='ERROR';notice('camera-ended','Camera disconnected. '+(speech.enabled?'Say':'Press')+' start scanning to retry.');updateStatus();}
    };
    return true;
  })();
  try{return await cameraPromise;}finally{cameraPromise=null;}
}
function stopResources(){
  state.live=false;objectLane.stop();textLane.stop();state.summary=null;
  video.srcObject?.getTracks().forEach(t=>{t.onended=null;t.stop();});video.srcObject=null;
  state.objectStatus='idle';state.textStatus='idle';memory.clear();clearView();
  state.sceneShown=-1;$('demoImage').hidden=true;$('demoBanner').hidden=true;
}
async function checkHealth(){
  if(healthPromise)return healthPromise;
  healthPromise=(async()=>{
    try{
      const response=await fetch('/api/health',{signal:AbortSignal.timeout(5000)});const h=await response.json();
      state.ready=!!h.ready;state.schema=h.object_schema;
      if(h.demo&&!state.demo)await loadDemo();
      // While a scan is starting, selectMode reports the not-ready state itself, so it is not said twice.
      if(!state.ready){state.waiting=true;if(state.phase!=='STARTING_SCAN')notice('models',h.error?'Vision models unavailable. Retrying automatically.':'Vision models are loading. Scanning will resume automatically.');return false;}
      // The promised automatic start is announced, so the wait never ends in silence.
      if(state.waiting&&state.live)notice('ready','The vision system is ready. Scanning has started.',0);
      state.waiting=false;return true;
    }catch(e){state.ready=false;state.waiting=true;notice('server','Vision server unavailable. Retrying automatically.');return false;}
    finally{healthPromise=null;}
  })();return healthPromise;
}
// Separate capture surfaces: simultaneous toBlob calls never redraw the same canvas.
const canvases={objects:document.createElement('canvas'),text:document.createElement('canvas')};
async function captureDemo(lane){
  // The client picks the scene by time and pins it in the request, so the picture on
  // screen and the recorded detections always belong to the same scene.
  const seconds=state.scenes[0].seconds||8;
  const i=Math.floor((Date.now()-state.demoStarted)/(seconds*1000))%state.scenes.length,scene=state.scenes[i];
  const shape='demo:'+i;
  if(viewShape&&shape!==viewShape){memory.clear();clearView();}
  viewShape=shape;
  if(state.sceneShown!==i){
    state.sceneShown=i;const el=$('demoImage');el.src=scene.image;el.hidden=false;
    const banner=$('demoBanner');banner.hidden=false;
    banner.textContent=`DEMO MODE · recorded scene ${i+1} of ${state.scenes.length}${scene.simulated?' · SIMULATED':''} · ${scene.description}`;
  }
  const w=scene.img.naturalWidth||720,h=scene.img.naturalHeight||960;
  const canvas=canvases[lane],scale=Math.min(1,960/Math.max(w,h));
  canvas.width=Math.round(w*scale);canvas.height=Math.round(h*scale);
  canvas.getContext('2d').drawImage(scene.img,0,0,canvas.width,canvas.height);
  const captured=Date.now();
  const blob=await new Promise((resolve,reject)=>canvas.toBlob(b=>b?resolve(b):reject(new Error('Frame capture failed')),'image/jpeg',.85));
  return {blob,captured,shape,scene:i};
}
async function capture(lane){
  if(state.demo)return captureDemo(lane);
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
  if(frame.scene!==undefined)fd.append('scene',String(frame.scene));
  const controller=new AbortController();const abort=()=>controller.abort();signal.addEventListener('abort',abort,{once:true});
  const timeout=setTimeout(abort,15000);
  try{
    const response=await fetch('/api/detect/'+lane,{method:'POST',body:fd,signal:controller.signal});
    const d=await response.json();
    if(!current()||generation!==workflow.generation||document.hidden)return;
    if(!response.ok){if(response.status===503&&!d.busy)state.ready=false;throw new Error(d.error||'Vision request failed');}
    if(Date.now()-frame.captured>options.max_result_age_ms||frame.shape!==viewShape){clearView(lane);state[status]='waiting for current frame';return;}
    if(d.component_errors?.length){clearView(lane);state[status]='unavailable';notice('component-'+lane,d.component_errors.join(' '));return;}
    // Hazards are tracked and announced before plain objects, closest first.
    const items=lane==='objects'?[...d.hazards].sort((a,b)=>(a.steps||99)-(b.steps||99)).concat(d.objects):d.texts;
    const tracks=memory.update(lane,items,d.frame,Date.now());
    state.cache[lane]={tracks,symbols:d.symbols||[],frame:d.frame,captured:frame.captured};state[status]='active';
    for(const t of tracks){
      if(d.match&&t.item.label===d.match.label&&t.item.box.every((x,i)=>x===d.match.box[i]))t.searchMatch=workflow.query;
      else if(workflow.mode==='SEARCH')t.searchMatch=null;
      announceTrack(t);
    }
    if(lane==='text'&&workflow.mode==='SEARCH'&&workflow.query&&!state.searchMissSpoken&&!state.searchFound&&
       Date.now()-state.searchStarted>8000&&!memory.fresh(Date.now()).some(t=>t.searchMatch===workflow.query)){
      state.searchMissSpoken=true;notice('search-missing-'+workflow.query,`I have not found ${workflow.query} yet. Search remains active.`);
    }
    // A search does not run forever: after 90 s with no match in view it ends and plain monitoring continues.
    // It never ends on top of a critical warning or a pending summary, and never denies a find it announced.
    if(workflow.mode==='SEARCH'&&workflow.query&&Date.now()-state.searchStarted>90000&&speech.active?.priority!==0&&!summaryPending()&&
       !memory.fresh(Date.now()).some(t=>t.searchMatch===workflow.query)){
      const q=workflow.query,found=state.searchFound;selectMode('ENVIRONMENT');
      say(found?`Search for ${q} ended. Scanning continues.`:`I couldn't find ${q}. Search stopped. Scanning continues.`);return;
    }
    if(state.summary&&frame.captured>=state.summary.started){state.summary.done.add(lane);finishSummary();}
    drawCurrent();setStatus(d.frame_quality==='limited'?'Monitoring · limited image quality':'Monitoring','ok');
  }catch(e){
    if(current()&&generation===workflow.generation)laneFailed(lane);
  }finally{clearTimeout(timeout);signal.removeEventListener('abort',abort);if(current())updateStatus();}
}
// A failed request and a failed frame capture are both spoken (rate limited), never only shown.
function laneFailed(lane){
  clearView(lane);state[lane==='objects'?'objectStatus':'textStatus']='unavailable';
  notice('request-'+lane,'Visual '+(lane==='text'?'text recognition':'object detection')+' temporarily unavailable. Retrying.');
}
const objectLane=new LatestLane(ctx=>analyzeLane('objects',ctx),{interval:options.frame_interval,onError:()=>{laneFailed('objects');updateStatus();}});
const textLane=new LatestLane(ctx=>analyzeLane('text',ctx),{interval:options.ocr_interval_ms,onError:()=>{laneFailed('text');updateStatus();}});
// True from the request for a summary until its sentence has been spoken, dropped or cancelled.
const summaryPending=()=>!!state.summary||[speech.active,...speech.queue].some(x=>x?.key.startsWith('summary-'));
// Hazards before everything else, then closest first.
const byUrgency=(a,b)=>Number(b.item.kind==='hazard')-Number(a.item.kind==='hazard')||(a.item.steps||99)-(b.item.steps||99);
function announceTrack(t){
  if(!t.stable)return;
  const searching=workflow.mode==='SEARCH'&&t.searchMatch===workflow.query;
  // Only a critical warning may interrupt a requested summary; the rest is announced after it.
  if(summaryPending()&&!t.item.critical)return;
  if(!searching&&(t.spoken||state.hazOnly&&t.item.kind!=='hazard'&&!t.item.critical))return;
  if(searching&&t.searchSpoken===workflow.query)return;
  // Incidental text (posters, small print) is not read out; signs and the searched keyword are.
  if(!searching&&t.item.kind==='text'&&!textIsImportant(t.item.label))return;
  const version=t.version,query=workflow.query,generation=workflow.generation;
  // Critical 0, then any hazard or the searched keyword 1, then sign text 2, then plain objects 3.
  const priority=t.item.critical?0:searching||t.item.kind==='hazard'?1:t.item.kind==='text'?2:3;
  // A simulated demo scene is never announced as if the detector had seen it.
  const text=simulatedPrefix()+(t.item.critical?'Warning. '+describe(t.item):searching?`I found ${query}. `+describe(t.item):describe(t.item));
  const spoke=()=>{memory.markSpoken(t,Date.now());if(searching){t.searchSpoken=query;state.searchFound=true;}if(t.item.critical)flashHazard();};
  // Muted: warnings, hazards and the searched keyword go to the screen-reader region (critical ones still vibrate); the rest stays quiet.
  if(state.muted){if(priority<=1){state.lastSpeech=text;show(text,true);spoke();}return;}
  speech.enqueue({key:`track-${t.id}-${version}-${searching?query:''}`,text,priority,
    expiresAt:Date.now()+options.max_result_age_ms,
    valid:()=>generation===workflow.generation&&memory.valid(t,version,Date.now()),onStart:spoke});
}
function flashHazard(){
  try{navigator.vibrate?.([120,60,120]);}catch{}
  const el=$('hazardFlash');el.style.display='block';setTimeout(()=>el.style.display='none',1200);
}
function finishSummary(force=false){
  const summary=state.summary;if(!summary||!force&&summary.done.size<2)return;
  state.summary=null;state.phase=workflow.mode==='SEARCH'?'SEARCHING':'SCANNING';
  // o and x: which lane analysed a frame taken after the request. A scan that nothing answered is never reported as an empty scene.
  const o=summary.done.has('objects'),x=summary.done.has('text');
  const tracks=!o&&!x?[]:Object.values(state.cache).filter(Boolean).flatMap(c=>c.tracks).filter(t=>!t.misses&&Date.now()-t.lastSeen<memory.ttl)
    .sort((a,b)=>Number(!!b.item.critical)-Number(!!a.item.critical)||byUrgency(a,b)).slice(0,3);
  const text=!o&&!x?'I could not scan the surroundings. The vision system is '+(state.ready?'not responding. I will keep trying.':'not ready yet. Scanning will begin automatically.'):
    simulatedPrefix()+(o&&x?'Environmental scan complete. ':`Environmental scan partly complete. ${o?'Text recognition':'Object detection'} did not respond. `)+
    (tracks.length?tracks.map(t=>describe(t.item)).join(' '):`No reliable ${o&&x?'objects or text':o?'objects':'text'} identified. Monitoring continues.`);
  // Staleness was filtered when the text was built; one flickering box must not cut the summary.
  const generation=workflow.generation,spoke=()=>tracks.forEach(t=>memory.markSpoken(t,Date.now()));
  if(state.muted){state.lastSpeech=text;show(text,true);spoke();}
  // Priority 1: notices and hazards wait for the answer the user asked for; only a critical warning pre-empts it.
  else speech.enqueue({key:'summary-'+summary.started,text,priority:1,expiresAt:Date.now()+options.max_result_age_ms+8000,
    valid:()=>state.live&&generation===workflow.generation,onStart:spoke});
  updateStatus();
}
async function environmentScan(){
  const wasLive=state.live;
  if(!wasLive)await selectMode('ENVIRONMENT');
  if(!state.live)return;
  state.phase='ENVIRONMENT_SCAN';state.summary={started:Date.now(),done:new Set()};
  // The summary can take a few seconds, so the request is acknowledged at once.
  if(wasLive)say('Scanning your surroundings.');
  objectLane.kick();textLane.kick();updateStatus();
}
function deadline(promise,ms){
  let timer;return Promise.race([promise,new Promise((_,reject)=>{timer=setTimeout(()=>reject(new Error('Operation timed out')),ms);})]).finally(()=>clearTimeout(timer));
}
// micOff is true only when the user asked for the microphone to go off (stop microphone, close app, Stop/Close buttons).
async function selectMode(mode,query='',micOff=false){
  if(mode==='ENVIRONMENT_SCAN'){await environmentScan();return;}
  const declined=workflow.awaiting==='MAPS',visual=['ENVIRONMENT','SEARCH'].includes(mode);
  // Leaving a visual mode ends hazard monitoring, and the user is always told so.
  const stopped=state.live&&!visual?'Scanning stopped. Camera is off. ':'';
  reveal();workflow.transition(mode,query);const generation=workflow.generation;
  speech.cancel();state.summary=null;state.mapUrl=null;state.mapsAsked=0;$('mapLink').hidden=true;
  $('modeChoices').hidden=mode!=='MAIN_MENU';$('queryForm').hidden=!['SEARCH','NAVIGATION'].includes(mode);
  $('queryInput').value=query;$('queryLabel').textContent=mode==='NAVIGATION'?'Where would you like to go?':'What would you like me to find?';
  $('stage').hidden=!visual;$('controls').hidden=!visual;
  $('modeStatus').textContent={MAIN_MENU:'Choose assistance',ENVIRONMENT:'Live monitoring',SEARCH:'Keyword search',NAVIGATION:'Navigation',STOPPED:'Assistance stopped'}[mode];
  if(!visual)stopResources();
  if(mode==='STOPPED'){
    state.phase='IDLE';
    // Stop ends scanning but keeps listening, so "start scanning" works again without a tap.
    // The scanning controls are hidden while stopped, so the touch instruction names the buttons that are on screen.
    const touch=' Press Back to main menu, then Scan environment, to resume.';
    if(micOff){speech.stop();say('Stopped. Camera and microphone are off.'+touch);}
    else say('Scanning stopped. Camera is off.'+(speech.enabled?' Say start scanning to resume.':touch));
    updateStatus();return;
  }
  if(mode==='MAIN_MENU'){state.phase='IDLE';say(stopped+(declined?'Okay, Google Maps was not opened. ':'')+
    (speech.enabled?'Say start scanning, two for keyword search, or three for navigation.':'Choose Scan environment, Keyword search or Navigation.'));updateStatus();return;}
  // With voice on, the input is not focused: a screen reader would read its label over the same spoken question.
  if(mode==='NAVIGATION'){
    state.phase='NAVIGATING';if(query)await navigateTo(query,stopped);else{say(stopped+'Where would you like to go?');if(!speech.enabled)$('queryInput').focus();}updateStatus();return;
  }
  if(mode==='SEARCH'){state.searchStarted=Date.now();state.searchMissSpoken=false;state.searchFound=false;
    if(!query){say('What would you like me to find?');if(!speech.enabled)$('queryInput').focus();updateStatus();return;}}
  if(state.live){state.phase=mode==='SEARCH'?'SEARCHING':'SCANNING';say(mode==='SEARCH'?'Searching for '+query:'Monitoring continues.');objectLane.kick();textLane.kick();updateStatus();return;}
  state.phase='STARTING_SCAN';say('Starting the camera and live monitoring.');updateStatus();
  try{
    await configReady;
    // Health first: it tells us whether the server is replaying recorded scenes (no camera needed).
    const ok=state.ready||await checkHealth();
    if(generation!==workflow.generation)return;
    if(!state.demo&&(!await startCamera(generation)||generation!==workflow.generation))return;
    session=crypto.randomUUID?.()||Math.random().toString(36).slice(2);memory.clear();clearView();
    state.live=true;state.phase=mode==='SEARCH'?'SEARCHING':'SCANNING';state.demoStarted=Date.now();viewShape='';
    const lead=state.demo?'Demo mode. I am replaying recorded scenes, not the live camera. ':'';
    // Never claim monitoring has started while the vision models are still loading.
    say(ok?lead+(mode==='SEARCH'?'Searching for '+query:state.demo?'Scanning started.':'Live monitoring started.'):
      (lead||'Camera is on. ')+'The vision system is not ready yet. Scanning will begin automatically.');
    objectLane.interval=options.frame_interval;textLane.interval=options.ocr_interval_ms;
    objectLane.start();textLane.start();
  }catch(e){
    if(generation!==workflow.generation)return;
    workflow.generation++;stopResources();state.phase='ERROR';say(`Camera unavailable. Allow camera access in browser settings and use HTTPS, then ${speech.enabled?'say':'press'} start scanning to retry.`);
  }finally{updateStatus();}
}
async function navigateTo(destination,lead=''){
  if(!destination){say('Please say navigate to, followed by a destination.');return;}
  workflow.destination=destination;const generation=workflow.generation;
  setStatus('Getting location','busy');say(lead+'Getting your location for '+destination+'.');
  const locationResult=await locate(navigator.geolocation,{timeout:options.location_timeout_ms});
  if(generation!==workflow.generation)return;
  // Real distance and travel time from the server's geocoder and router, spoken before the handoff.
  // Nothing is estimated here: when the lookup fails, Maps is offered without numbers.
  let route=null,failed='';
  try{
    const body={destination};
    if(locationResult.coords){body.latitude=locationResult.coords.latitude;body.longitude=locationResult.coords.longitude;}
    const r=await fetch('/api/navigation',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body),signal:AbortSignal.timeout(15000)});
    if(r.ok)route=await r.json();
    else failed=r.status===429?'The directions service is busy right now. ':'The directions service returned an error. ';
  }catch(e){failed='I could not reach the directions service. ';}
  if(generation!==workflow.generation)return;
  const resolved=!!(route&&route.resolved===true&&typeof route.speech==='string'&&route.maps_url);
  state.mapUrl=resolved?route.maps_url:mapsUrl(destination,locationResult.coords);$('mapLink').href=state.mapUrl;$('mapLink').hidden=false;
  state.routeSpeech=resolved?route.speech:'';setStatus(resolved?'Route found':'Directions ready','ok');
  const indoor=state.schema&&state.schema!=='coco'?'Your object model does not detect vehicles. ':'';
  // The server names the real cause (place not found, no location, router down); say that, not a generic line.
  const reason=!resolved&&route&&route.error&&typeof route.speech==='string'?route.speech+' ':failed;
  const text=(resolved?route.speech+' ':reason?reason+'Google Maps may still find it. ':(locationResult.error?locationResult.error+' Google Maps can choose your starting point. ':'')+
    'I could not get the distance and travel time to '+destination+'. Google Maps can show them. ')+indoor+'Would you like me to open Google Maps? Say yes or no.'; // the last words are not themselves a command
  // Maps opens only on an explicit yes or "open maps" (or the visible link); a finished or interrupted sentence never navigates away.
  workflow.awaiting='MAPS';
  say(text,{onDone:()=>{if(generation===workflow.generation)state.mapsAsked=Date.now();}});
}
function mapsPromptExpired(){
  // An unanswered question stops waiting after 30 s, so a much later stray "okay" cannot open Maps.
  if(workflow.awaiting!=='MAPS'||!state.mapsAsked)return;
  // The window restarts while the app itself is speaking (repeat, help), so the answer to a repeated question still counts.
  if(speech.active){state.mapsAsked=Date.now();return;}
  if(Date.now()-state.mapsAsked<30000)return;
  workflow.awaiting=null;state.mapsAsked=0;say('Google Maps was not opened. Say open maps when you are ready, or say menu.');
}
function openMaps(){
  if(!state.mapUrl)return;
  const generation=workflow.generation;workflow.awaiting=null;state.mapsAsked=0;
  // Same-tab navigation works from voice callbacks without popup permissions. Stop or menu cancels the handoff.
  say('Opening Google Maps.',{onDone:()=>{
    if(generation!==workflow.generation||!state.mapUrl)return;
    try{window.location.assign(state.mapUrl);}
    catch(e){setStatus('Google Maps could not open','err');say('Google Maps could not open. Say open maps to retry.');}
  }});
}
function handleCommand(raw){
  const command=workflow.command(raw);if(!command)return;
  const [intent,target]=command;
  if(state.phase==='STARTING_SCAN'&&intent==='ENVIRONMENT')return;
  if(intent==='REPEAT'){if(state.lastSpeech)say(state.lastSpeech);return;}
  if(intent==='NAV_INFO'){
    if(state.live){ // A question never stops hazard monitoring: answer from what is in view right now.
      const fresh=memory.fresh(Date.now()),searching=workflow.mode==='SEARCH'&&workflow.query;
      const hit=searching?fresh.find(t=>t.searchMatch===workflow.query):fresh.filter(t=>t.item.steps).sort((a,b)=>a.item.steps-b.item.steps)[0];
      say(hit?simulatedPrefix()+(searching?'':'Closest: ')+describe(hit.item):searching?`I have not found ${workflow.query} yet.`:
        state.routeSpeech?'Last route result. '+state.routeSpeech:'Nothing with a measurable distance is in view. For a route, say navigate to, followed by a destination.');
      return;
    }
    selectMode('NAVIGATION',workflow.destination);return;
  }
  if(intent==='OPEN_MAPS'){if(state.mapUrl)openMaps();else selectMode('NAVIGATION',workflow.destination);return;}
  if(intent==='HELP'){say('Say start scanning, scan environment, find or where is followed by a sign, navigate to a destination, how far is it, open maps, stop, cancel, or repeat. Say stop microphone to turn listening off.');return;}
  if(intent==='MUTE'||intent==='UNMUTE'){setMuted(intent==='MUTE');return;}
  if(intent==='HAZARDS'){$('hazOnly').click();return;}
  // The measured field of view is typed in a visual dialog, so the voice command explains instead of opening it.
  if(intent==='CALIBRATE'){say('Calibration needs a measured camera field of view typed on screen. Ask a sighted helper to press the Calibrate button. Distances stay approximate without it.');return;}
  if(intent==='CLOSE'){selectMode('STOPPED','',true);return;}
  // Cancel drops a search or a pending question. It never switches live hazard monitoring off; menu, back and stop do that.
  if(intent==='CANCEL'){if(!state.live)selectMode('MAIN_MENU');else if(workflow.mode==='SEARCH')selectMode('ENVIRONMENT');return;}
  selectMode(intent,target||'');
}
function setMuted(value){
  // The label stays "Mute voice output"; aria-pressed and the icon carry the state.
  state.muted=value;$('muteBtn').setAttribute('aria-pressed',value);$('muteBtn').textContent=value?'🔇':'🔊';
  // Muting is confirmed aloud once (straight to the speech manager), so a silent app is never mistaken for a dead one.
  if(value)speech.say('Sound off. Hazard warnings are silenced too and appear on screen only. Say unmute to turn sound back on.');else say('Sound on.');
}
function toggleVoice(){
  const was=speech.enabled;speech.listen(!was);
  if(was)say('Voice commands off. Use the Enable voice button to turn them on.');
  else if(speech.enabled)say('Voice commands enabled.');
  else if(!speech.rec)say('Voice commands are not available in this browser. Use the on-screen buttons.');
}
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
$('menuBtn').onclick=()=>selectMode('MAIN_MENU');$('stopAll').onclick=()=>selectMode('STOPPED','',true);
$('repeatBtn').onclick=()=>state.lastSpeech&&say(state.lastSpeech);
$('liveBtn').onclick=()=>selectMode(state.live?'STOPPED':'ENVIRONMENT');
$('scanOnce').onclick=environmentScan;$('navBtn').onclick=()=>selectMode('NAVIGATION');
$('closeBtn').onclick=()=>selectMode('STOPPED','',true);$('calibChip').onclick=calibrate;
$('muteBtn').onclick=()=>setMuted(!state.muted);
$('hazOnly').onclick=()=>{state.hazOnly=!state.hazOnly;$('hazOnly').classList.toggle('tog',state.hazOnly);$('hazOnly').setAttribute('aria-pressed',state.hazOnly);say(state.hazOnly?'Hazard announcements only.':'All new information enabled.');};
document.querySelectorAll('[data-mode]').forEach(b=>b.onclick=()=>selectMode(b.dataset.mode==='ENVIRONMENT'?'ENVIRONMENT_SCAN':b.dataset.mode));
document.querySelectorAll('.chip[data-k]').forEach(b=>b.onclick=()=>selectMode('SEARCH',b.dataset.k));
$('queryForm').onsubmit=e=>{
  e.preventDefault();const target=$('queryInput').value.trim();if(!target)return;
  // A typed yes, no or open maps answers the Maps question, also after its 30 s voice window; anything else is a new target.
  const c=workflow.command(target);
  if(c&&(workflow.awaiting==='MAPS'||c[0]==='OPEN_MAPS'&&state.mapUrl))handleCommand(target);
  else if(state.mapUrl&&/^(yes|ok|okay)$/i.test(target))openMaps();
  else if(state.mapUrl&&/^no$/i.test(target))selectMode('MAIN_MENU');
  else selectMode(workflow.mode,target);
};
window.addEventListener('deviceorientation',e=>{
  devicePitch=e.beta!=null&&Math.abs(screen.orientation?.angle||0)%180===0?Math.max(-10,Math.min(70,90-e.beta)):0;
});
// iOS Safari delivers tilt only after a permission request made inside a user gesture; elsewhere this is a no-op.
document.addEventListener('click',()=>{
  try{window.DeviceOrientationEvent?.requestPermission?.().catch(()=>{});}catch{}
  // A greeting the browser blocked before any gesture is replayed on the first tap that is not the Start button.
  if(workflow.mode==='WELCOME'&&state.speechFailed&&state.lastSpeech)say(state.lastSpeech);
},{once:true});
document.addEventListener('visibilitychange',()=>{
  // Visibility is not a Stop command. Keep the voice session's listening intent.
  if(document.hidden){clearView('objects');clearView('text');}
});
window.addEventListener('pagehide',()=>{workflow.generation++;stopResources();speech.stop();});
const expiryTimer=setInterval(()=>{
  drawCurrent();if(state.summary&&Date.now()-state.summary.started>12000)finishSummary(true);
  // Retry any eligible unspoken current items after the previous utterance finishes, in the same order as a new frame.
  if(state.live)for(const t of memory.fresh(Date.now()).sort(byUrgency))announceTrack(t);
  mapsPromptExpired();updateStatus();
},250);
window.addEventListener('load',async()=>{
  // The greeting reports the real backend state instead of always claiming to be ready.
  const ok=await checkHealth();
  // The user already acted while the check was pending: Start arms the microphone itself, and Stop must stay stopped.
  if(workflow.mode!=='WELCOME')return;
  // A browser without speech recognition is told to use the button instead of being asked to speak.
  const begin=speech.rec?'Please say start scanning':'Voice commands are not available in this browser. Press the Start scanning button';
  say('Welcome to Nexora. '+(ok?(state.demo?'Demo mode is on. ':'Your visual assistant is ready. ')+begin+' to begin.':
    'The vision system is not ready yet. '+begin+' and I will begin as soon as it is.'));
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

  const list=document.createElement('div');
  const short=dir=>dir.includes('left')?'← LEFT':dir.includes('right')?'RIGHT →':'↑ AHEAD';
  const add=(it,cls)=>{
    const div=document.createElement('div');div.className='hit '+cls;
    if(it.confirmed===false){div.style.opacity='.45';div.style.borderStyle='dashed';}
    div.innerHTML='<span class="t"></span><span class="d"></span>';
    // The kind is written out, not shown by border colour alone.
    div.querySelector('.t').textContent=(cls==='hazard'?'HAZARD · ':cls==='match'?'MATCH · ':'')+it.label;
    div.querySelector('.d').textContent=(it.confirmed===false?'CONFIRMING… ':'')+short(it.direction)+
      (it.steps?(it.steps>30?' · FAR':' · ≈'+it.steps+' STEP'+(it.steps>1?'S':'')):
       it.proximity==='very close'?' · CLOSE':it.proximity==='nearby'?' · NEAR':'');
    list.appendChild(div);
  };
  d.hazards.forEach(x=>add(x,'hazard'));
  if(d.match)add(d.match,'match');
  d.symbols.slice(0,4).forEach(x=>{if(!d.match||x.label!==d.match.label)add(x,'symbol');});
  d.texts.slice(0,5).forEach(x=>add(x,'text'));
  d.objects.slice(0,5).forEach(x=>add(x,'object'));
  // Rebuilt only when the content changed: no animation restart or screen-reader churn every 250 ms.
  if(list.innerHTML!==resultsEl.innerHTML)resultsEl.replaceChildren(...list.children);
}

