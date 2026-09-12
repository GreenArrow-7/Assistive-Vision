/* DOM/API test doubles exercise the actual app entry point, not real model accuracy. */
const {test}=require('node:test');const assert=require('node:assert/strict');
const fs=require('node:fs');const vm=require('node:vm');
function fixture(){
  const nodes=new Map(),listeners={},requests=[],spoken=[],opened=[],timers=new Map();let timer=0,now=10000,cameraCalls=0;
  class Element{
    constructor(){this.style={};this.dataset={};this.classList={add(){},remove(){},toggle(){}};this.value='';this.readyState=1;this.videoWidth=640;this.videoHeight=480;this.children={};}
    querySelector(s){return this.children[s]??=new Element();} addEventListener(name,fn){(this.events??={})[name]=fn;} removeEventListener(){} focus(){} remove(){} replaceChildren(){} appendChild(){} setAttribute(){}
    getContext(){return new Proxy({measureText:()=>({width:20})},{get:(o,k)=>o[k]||(()=>{})});}
    toBlob(fn){fn(new Blob(['image']));} play(){return Promise.resolve();}
  }
  const document={getElementById:id=>{if(!nodes.has(id))nodes.set(id,new Element());return nodes.get(id);},createElement:()=>new Element(),
    querySelectorAll:()=>[],addEventListener:(name,fn)=>{listeners[name]=fn;},hidden:false};
  class Recognition{constructor(){fixture.rec=this;}start(){this.onstart?.();}abort(){this.onend?.();}}
  const synth={speaking:false,getVoices:()=>[],cancel(){this.speaking=false;},speak(u){this.speaking=true;spoken.push(u);u.onstart?.();}};
  const track={readyState:'live',stop(){this.readyState='ended';}};
  const response=(lane)=>({frame:{w:640,h:480},objects:[],hazards:[],texts:[],symbols:[],match:null,component_errors:[],ms:10,
    object_active:lane==='objects',ocr_active:lane==='text',frame_quality:'normal'});
  const context={console,document,URLSearchParams,AbortController,AbortSignal,Blob,FormData,performance,
    Date:class extends Date{static now(){return now;}},
    navigator:{mediaDevices:{getUserMedia:async()=>{cameraCalls++;track.readyState='live';return {getTracks:()=>[track]};}},
      geolocation:{getCurrentPosition:ok=>ok({coords:{latitude:12,longitude:76}})}},
    localStorage:{getItem:()=>null,setItem(){}},screen:{orientation:{angle:0}},crypto:{randomUUID:()=>String(Math.random())},
    location:{assign:url=>opened.push(url)},SpeechRecognition:Recognition,speechSynthesis:synth,SpeechSynthesisUtterance:class{constructor(text){this.text=text;}},
    setTimeout:(fn,ms)=>{timers.set(++timer,{fn,at:now+ms});return timer;},clearTimeout:id=>timers.delete(id),setInterval:()=>0,
    fetch:async(url,opts)=>{requests.push(url);return {ok:true,status:200,json:async()=>url.includes('config')?{}:url.includes('health')?{ready:true,object_schema:'coco'}:response(url.endsWith('objects')?'objects':'text')};},
    addEventListener:(name,fn)=>listeners[name]=fn};
  context.window=context;vm.createContext(context);
  for(const file of ['workflow','speech','realtime','app'])vm.runInContext(fs.readFileSync(`web/${file}.js`,'utf8'),context);
  const tick=ms=>{now+=ms;const due=[...timers].filter(([,t])=>t.at<=now);for(const [id,t] of due){timers.delete(id);t.fn();}};
  const flush=async()=>{for(let i=0;i<30;i++)await Promise.resolve();};
  const end=()=>{synth.speaking=false;spoken.at(-1)?.onend?.();};
  return {context,nodes,listeners,requests,spoken,opened,tick,flush,end,cameraCalls:()=>cameraCalls,
    run:s=>vm.runInContext(s,context),stop:()=>vm.runInContext("selectMode('STOPPED')",context)};
}
test('welcome -> spoken start scanning -> both inference lanes; Stop clears actual resources',async()=>{
  const f=fixture();f.listeners.load();assert.match(f.spoken[0].text,/say start scanning/);
  f.end();f.tick(650);
  fixture.rec.onresult({resultIndex:0,results:[Object.assign([{transcript:'start scanning'}],{isFinal:true})]});
  await f.flush();
  assert.equal(f.run('state.live'),true);assert.equal(f.cameraCalls(),1);
  assert.ok(f.requests.includes('/api/detect/objects'));assert.ok(f.requests.includes('/api/detect/text'));
  await f.stop();assert.equal(f.run('video.srcObject'),null);assert.equal(f.run('speech.enabled'),false);
  assert.match(f.nodes.get('componentStatus').textContent,/Camera: off/);
});
test('navigation opens Maps on TTS completion; distance and time reuses destination; stop cancels handoff',async()=>{
  const f=fixture();await f.run("selectMode('NAVIGATION','Mysore Palace')");
  assert.equal(f.opened.length,0);f.end();assert.equal(f.opened.length,1);
  assert.equal(new URL(f.opened[0]).searchParams.get('origin'),'12,76');
  f.run("handleCommand('distance and time')");await f.flush();f.end();assert.equal(f.opened.length,2);
  assert.equal(new URL(f.opened[1]).searchParams.get('destination'),'Mysore Palace');
  await f.run("selectMode('NAVIGATION','Other place')");await f.stop();f.end();assert.equal(f.opened.length,2);
});
test('one-time environment summary returns to monitoring and ignores repeat completions',async()=>{
  const f=fixture();await f.run("selectMode('ENVIRONMENT')");await f.flush();
  f.end();await f.run('environmentScan()');await f.flush();
  // Both independent requests have completed; an additional completion cannot resummarize.
  f.run('finishSummary(true)');f.end();f.run('finishSummary(true)');f.end();
  assert.equal(f.spoken.filter(u=>u.text.startsWith('Environmental scan complete.')).length,1);
  assert.equal(f.run('state.live'),true);assert.equal(f.run('state.summary'),null);await f.stop();
});
test('camera permission failure leaves no scanning jobs and reports a recoverable error',async()=>{
  const f=fixture();f.context.navigator.mediaDevices.getUserMedia=async()=>{throw new Error('denied');};
  await f.run("selectMode('ENVIRONMENT')");
  assert.equal(f.run('state.live'),false);assert.equal(f.run('state.phase'),'ERROR');
  assert.equal(f.requests.some(x=>x.includes('/api/detect/')),false);
  assert.match(f.spoken.at(-1).text,/Camera unavailable/);await f.stop();
});
test('late OCR response is discarded rather than painting an old camera frame',async()=>{
  const f=fixture();const normal=f.context.fetch;let release;
  f.context.fetch=(url,opts)=>url==='/api/detect/text'?new Promise(resolve=>release=resolve):normal(url,opts);
  await f.run("selectMode('ENVIRONMENT')");await f.flush();assert.ok(release);
  f.tick(5000);
  release({ok:true,json:async()=>({frame:{w:640,h:480},texts:[{label:'OLD SIGN',box:[0,0,100,100],kind:'text',conf:.9}],objects:[],hazards:[],symbols:[],component_errors:[],match:null})});
  await f.flush();assert.equal(f.run('state.cache.text'),null);assert.equal(f.run('state.textStatus'),'waiting for current frame');await f.stop();
});

test('Start supports repeated spoken commands; passive clicks and visibility do not trigger speech or stop voice',async()=>{
  const f=fixture();
  for(const id of ['stage','gate']){
    const element=f.context.document.getElementById(id);
    assert.equal(element.onclick,undefined);assert.equal(element.events?.click,undefined);
    assert.equal(element.onkeydown,undefined);
  }
  f.nodes.get('startBtn').onclick();await f.flush();f.end();f.tick(650);
  for(const command of ['help','find exit','find washroom','start scanning']){
    assert.equal(f.run('speech.recState'),'listening');
    fixture.rec.onresult({resultIndex:0,results:[Object.assign([{transcript:command}],{isFinal:true})]});
    await f.flush();f.end();f.tick(650);
    assert.equal(f.run('speech.enabled'),true);
  }
  assert.equal(f.cameraCalls(),1);
  const count=f.spoken.length;f.context.document.hidden=true;f.listeners.visibilitychange();
  assert.equal(f.spoken.length,count);assert.equal(f.run('speech.enabled'),true);
  f.context.document.hidden=false;f.listeners.visibilitychange();
  await f.stop();f.end();f.tick(5000);assert.equal(f.run('speech.enabled'),false);
});
