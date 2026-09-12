const {test}=require('node:test');
const assert=require('node:assert/strict');
const {DetectionMemory,LatestLane,locate,mapsUrl}=require('../web/realtime.js');
const {Workflow}=require('../web/workflow.js');
const frame={w:1000,h:1000};
const chair=(x=100)=>({label:'chair',raw:'chair',box:[x,100,x+100,300],conf:.9,steps:5,kind:'object',direction:'on your left'});
test('ten unchanged frames speak once; disappearance removes and reappearance announces',()=>{
  const m=new DetectionMemory();let spoken=0;
  for(let n=0;n<10;n++){
    const [t]=m.update('objects',[chair()],frame,n*500);
    if(t.stable&&!t.spoken){m.markSpoken(t,n*500);spoken++;}
  }
  assert.equal(spoken,1);
  assert.equal(m.update('objects',[],frame,5100).length,0);
  m.update('objects',[],frame,5200);m.update('objects',[],frame,5300);assert.equal(m.tracks.size,0);
  m.update('objects',[chair()],frame,5400);const [again]=m.update('objects',[chair()],frame,5500);
  assert.equal(again.stable,true);assert.equal(again.spoken,null);
});
test('unchanged EXIT never expires while observed; noise does not reset speech but meaningful motion does',()=>{
  const m=new DetectionMemory();let t;
  for(let n=0;n<10;n++){
    [t]=m.update('text',[{...chair(100+n%2*3),raw:undefined,label:'EXIT',kind:'text'}],frame,n*1000);
    if(n===1)m.markSpoken(t,n*1000);
  }
  assert.ok(t.spoken);assert.equal(m.tracks.size,1);
  [t]=m.update('text',[{...chair(300),raw:undefined,label:'EXIT',kind:'text'}],frame,10000);
  assert.equal(t.spoken,null);
});
test('distinct same-class objects track one-to-one; critical bypasses persistence and TTL removes stale state',()=>{
  const m=new DetectionMemory({ttl:1000});
  const tracks=m.update('objects',[chair(100),{...chair(700),critical:true}],frame,0);
  assert.equal(tracks[0].stable,false);assert.equal(tracks[1].stable,true);assert.equal(m.tracks.size,2);
  m.expire(1001);assert.equal(m.tracks.size,0);
});
test('failed or obsolete lane hides detections and invalidates pending speech immediately',()=>{
  const m=new DetectionMemory();m.update('objects',[chair()],frame,0);
  const [t]=m.update('objects',[chair()],frame,10);
  assert.equal(m.valid(t,t.version,20),true);m.hide('objects');
  assert.equal(m.fresh(20).length,0);assert.equal(m.valid(t,t.version,20),false);
});
test('latest lane permits only one job; stop/restart invalidates obsolete result without backlog',async()=>{
  let release,ctx,calls=0;const scheduled=[];
  const lane=new LatestLane(c=>{calls++;ctx=c;return new Promise(r=>release=r);},
    {setTimer:fn=>{scheduled.push(fn);return scheduled.length;},clearTimer:()=>{}});
  lane.start();lane.start();lane.kick();assert.equal(calls,1);
  lane.stop();assert.equal(ctx.current(),false);assert.equal(ctx.signal.aborted,true);
  lane.start();assert.equal(calls,1);release();await new Promise(r=>setImmediate(r));
  scheduled.shift()();assert.equal(calls,2);lane.stop();release();
});
test('command parser handles all requested intents and never turns a spoken description into a command',()=>{
  const w=new Workflow();
  for(const phrase of ['start','start scan','start scanning','begin scanning'])assert.deepEqual(w.command(phrase),['ENVIRONMENT']);
  for(const phrase of ['scan environment','describe environment','what is around me'])assert.deepEqual(w.command(phrase),['ENVIRONMENT_SCAN']);
  for(const target of ['washroom','pharmacy','exit','emergency exit','room 101','stairs','reception'])assert.deepEqual(w.command('find '+target),['SEARCH',target]);
  for(const phrase of ['stop camera','stop microphone','stop scanning'])assert.deepEqual(w.command(phrase),['STOPPED']);
  assert.deepEqual(w.command('take me to Mysore Palace'),['NAVIGATION','mysore palace']);
  w.transition('NAVIGATION','Mysore Palace');w.transition('MAIN_MENU');assert.equal(w.destination,'Mysore Palace');
  for(const phrase of ['distance and time','how far is it','travel time'])assert.deepEqual(w.command(phrase),['NAV_INFO']);
  assert.equal(w.command('There is a chair two meters ahead.'),null);
  w.transition('SEARCH','washroom');assert.equal(w.command('There is a chair two meters ahead.'),null);
});
test('geolocation cannot hang; denial falls back; URL encodes destination and uses actual origin',async()=>{
  let fire;
  const pending=locate({getCurrentPosition(){}},{setTimer:fn=>{fire=fn;return 1;},clearTimer:()=>{}});
  fire();assert.equal((await pending).error,'Location timed out.');
  assert.equal((await locate({getCurrentPosition(ok,fail){fail({code:1});}})).error,'Location permission denied.');
  const value=await locate({getCurrentPosition(ok){ok({coords:{latitude:12,longitude:76}});}});
  const url=new URL(mapsUrl('Mysore Palace & museum',value.coords));
  assert.equal(url.searchParams.get('destination'),'Mysore Palace & museum');assert.equal(url.searchParams.get('origin'),'12,76');
  assert.equal(url.searchParams.get('dir_action'),'navigate');
  assert.equal(new URL(mapsUrl('Palace')).searchParams.has('origin'),false);
});
