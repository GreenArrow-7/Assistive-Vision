const {test}=require('node:test');
const assert=require('node:assert/strict');
const {DetectionMemory,LatestLane,locate,mapsUrl,textIsImportant}=require('../web/realtime.js');
const {Workflow}=require('../web/workflow.js');
const frame={w:1000,h:1000};
const chair=(x=100)=>({label:'chair',raw:'chair',box:[x,100,x+100,300],conf:.9,steps:5,kind:'object',direction:'on your left'});
test('ten unchanged frames speak once; a brief gap is remembered, a long one is not',()=>{
  const m=new DetectionMemory();let spoken=0;
  for(let n=0;n<10;n++){
    const [t]=m.update('objects',[chair()],frame,n*500);
    if(t.stable&&!t.spoken){m.markSpoken(t,n*500);spoken++;}
  }
  assert.equal(spoken,1);
  assert.equal(m.update('objects',[],frame,5100).length,0);
  // A brief disappearance -- someone walks past, the detector drops a frame -- must not make the
  // object new again, or the same sentence is spoken twice with nothing having changed.
  m.update('objects',[],frame,5200);m.update('objects',[],frame,5300);assert.equal(m.tracks.size,1);
  m.update('objects',[chair()],frame,5400);const [again]=m.update('objects',[chair()],frame,5500);
  assert.equal(again.stable,true);assert.ok(again.spoken,'a briefly occluded object is still remembered');
  // Gone for longer than the track lifetime, it genuinely is a new sighting.
  m.update('objects',[],frame,13000);
  assert.equal(m.tracks.size,0);
  m.update('objects',[chair()],frame,13500);const [fresh]=m.update('objects',[chair()],frame,14000);
  assert.equal(fresh.spoken,null);
});
test('drift is not narrated every frame, but an object closing in still is',()=>{
  const m=new DetectionMemory();const said=[];
  // steps fall from 10 to 2 while the box grows, as it does when you walk towards something
  for(const [n,steps] of [10,10,9,8,7,6,5,4,3,2].entries()){
    const r=10*(11-steps);
    const [t]=m.update('objects',[{...chair(),steps,box:[500-r,500-r,500+r,500+r]}],frame,n*350);
    if(t.stable&&!t.spoken){said.push(steps);m.markSpoken(t,n*350);}
  }
  // halving the distance is worth saying; every intermediate step is not
  assert.deepEqual(said,[10,5,2]);

  // the same object, not moving, is announced once however long it is watched
  const still=new DetectionMemory();let count=0;
  for(let n=0;n<60;n++){
    const [t]=still.update('objects',[{...chair(),steps:4}],frame,n*350);
    if(t.stable&&!t.spoken){count++;still.markSpoken(t,n*350);}
  }
  assert.equal(count,1,'a motionless object was re-announced');
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
  for(const phrase of ['stop camera','stop scanning'])assert.deepEqual(w.command(phrase),['STOPPED']);
  // Only an explicit microphone request ends listening; stop scanning keeps the voice loop alive.
  assert.deepEqual(w.command('stop microphone'),['CLOSE']);
  assert.deepEqual(w.command('where is the washroom'),['SEARCH','the washroom']);
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
test('signs and safety words are important text; incidental small print is not',()=>{
  for(const label of ['EXIT','Fire exit','Room 205','PHARMACY','GATE B2','CAFE'])assert.equal(textIsImportant(label),true,label);
  for(const label of ['Terms and conditions apply to all offers','best before see cap','hello'])assert.equal(textIsImportant(label),false,label);
});
