const {test}=require('node:test');
const assert=require('node:assert/strict');
const {SpeechManager}=require('../web/speech.js');
function fixture(){
  let now=0,next=0,instances=0;const timers=new Map(),commands=[],errors=[],spoken=[],states=[];
  const clock={set:(fn,ms)=>{timers.set(++next,{fn,at:now+ms});return next;},clear:id=>timers.delete(id),
    tick:ms=>{now+=ms;for(let i=0;i<20;i++){const due=[...timers].filter(([,v])=>v.at<=now);if(!due.length)break;for(const [id,v] of due){timers.delete(id);v.fn();}}}};
  class Recognition{
    constructor(){instances++;this.starts=0;this.aborts=0;}
    start(){this.starts++;}
    abort(){this.aborts++;}
    result(text){this.onresult({resultIndex:0,results:[Object.assign([{transcript:text}],{isFinal:true})]});}
  }
  const synth={speaking:false,cancel(){this.speaking=false;},speak(u){this.speaking=true;spoken.push(u);u.onstart?.();},getVoices(){return [];}};
  const manager=new SpeechManager({Recognition,synthesis:synth,Utterance:class{constructor(text){this.text=text;}},
    now:()=>now,setTimer:clock.set,clearTimer:clock.clear,onCommand:t=>commands.push(t),onError:t=>errors.push(t),onState:s=>states.push(s)});
  const end=()=>{synth.speaking=false;spoken.at(-1).onend?.();};
  return {manager,rec:manager.rec,clock,spoken,end,commands,errors,states,instances:()=>instances,synth};
}
test('recognition stops by event before TTS; delayed results and cooldown cannot echo',()=>{
  const f=fixture();f.manager.listen();f.rec.onstart();f.rec.result('start scanning');
  f.manager.say('There is a chair ahead.');
  assert.equal(f.rec.aborts,1);assert.equal(f.spoken.length,0);
  f.rec.result('There is a chair ahead.');assert.deepEqual(f.commands,['start scanning']);
  f.rec.onend();assert.equal(f.spoken.length,1);
  f.manager.listen();f.clock.tick(5000);assert.equal(f.rec.starts,1);
  f.end();f.clock.tick(649);assert.equal(f.rec.starts,1);
  f.clock.tick(1);assert.equal(f.rec.starts,2);f.rec.onstart();
  f.rec.result('There is a chair ahead.');assert.equal(f.commands.length,1);
  f.rec.result('find washroom');assert.equal(f.commands.at(-1),'find washroom');
  assert.equal(f.instances(),1);
});
test('critical interruption invalidates old completion callbacks and queued speech',()=>{
  const f=fixture();f.manager.listen();f.rec.onstart();
  f.manager.enqueue({text:'Chair',priority:3});f.rec.onend();const oldEnd=f.spoken[0].onend;
  f.manager.enqueue({text:'Warning obstacle',priority:0});
  assert.equal(f.spoken.length,2);oldEnd();f.clock.tick(700);assert.equal(f.rec.starts,1);
  f.end();f.clock.tick(650);assert.equal(f.rec.starts,2);
});
test('one instance and no restart after stop or microphone permission denial',()=>{
  const f=fixture();f.manager.listen();f.manager.listen();assert.equal(f.rec.starts,1);
  f.rec.onstart();f.manager.stop();f.rec.onend();f.clock.tick(3000);assert.equal(f.rec.starts,1);
  f.manager.listen();f.rec.onerror({error:'not-allowed'});f.rec.onend();f.clock.tick(3000);
  assert.equal(f.rec.starts,2);assert.equal(f.manager.enabled,false);assert.equal(f.instances(),1);
});
test('recognition stop watchdog fails closed rather than starting a second session',()=>{
  const f=fixture();f.manager.listen();f.rec.onstart();f.manager.say('Hello');
  f.clock.tick(2000);assert.equal(f.manager.enabled,true);assert.equal(f.spoken.length,1);
  f.end();f.manager.listen();f.clock.tick(5000);assert.equal(f.rec.starts,1);
  f.rec.onend();assert.equal(f.rec.starts,2);
});
test('TTS failure resumes after cooldown; obsolete and duplicate queue entries are discarded',()=>{
  const f=fixture();f.manager.listen();f.rec.onstart();f.manager.say('Hello');f.rec.onend();
  f.manager.enqueue({key:'x',text:'New',valid:()=>false});
  f.manager.enqueue({key:'x',text:'New'});assert.equal(f.manager.queue.length,1);
  f.synth.speaking=false;f.spoken[0].onerror();f.clock.tick(650);
  assert.equal(f.spoken.length,1);assert.equal(f.rec.starts,2);
});
test('native-like browser timers are called without the manager as receiver',()=>{
  let manager;
  function clearTimer(){'use strict';assert.notEqual(this,manager);}
  function setTimer(){'use strict';assert.notEqual(this,manager);return 1;}
  manager=new SpeechManager({clearTimer,setTimer});
  manager.say('Message');manager.stop();
});

test('multiple follow-ups resume even when native speaking flag lags utterance end',()=>{
  const f=fixture();f.manager.listen();f.rec.onstart();
  for(const command of ['help','find exit','scan environment','repeat']){
    f.rec.result(command);assert.equal(f.commands.at(-1),command);
    f.manager.say('Response to '+command);f.rec.onend();
    f.spoken.at(-1).onend(); // Browser can still report speaking inside this callback.
    assert.equal(f.synth.speaking,true);f.clock.tick(650);f.rec.onstart();
    assert.equal(f.manager.recState,'listening');
  }
  assert.equal(f.rec.starts,5);assert.equal(f.instances(),1);
  f.manager.stop();f.rec.onend();f.clock.tick(5000);assert.equal(f.rec.starts,5);
});

test('transient recognition teardown race retries without losing listening intent',()=>{
  const f=fixture();let attempts=0;
  f.rec.start=()=>{if(++attempts<3)throw Object.assign(new Error('busy'),{name:'InvalidStateError'});};
  f.manager.listen();assert.equal(f.manager.enabled,true);
  f.clock.tick(500);f.clock.tick(500);f.rec.onstart();
  f.rec.result('help');assert.deepEqual(f.commands,['help']);assert.equal(attempts,3);
  f.manager.stop();f.clock.tick(5000);assert.equal(attempts,3);
});
