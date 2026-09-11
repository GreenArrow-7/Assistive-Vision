const {test}=require('node:test');
const assert=require('node:assert/strict');
const {Workflow,AnnouncementQueue}=require('../web/workflow.js');
test('welcome, modes, back, stop, repeat and natural commands',()=>{
  const w=new Workflow();assert.equal(w.mode,'WELCOME');
  for(const [q,mode] of [['Start','MAIN_MENU'],['one','ENVIRONMENT'],['two','SEARCH'],['three','NAVIGATION'],['back','MAIN_MENU'],['stop','STOPPED'],['repeat','REPEAT']]){
    const c=w.command(q);assert.equal(c[0],mode);if(mode!=='REPEAT')w.transition(...c);
  }
  assert.deepEqual(w.command('find room 205'),['SEARCH','room 205']);
  assert.deepEqual(w.command('go to Bangalore'),['NAVIGATION','bangalore']);
  w.transition('SEARCH');assert.deepEqual(w.command('washroom'),['SEARCH','washroom']);
});
test('critical interruption, cooldown, expiry, and bounded backlog',()=>{
  const q=new AnnouncementQueue(9000);
  q.push('Chair',3,0);q.push('Warning',0,1);
  assert.equal(q.take(2).text,'Warning');assert.equal(q.take(3),undefined);
  assert.equal(q.push('Warning',0,5),false);
  assert.equal(q.push('Warning',0,10000),true);
  assert.equal(q.take(19000),undefined);
  q.push('A',3,20000);q.push('B',2,20000);q.push('C',1,20000);q.push('D',3,20000);
  assert.equal(q.pending.length,3);assert.equal(q.take(20001).text,'C');
});
