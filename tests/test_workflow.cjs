const {test}=require('node:test');
const assert=require('node:assert/strict');
const {Workflow,AnnouncementQueue,menuDigit,isEcho}=require('../web/workflow.js');
test('menu digits survive real recogniser output, at any point',()=>{
  for(const [heard,d] of [['4 4','4'],['number four','4'],['for','4'],['to','2'],['tree','3'],['option 5','5'],['one.','1'],['4 2',null],['find room 4',null]])
    assert.equal(menuDigit(heard.toLowerCase().replace(/[^a-z0-9' ]+/g,' ').trim()),d,heard);
  const w=new Workflow();
  assert.deepEqual(w.command('4 4'),['MAIN_MENU']);assert.deepEqual(w.command('number five'),['STOPPED']);
  w.transition('SEARCH');                      // awaiting a keyword: digits still win, words pass through
  assert.deepEqual(w.command('four'),['MAIN_MENU']);assert.deepEqual(w.command('pharmacy'),['SEARCH','pharmacy']);
  assert.deepEqual(w.command('find room 4'),['SEARCH','room 4']);
  w.transition('ENVIRONMENT');assert.deepEqual(w.command('2'),['SEARCH']);assert.deepEqual(w.command('3'),['NAVIGATION']);
});
test('barge-in echo detection',()=>{
  assert.equal(isEcho('i can see chair','I can see: chair, about two steps straight ahead.'),true);
  assert.equal(isEcho('four','I can see: chair straight ahead.'),false);
  assert.equal(isEcho('stop scanning','Caution. person straight ahead.'),false);
  assert.equal(isEcho('',''),false);
});
test('welcome, modes, back, stop, repeat and natural commands',()=>{
  const w=new Workflow();assert.equal(w.mode,'WELCOME');
  for(const [q,mode] of [['Start','MAIN_MENU'],['one','ENVIRONMENT'],['two','SEARCH'],['three','NAVIGATION'],['back','MAIN_MENU'],['stop','STOPPED'],['repeat','REPEAT']]){
    const c=w.command(q);assert.equal(c[0],mode);if(mode!=='REPEAT')w.transition(...c);
  }
  assert.deepEqual(w.command('find room 205'),['SEARCH','room 205']);
  assert.deepEqual(w.command('go to Bangalore'),['NAVIGATION','bangalore']);
  assert.deepEqual(w.command('take me to the pharmacy'),['NAVIGATION','the pharmacy']);
  w.transition('SEARCH');assert.deepEqual(w.command('washroom'),['SEARCH','washroom']);
});
test('numbered home menu: 4 = home, 5 = stop',()=>{
  const w=new Workflow();w.transition('ENVIRONMENT');
  for(const q of ['four','4','home','go back'])assert.deepEqual(w.command(q),['MAIN_MENU'],q);
  for(const q of ['five','5','stop scanning'])assert.deepEqual(w.command(q),['STOPPED'],q);
});
test('voice start variants reach scanning, not just the menu',()=>{
  const w=new Workflow();
  for(const q of ['start scanning','start scan','begin scanning','scan'])
    assert.deepEqual(w.command(q),['ENVIRONMENT'],q);
});
test('distance-and-time intent outranks the NAVIGATION destination passthrough',()=>{
  const w=new Workflow();
  assert.deepEqual(w.command('distance and time'),['DISTANCE_TIME']);
  assert.deepEqual(w.command('travel time'),['DISTANCE_TIME']);
  w.transition('NAVIGATION');
  // while awaiting a destination, this phrase must NOT become the destination
  assert.deepEqual(w.command('Distance and time.'),['DISTANCE_TIME']);
  assert.deepEqual(w.command('how far is it'),['DISTANCE_TIME']);
  // but a real destination still passes through
  assert.deepEqual(w.command('city hospital'),['NAVIGATION','city hospital']);
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
