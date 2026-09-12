const {test}=require('node:test');
const assert=require('node:assert/strict');
const {Workflow}=require('../web/workflow.js');
test('welcome, modes, back, stop, repeat and natural commands',()=>{
  const w=new Workflow();assert.equal(w.mode,'WELCOME');
  for(const [q,mode] of [['Start','ENVIRONMENT'],['one','ENVIRONMENT_SCAN'],['two','SEARCH'],['three','NAVIGATION'],['back','MAIN_MENU'],['stop','STOPPED'],['repeat','REPEAT']]){
    const c=w.command(q);assert.equal(c[0],mode);if(!['REPEAT','ENVIRONMENT_SCAN'].includes(mode))w.transition(...c);
  }
  assert.deepEqual(w.command('find room 205'),['SEARCH','room 205']);
  assert.deepEqual(w.command('go to Bangalore'),['NAVIGATION','bangalore']);
  w.transition('SEARCH');assert.deepEqual(w.command('washroom'),['SEARCH','washroom']);
});
