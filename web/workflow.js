/* Pure state and speech scheduling logic, shared with Node regression tests. */
(function (root) {
  class Workflow {
    constructor() { this.mode='WELCOME';this.query='';this.destination='';this.awaiting=null;this.generation=0; }
    transition(mode,query='') {
      if(!['WELCOME','MAIN_MENU','ENVIRONMENT','SEARCH','NAVIGATION','STOPPED'].includes(mode))throw new Error('Unknown assistance mode');
      this.mode=mode;this.query=mode==='SEARCH'?query:'';
      if(mode==='NAVIGATION'&&query)this.destination=query;
      this.awaiting=!query&&['SEARCH','NAVIGATION'].includes(mode)?mode:null;
      this.generation++;return mode;
    }
    command(raw) {
      const q=raw.toLowerCase().trim().replace(/[.!?]+$/,'').replace(/\s+/g,' ');
      if(/^(stop|pause|stop scanning|stop scan|stop camera|stop microphone)$/.test(q))return ['STOPPED'];
      if(/^(back|go back|main menu|menu)$/.test(q))return ['MAIN_MENU'];
      if(/^(repeat|say that again)$/.test(q))return ['REPEAT'];
      if(/^(start|start scanning|start scan|begin|begin scanning|let's start|resume scanning|scan)$/.test(q))return ['ENVIRONMENT'];
      if(/^(scan environment|describe environment|describe surroundings|what is around me|environment scan)$/.test(q))return ['ENVIRONMENT_SCAN'];
      if(/^(distance and time|distance & time|how far is it|travel time|route distance|estimated travel time)$/.test(q))return ['NAV_INFO'];
      if(/^(one|1|environment|environment summary|summary)$/.test(q))return ['ENVIRONMENT_SCAN'];
      if(/^(two|2|search|keyword search|find)$/.test(q))return ['SEARCH'];
      if(/^(three|3|navigation|navigate)$/.test(q))return ['NAVIGATION'];
      const search=q.match(/^(?:find|locate|search for)\s+(.+)$/);
      if(search)return ['SEARCH',search[1]];
      const nav=q.match(/^(?:go to|navigate to|take me to|directions to)\s+(.+)$/);
      if(nav)return ['NAVIGATION',nav[1]];
      for(const [pattern,intent] of [[/^(help|commands|what can you do)$/,'HELP'],[/^(mute|be quiet|silence)$/,'MUTE'],[/^(unmute|sound on)$/,'UNMUTE'],[/^(close app|goodbye|turn off)$/,'CLOSE'],[/^(calibrate|calibration)$/,'CALIBRATE'],[/^hazards only$/,'HAZARDS']])if(pattern.test(q))return [intent];
      // Bare words are accepted only directly after the app asked for a target.
      if(this.awaiting&&q.length<=200)return [this.awaiting,q];
      return null;
    }
  }
  root.AssistiveWorkflow={Workflow};
  if(typeof module!=='undefined')module.exports=root.AssistiveWorkflow;
})(typeof window!=='undefined'?window:globalThis);
