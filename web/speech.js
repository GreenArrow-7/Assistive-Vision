/* One recognizer, one utterance, and one restart timer. No UI-specific logic. */
(function(root){
  class SpeechManager {
    constructor({Recognition, synthesis, Utterance, onCommand=()=>{}, onState=()=>{},
      onError=()=>{}, onText=()=>{}, now=()=>Date.now(), setTimer=setTimeout,
      clearTimer=clearTimeout, cooldown=650, language='en-IN', rate=1.05}={}) {
      Object.assign(this,{synthesis,Utterance,onCommand,onState,onError,onText,now,setTimer,clearTimer,cooldown,language,rate});
      // Browser timer functions reject an arbitrary class instance as their receiver.
      this.setTimer=(fn,ms)=>setTimer(fn,ms);this.clearTimer=id=>clearTimer(id);
      this.enabled=false; this.phase='off'; this.active=null; this.queue=[];
      this.recState='stopped'; this.resumeAt=0; this.lastText=''; this.lastEnd=-Infinity;
      this.restartTimer=null; this.stopTimer=null; this.speechTimer=null; this.serial=0;
      this.rec=Recognition?new Recognition():null;
      if(this.rec){
        this.rec.continuous=true; this.rec.interimResults=false;
        this.rec.onstart=()=>{
          this.recState='listening';
          if(!this.enabled||this.active||this.now()<this.resumeAt) this._stopRecognition();
          else this._state();
        };
        this.rec.onresult=e=>{
          if(!this.enabled||this.recState!=='listening'||this.active||this.now()<this.resumeAt)return;
          // Ignore events captured before this listening session and delayed echo.
          if(e.timeStamp && this.listenStamp && e.timeStamp<this.listenStamp)return;
          for(let i=e.resultIndex??0;i<e.results.length;i++){
            if(e.results[i].isFinal===false)continue;
            const text=e.results[i][0].transcript.trim();
            if(!text)continue;
            if(this.now()-this.lastEnd<3000 && this._normal(text)===this._normal(this.lastText))continue;
            // A command can synchronously start TTS. Never process further results then.
            this.onCommand(text);
            if(this.active||!this.enabled)break;
          }
        };
        this.rec.onend=()=>{
          this.clearTimer(this.stopTimer);this.stopTimer=null;this.recState='stopped';
          if(this.active&&!this.active.started)this._begin(this.active);
          else this._resume();
        };
        this.rec.onerror=e=>{
          if(e.error==='aborted')return;
          if(['not-allowed','service-not-allowed','audio-capture'].includes(e.error)){
            this.enabled=false;this.onError('Microphone unavailable. Enable microphone access or use the controls.','recognition');
          }else if(e.error==='network'){
            this.onError('Voice recognition is unavailable online. You can use the controls.','recognition');
            this.resumeAt=this.now()+3000;
          }
          this._stopRecognition();
        };
      }
    }
    _normal(text){return text.toLowerCase().replace(/[^\p{L}\p{N} ]/gu,'').replace(/\s+/g,' ').trim();}
    _state(){
      this.phase=this.active?'speaking':!this.rec?'unavailable':!this.enabled?'off':
        this.recState==='listening'?'listening':this.recState==='faulted'?'error':'waiting';
      this.onState({microphone:!this.rec?'unavailable':!this.enabled?'off':this.active?'paused during speech':this.phase,
        tts:this.active?'speaking':'idle',enabled:this.enabled});
    }
    listen(enabled=true){
      this.enabled=!!enabled&&!!this.rec;
      if(!enabled){this.clearTimer(this.restartTimer);this.restartTimer=null;this._stopRecognition();}
      else this._resume();
      this._state();
    }
    _resume(){
      this.clearTimer(this.restartTimer);this.restartTimer=null;
      this._state();
      // Our utterance completion and cooldown own this transition. Native speaking
      // can remain true inside onend and would strand the session without a timer.
      if(!this.enabled||!this.rec||this.active||this.recState!=='stopped')return;
      const delay=Math.max(0,this.resumeAt-this.now());
      if(delay){this.restartTimer=this.setTimer(()=>{this.restartTimer=null;this._resume();},delay);return;}
      try{
        this.recState='starting'; this.rec.lang=this.language;
        this.listenStamp=typeof performance!=='undefined'?performance.now():0;
        this.rec.start();
      }catch(e){
        this.recState='stopped';
        if(e.name==='InvalidStateError'){
          // Some engines dispatch end before releasing their previous session.
          this.resumeAt=this.now()+500;
          this.restartTimer=this.setTimer(()=>this._resume(),500);
        }else{
          this.enabled=false;
          this.onError('Voice activation needs a tap in this browser. Use Enable voice or Start scanning.','recognition');
        }
      }
      this._state();
    }
    _stopRecognition(){
      this.clearTimer(this.restartTimer);this.restartTimer=null;
      if(!this.rec||['stopped','stopping','faulted'].includes(this.recState))return;
      this.clearTimer(this.stopTimer);
      this.recState='stopping';
      // Watchdog is recovery only; normal sequencing waits for recognition.onend.
      this.stopTimer=this.setTimer(()=>{
        if(this.recState!=='stopping')return;
        this.recState='faulted';
        this.onError('Waiting for the microphone to finish stopping. You can use the controls.','recognition');
        if(this.active&&!this.active.started)this._begin(this.active);
        this._state();
      },2000);
      try{this.rec.abort();}catch(e){this.recState='stopped';this.clearTimer(this.stopTimer);}
    }
    cancel(){
      this.queue=[];const old=this.active;this.active=null;this.serial++;
      this.clearTimer(this.speechTimer);this.speechTimer=null;
      if(old?.utterance){old.utterance.onend=null;old.utterance.onerror=null;}
      this.synthesis?.cancel();
      if(old){this.lastEnd=this.now();this.resumeAt=this.now()+this.cooldown;old.onDone?.(false);}
      this._resume();
    }
    say(text,options={}){this.cancel();return this.enqueue({text,priority:1,...options});}
    enqueue(item){
      if(!item.text)return false;
      item={priority:3,key:item.text,expiresAt:this.now()+5000,...item};
      if(this.active?.key===item.key||this.queue.some(x=>x.key===item.key))return false;
      if(this.active&&item.priority<this.active.priority){
        this.cancel();
      }
      // Keep only the newest normal observation. Critical/search can coexist briefly.
      if(item.priority>=3)this.queue=this.queue.filter(x=>x.priority<3);
      this.queue.push(item);this.queue.sort((a,b)=>a.priority-b.priority);this.queue=this.queue.slice(0,3);
      this._drain();return true;
    }
    prune(){
      this.queue=this.queue.filter(x=>x.expiresAt>=this.now()&&(!x.valid||x.valid()));
      if(this.active?.priority>0&&this.active.valid&&!this.active.valid())this.cancel();
    }
    _drain(){
      if(this.active)return;
      this.prune();const item=this.queue.shift();if(!item){this._resume();return;}
      item.id=++this.serial;this.active=item;
      this._stopRecognition();this._state();
      if(this.recState==='stopped'||this.recState==='faulted'||!this.rec)this._begin(item);
    }
    _begin(item){
      if(this.active!==item||item.started)return;
      if(item.valid&&!item.valid()){this._finish(item,false);return;}
      item.started=true;this.onText(item.text);this.lastText=item.text;
      if(!this.synthesis||!this.Utterance){item.onStart?.();this._finish(item,true);return;}
      try{
        const u=new this.Utterance(item.text);item.utterance=u;u.rate=this.rate;u.lang=this.language;
        const voices=this.synthesis.getVoices?.()||[];
        u.voice=voices.find(v=>v.lang===this.language&&v.localService)||voices.find(v=>v.lang===this.language)||null;
        u.onstart=()=>{if(this.active===item)item.onStart?.();};
        u.onend=()=>this._finish(item,true);
        u.onerror=()=>{if(this.active===item)this.onError('Speech output unavailable. The message is shown on screen.');this._finish(item,false);};
        this.speechTimer=this.setTimer(()=>{
          if(this.active!==item)return;
          u.onend=null;u.onerror=null;this.synthesis.cancel();
          this.onError('Speech output timed out.');this._finish(item,false);
        },Math.max(8000,Math.min(45000,item.text.length*110)));
        this.synthesis.speak(u);
      }catch(e){this.onError('Speech output unavailable.');this._finish(item,false);}
    }
    _finish(item,completed){
      if(this.active!==item)return;
      this.clearTimer(this.speechTimer);this.speechTimer=null;
      this.active=null;this.lastEnd=this.now();this.resumeAt=this.now()+this.cooldown;
      item.onDone?.(completed);this._state();this._drain();
    }
    stop(){this.listen(false);this.cancel();}
  }
  root.AssistiveSpeech={SpeechManager};
  if(typeof module!=='undefined')module.exports=root.AssistiveSpeech;
})(typeof window!=='undefined'?window:globalThis);
