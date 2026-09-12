/* Current-observation memory and one-request-per-lane scheduling. */
(function(root){
  const norm=s=>s.toLowerCase().normalize('NFKC').replace(/[^\p{L}\p{N}]+/gu,' ').trim();
  function iou(a,b){
    const n=Math.max(0,Math.min(a[2],b[2])-Math.max(a[0],b[0]))*Math.max(0,Math.min(a[3],b[3])-Math.max(a[1],b[1]));
    const u=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-n;
    return u>0?n/u:0;
  }
  const center=b=>[(b[0]+b[2])/2,(b[1]+b[3])/2];
  class DetectionMemory {
    constructor({ttl=6500,misses=3,persistence=2,move=.18}={}){
      Object.assign(this,{ttl,maxMisses:misses,persistence,move});this.tracks=new Map();this.nextId=0;
    }
    clear(){this.tracks.clear();}
    hide(lane){for(const t of this.tracks.values())if(t.lane===lane)t.active=false;}
    expire(now){for(const [id,t] of this.tracks)if(now-t.lastSeen>this.ttl)this.tracks.delete(id);}
    update(lane,items,frame,now){
      this.expire(now);const assigned=new Set();const visible=[];
      for(const item of items.slice(0,100)){
        if(!item.box||item.conf<.4)continue;
        const box=item.box.map((v,i)=>v/(i%2?frame.h:frame.w));
        const label=norm(item.raw||item.label),c=center(box);
        const candidates=[...this.tracks.values()].filter(t=>t.lane===lane&&t.label===label&&!assigned.has(t.id))
          .map(t=>({t,overlap:iou(t.box,box),distance:Math.hypot(...center(t.box).map((v,i)=>v-c[i]))}))
          .filter(x=>x.overlap>.12||x.distance<.22).sort((a,b)=>b.overlap-a.overlap||a.distance-b.distance);
        let track=candidates[0]?.t;
        if(!track){track={id:++this.nextId,lane,label,count:0,misses:0,lastSpoken:null,spoken:null,version:0};this.tracks.set(track.id,track);}
        track.count=track.misses?1:track.count+1;track.misses=0;track.box=box;track.lastSeen=now;track.active=true;
        track.item={...item,track_id:track.id};assigned.add(track.id);
        const old=track.spoken;
        const changed=old&&(Math.hypot(...center(old.box).map((v,i)=>v-c[i]))>=this.move||
          (item.steps&&old.steps&&Math.abs(item.steps-old.steps)>=Math.max(2,old.steps*.35))||
          (!!item.critical&&!old.critical));
        if(changed){track.spoken=null;track.searchSpoken=null;track.version++;}
        track.stable=!!item.critical||track.count>=this.persistence;
        visible.push(track);
      }
      for(const [id,t] of this.tracks)if(t.lane===lane&&!assigned.has(id)){
        t.misses++;t.active=false;if(t.misses>=this.maxMisses)this.tracks.delete(id);
      }
      return visible; // Only current detections are rendered, never missed tracks.
    }
    markSpoken(track,now){
      if(!this.tracks.has(track.id))return;
      track.lastSpoken=now;track.spoken={box:[...track.box],steps:track.item.steps,critical:track.item.critical};
    }
    valid(track,version,now){
      return this.tracks.get(track.id)===track&&track.active&&track.version===version&&track.misses===0&&now-track.lastSeen<=this.ttl;
    }
    fresh(now){this.expire(now);return [...this.tracks.values()].filter(t=>t.active&&!t.misses&&t.stable);}
  }
  class LatestLane {
    constructor(job,{interval=350,onError=()=>{},setTimer=setTimeout,clearTimer=clearTimeout}={}){
      Object.assign(this,{job,interval,onError,setTimer,clearTimer});this.generation=0;this.running=false;this.busy=false;this.timer=null;
      this.setTimer=(fn,ms)=>setTimer(fn,ms);this.clearTimer=id=>clearTimer(id);
    }
    start(){if(this.running)return;this.running=true;this.generation++;this._tick();}
    stop(){this.running=false;this.generation++;this.clearTimer(this.timer);this.timer=null;this.controller?.abort();}
    kick(){this.clearTimer(this.timer);this.timer=null;if(this.running&&!this.busy)this._tick();}
    async _tick(){
      if(!this.running||this.busy)return;
      this.busy=true;const generation=this.generation;const controller=new AbortController();this.controller=controller;
      try{await this.job({signal:controller.signal,current:()=>this.running&&generation===this.generation});}
      catch(e){if(this.running&&generation===this.generation)this.onError(e);}
      finally{
        this.busy=false;if(this.controller===controller)this.controller=null;
        // A restart waits for an old request to settle; it never starts a second request.
        if(this.running)this.timer=this.setTimer(()=>this._tick(),generation===this.generation?this.interval:0);
      }
    }
  }
  function describe(item){
    const distance=item.steps?(item.steps>30?'far away':`approximately ${item.steps} ${item.steps===1?'step':'steps'}`):'';
    return [item.label,distance,item.direction||'ahead'].filter(Boolean).join(', ')+'.';
  }
  function mapsUrl(destination,coords={}){
    const p=new URLSearchParams({api:'1',destination,travelmode:'walking',dir_action:'navigate'});
    if(Number.isFinite(coords.latitude)&&Number.isFinite(coords.longitude))p.set('origin',`${coords.latitude},${coords.longitude}`);
    return 'https://www.google.com/maps/dir/?'+p;
  }
  function locate(geolocation,{timeout=8000,setTimer=setTimeout,clearTimer=clearTimeout}={}){
    return new Promise(resolve=>{
      let settled=false;
      const finish=value=>{if(settled)return;settled=true;clearTimer(timer);resolve(value);};
      const timer=setTimer(()=>finish({error:'Location timed out.'}),timeout);
      if(!geolocation){finish({error:'Location unavailable.'});return;}
      try{geolocation.getCurrentPosition(p=>finish({coords:p.coords}),e=>finish({error:
        e.code===1?'Location permission denied.':e.code===3?'Location timed out.':'Location unavailable.'}),
        {enableHighAccuracy:true,timeout,maximumAge:15000});}
      catch(e){finish({error:'Location unavailable.'});}
    });
  }
  root.AssistiveRealtime={DetectionMemory,LatestLane,describe,mapsUrl,locate};
  if(typeof module!=='undefined')module.exports=root.AssistiveRealtime;
})(typeof window!=='undefined'?window:globalThis);
