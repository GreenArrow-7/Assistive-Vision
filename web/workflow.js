/* Pure state and speech scheduling logic, shared with Node regression tests. */
(function (root) {
  class Workflow {
    constructor() { this.mode = 'WELCOME'; this.query = ''; this.generation = 0; }
    transition(mode, query = '') {
      if (!['WELCOME', 'MAIN_MENU', 'ENVIRONMENT', 'SEARCH', 'NAVIGATION', 'STOPPED'].includes(mode))
        throw new Error('Unknown assistance mode');
      this.mode = mode; this.query = query; this.generation++;
      return mode;
    }
    command(raw) {
      const q = raw.toLowerCase().trim().replace(/[.!?]+$/, '');
      // Numbered menu: 1 environment, 2 search, 3 navigation, 4 home, 5 stop
      if (/^(stop|pause|stop scanning|five|5)$/.test(q)) return ['STOPPED'];
      if (/^(back|go back|main menu|menu|home|home page|four|4)$/.test(q)) return ['MAIN_MENU'];
      if (/^(repeat|say that again)$/.test(q)) return ['REPEAT'];
      if (/^(start|begin|let's start)$/.test(q)) return ['MAIN_MENU'];
      if (/^(one|1|environment|environment summary|summary|start scanning|start scan|begin scanning|scan)$/.test(q)) return ['ENVIRONMENT'];
      if (/^(two|2|search|keyword search)$/.test(q)) return ['SEARCH'];
      if (/^(three|3|navigation|navigate)$/.test(q)) return ['NAVIGATION'];
      // Route-info intent must outrank the NAVIGATION passthrough below, or
      // "distance and time" becomes a destination named "distance and time".
      if (/^(distance and time|travel time|how far is it|how long will it take)$/.test(q)) return ['DISTANCE_TIME'];
      if (/^(find|locate|search for)\s+/.test(q)) return ['SEARCH', q.replace(/^(find|locate|search for)\s+/, '')];
      if (/^(go to|navigate to|take me to)\s+/.test(q)) return ['NAVIGATION', q.replace(/^(go to|navigate to|take me to)\s+/, '')];
      if (this.mode === 'SEARCH' || this.mode === 'NAVIGATION') return [this.mode, q];
      return null;
    }
  }
  class AnnouncementQueue {
    constructor(cooldown = 9000) { this.cooldown = cooldown; this.last = new Map(); this.pending = []; }
    push(text, priority = 3, now = Date.now()) {
      if (!text || now - (this.last.get(text) ?? -Infinity) < this.cooldown) return false;
      if (this.pending.some(x => x.text === text)) return false;
      if (priority === 0) this.pending = [];
      this.pending.push({text, priority, time: now});
      this.pending.sort((a,b) => a.priority-b.priority || a.time-b.time);
      this.pending = this.pending.slice(0, 3);
      return true;
    }
    take(now = Date.now()) {
      this.pending = this.pending.filter(x => now-x.time < 8000);
      const item = this.pending.shift();
      if (item) this.last.set(item.text, now);
      for (const [key, time] of this.last) if (now-time > this.cooldown) this.last.delete(key);
      return item;
    }
    clear() { this.pending = []; }
  }
  root.AssistiveWorkflow = {Workflow, AnnouncementQueue};
  if (typeof module !== 'undefined') module.exports = root.AssistiveWorkflow;
})(typeof window !== 'undefined' ? window : globalThis);
