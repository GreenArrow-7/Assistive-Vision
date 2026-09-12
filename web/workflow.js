/* Pure state and speech scheduling logic, shared with Node regression tests. */
(function (root) {
  // Real speech recognisers return "4 4", "number four", "for", "to", "tree"...
  // for a user who said a single menu digit. Normalise before matching.
  const HOMOPHONES = {one: '1', won: '1', two: '2', to: '2', too: '2', three: '3',
                      tree: '3', free: '3', four: '4', for: '4', fore: '4', five: '5',
                      fife: '5'};
  const FILLER = new Set(['number', 'option', 'mode', 'select', 'choose', 'press', 'say',
                          'please', 'ok', 'okay', 'hey', 'the', 'a', 'go', 'with']);
  function menuDigit(q) {
    // A short utterance made only of one digit (any spelling) and filler words.
    const toks = q.split(' ').filter(t => t && !FILLER.has(t));
    const digits = toks.map(t => HOMOPHONES[t] || t).filter(t => /^[1-5]$/.test(t));
    if (!digits.length || digits.length !== toks.length) return null;   // no other words
    return digits.every(d => d === digits[0]) ? digits[0] : null;         // "4 4" -> 4, "4 2" -> ambiguous
  }
  class Workflow {
    constructor() { this.mode = 'WELCOME'; this.query = ''; this.generation = 0; }
    transition(mode, query = '') {
      if (!['WELCOME', 'MAIN_MENU', 'ENVIRONMENT', 'SEARCH', 'NAVIGATION', 'STOPPED'].includes(mode))
        throw new Error('Unknown assistance mode');
      this.mode = mode; this.query = query; this.generation++;
      return mode;
    }
    command(raw) {
      const q = raw.toLowerCase().replace(/[^a-z0-9' ]+/g, ' ').replace(/\s+/g, ' ').trim();
      if (!q) return null;
      // Numbered menu works AT ANY POINT: 1 environment, 2 search, 3 navigation, 4 home, 5 stop
      const d = menuDigit(q);
      if (d) return [['ENVIRONMENT', 'SEARCH', 'NAVIGATION', 'MAIN_MENU', 'STOPPED'][+d - 1]];
      if (/^(stop|pause|stop scanning|stop scan|stop the scan|stop system)$/.test(q)) return ['STOPPED'];
      if (/^(back|go back|main menu|menu|home|home page|go home|main page)$/.test(q)) return ['MAIN_MENU'];
      if (/^(repeat|say that again|repeat that)$/.test(q)) return ['REPEAT'];
      if (/^(start|begin|let's start|start system)$/.test(q)) return ['MAIN_MENU'];
      if (/^(environment|environment summary|environmental summary|summary|start scanning|start scan|begin scanning|scan|scan environment)$/.test(q)) return ['ENVIRONMENT'];
      if (/^(search|keyword search|keyword|keyword based search)$/.test(q)) return ['SEARCH'];
      if (/^(navigation|navigate|navigation assistance)$/.test(q)) return ['NAVIGATION'];
      // Route-info intent must outrank the NAVIGATION passthrough below, or
      // "distance and time" becomes a destination named "distance and time".
      if (/^(distance and time|travel time|how far is it|how long will it take)$/.test(q)) return ['DISTANCE_TIME'];
      if (/^(find|locate|search for|look for)\s+/.test(q)) return ['SEARCH', q.replace(/^(find|locate|search for|look for)\s+/, '')];
      if (/^(go to|navigate to|take me to|directions to)\s+/.test(q)) return ['NAVIGATION', q.replace(/^(go to|navigate to|take me to|directions to)\s+/, '')];
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
  // Self-echo test for barge-in: a transcript whose words all occur in what we
  // are currently speaking is our own voice coming back through the mic.
  function isEcho(transcript, spoken) {
    if (!spoken) return false;
    const said = new Set(spoken.toLowerCase().replace(/[^a-z0-9 ]+/g, ' ').split(/\s+/));
    const words = transcript.toLowerCase().replace(/[^a-z0-9 ]+/g, ' ').split(/\s+/).filter(Boolean);
    return words.length > 0 && words.every(w => said.has(w));
  }
  root.AssistiveWorkflow = {Workflow, AnnouncementQueue, menuDigit, isEcho};
  if (typeof module !== 'undefined') module.exports = root.AssistiveWorkflow;
})(typeof window !== 'undefined' ? window : globalThis);
