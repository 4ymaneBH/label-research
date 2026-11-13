class WaitArt {
  constructor(canvas, toggle) {
    this.canvas = canvas;
    this.context = canvas.getContext("2d");
    this.toggle = toggle;
    this.motion = matchMedia("(prefers-reduced-motion: reduce)");
    this.active = false;
    this.paused = false;
    this.frame = null;
    this.time = 0;
    this.tick = this.tick.bind(this);

    toggle.addEventListener("click", () => {
      this.paused = !this.paused;
      this.updateMotion();
    });
    this.motion.addEventListener("change", () => this.updateMotion());
  }

  start() {
    if (!this.context) return;
    this.active = true;
    this.time = 0;
    this.seed = Math.random() * 2 * Math.PI;
    this.draw();
    this.updateMotion();
  }

  stop() {
    this.active = false;
    this.updateMotion();
  }

  updateMotion() {
    if (this.frame !== null) {
      cancelAnimationFrame(this.frame);
      this.frame = null;
    }
    this.toggle.hidden = !this.active || this.motion.matches;
    this.toggle.textContent = this.paused ? "Resume motion" : "Pause motion";
    if (this.active && !this.paused && !this.motion.matches) {
      this.previousFrame = null;
      this.frame = requestAnimationFrame(this.tick);
    }
  }

  tick(now) {
    if (this.previousFrame !== null) {
      // Cap the step after a suspended tab so returning never jumps the drawing.
      this.time += Math.min(now - this.previousFrame, 50) / 1000;
    }
    this.previousFrame = now;
    this.draw();
    this.frame = requestAnimationFrame(this.tick);
  }

  draw() {
    const ctx = this.context;
    const { width, height } = this.canvas;
    const count = 15;
    const steps = 112;
    ctx.clearRect(0, 0, width, height);

    // Each request starts a different ink study. Smooth waves vary with time,
    // while the tapered ends keep the lines within the page margins.
    for (let line = 0; line < count; line++) {
      const offset = line - (count - 1) / 2;
      const accent = line === 7;
      ctx.strokeStyle = accent ? "#0007cb" : "#8f8579";
      ctx.globalAlpha = accent ? 0.86 : 0.44;
      ctx.lineWidth = accent ? 1.8 : 1;
      ctx.beginPath();
      for (let step = 0; step <= steps; step++) {
        const u = step / steps;
        const x = 32 + (width - 64) * u;
        const envelope = Math.sin(Math.PI * u);
        const bend = Math.sin(u * 7.4 + this.seed + this.time * 0.62 + offset * 0.12);
        const fold = Math.sin(u * 12.8 - this.time * 0.42 + this.seed);
        const spread = 3 + 5 * (0.5 + 0.5 * Math.cos(u * 7 + this.time * 0.5));
        const y = height / 2 + envelope * (bend * 37 + fold * 13 + offset * spread);
        if (step === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
  }
}
