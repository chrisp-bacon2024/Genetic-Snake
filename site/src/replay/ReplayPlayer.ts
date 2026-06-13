import type { LiteReplayFrame, ReplayDocument } from "./types";

export type FrameListener = (frame: LiteReplayFrame, index: number) => void;
export type CompleteListener = () => void;

export class ReplayPlayer {
  private document: ReplayDocument | null = null;
  private frameIndex = 0;
  private accumulator = 0;
  private lastTimestamp = 0;
  private playing = true;
  private speed = 1;
  private loop = false;
  private finished = false;
  private rafId = 0;
  private listeners = new Set<FrameListener>();
  private completeListeners = new Set<CompleteListener>();

  async load(url: string): Promise<ReplayDocument> {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to load replay: ${url}`);
    }
    this.document = (await response.json()) as ReplayDocument;
    this.frameIndex = 0;
    this.accumulator = 0;
    this.finished = false;
    this.emitFrame();
    return this.document;
  }

  get replay(): ReplayDocument | null {
    return this.document;
  }

  get currentFrame(): LiteReplayFrame | null {
    if (!this.document || this.document.frames.length === 0) return null;
    return this.document.frames[this.frameIndex] ?? null;
  }

  get isFinished(): boolean {
    return this.finished;
  }

  onFrame(listener: FrameListener): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  onComplete(listener: CompleteListener): () => void {
    this.completeListeners.add(listener);
    return () => this.completeListeners.delete(listener);
  }

  setPlaying(value: boolean): void {
    this.playing = value;
    if (value) {
      this.lastTimestamp = 0;
      this.startLoop();
    }
  }

  setSpeed(value: number): void {
    this.speed = value;
  }

  setLoop(value: boolean): void {
    this.loop = value;
  }

  restart(): void {
    this.frameIndex = 0;
    this.accumulator = 0;
    this.finished = false;
    this.lastTimestamp = 0;
    this.emitFrame();
  }

  startLoop(): void {
    cancelAnimationFrame(this.rafId);
    const tick = (timestamp: number) => {
      if (this.playing && this.document && this.document.frames.length > 0 && !this.finished) {
        if (this.lastTimestamp === 0) {
          this.lastTimestamp = timestamp;
        }
        const delta = timestamp - this.lastTimestamp;
        this.lastTimestamp = timestamp;
        this.accumulator += delta;

        const frameDuration = 1000 / (this.document.ticks_per_second * this.speed);
        while (this.accumulator >= frameDuration && !this.finished) {
          this.accumulator -= frameDuration;
          this.advanceFrame();
        }
      }
      this.rafId = requestAnimationFrame(tick);
    };
    this.rafId = requestAnimationFrame(tick);
  }

  stopLoop(): void {
    cancelAnimationFrame(this.rafId);
  }

  private advanceFrame(): void {
    if (!this.document || this.finished) {
      return;
    }

    const lastIndex = this.document.frames.length - 1;
    if (this.frameIndex >= lastIndex) {
      this.finished = true;
      this.emitComplete();
      if (this.loop) {
        this.restart();
      }
      return;
    }

    this.frameIndex += 1;
    this.emitFrame();

    if (this.frameIndex >= lastIndex) {
      this.finished = true;
      this.emitComplete();
      if (this.loop) {
        this.restart();
      }
    }
  }

  private emitFrame(): void {
    const frame = this.currentFrame;
    if (!frame) return;
    for (const listener of this.listeners) {
      listener(frame, this.frameIndex);
    }
  }

  private emitComplete(): void {
    for (const listener of this.completeListeners) {
      listener();
    }
  }
}
