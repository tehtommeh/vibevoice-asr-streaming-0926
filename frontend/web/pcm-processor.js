/**
 * Forwards mono float32 frames from the mic to the main thread.
 * AudioWorklet runs on the audio thread, so this stays out of the way of
 * rendering the transcript.
 */
class PCMProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (channel && channel.length) {
      // postMessage cannot keep a reference to the live render quantum, so copy.
      const copy = new Float32Array(channel);
      this.port.postMessage(copy.buffer, [copy.buffer]);
    }
    return true;
  }
}
registerProcessor('pcm-processor', PCMProcessor);
