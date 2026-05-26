"""Audio recording with VAD (Voice Activity Detection)."""
from __future__ import annotations
import collections
import queue
import threading
import numpy as np


class AudioRecorder:
    def __init__(self, sample_rate: int = 16000, vad_aggressiveness: int = 2,
                 silence_duration: float = 1.2):
        self.sample_rate = sample_rate
        self.vad_aggressiveness = vad_aggressiveness
        self.silence_duration = silence_duration
        self._vad = None

    def _get_vad(self):
        if self._vad is None:
            import webrtcvad
            self._vad = webrtcvad.Vad(self.vad_aggressiveness)
        return self._vad

    def record_until_silence(self, on_speaking: callable = None) -> np.ndarray:
        """
        Record audio until silence detected.
        Returns float32 numpy array at self.sample_rate.
        """
        import sounddevice as sd

        frame_ms = 30  # webrtcvad supports 10/20/30ms
        frame_samples = int(self.sample_rate * frame_ms / 1000)
        silence_frames = int(self.silence_duration * 1000 / frame_ms)
        min_speech_frames = 3  # 至少 3 帧才算有效语音

        vad = self._get_vad()
        audio_queue: queue.Queue = queue.Queue()
        stop_event = threading.Event()

        def callback(indata, frames, time_info, status):
            audio_queue.put(indata.copy())

        recorded_frames = []
        speech_frames = 0
        silent_frames = 0
        started = False

        with sd.InputStream(samplerate=self.sample_rate, channels=1,
                            dtype="int16", blocksize=frame_samples,
                            callback=callback):
            while not stop_event.is_set():
                try:
                    chunk = audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                pcm = chunk.flatten().tobytes()
                try:
                    is_speech = vad.is_speech(pcm, self.sample_rate)
                except Exception:
                    is_speech = False

                if is_speech:
                    if not started:
                        started = True
                        if on_speaking:
                            threading.Thread(target=on_speaking, daemon=True).start()
                    speech_frames += 1
                    silent_frames = 0
                    recorded_frames.append(chunk.flatten())
                elif started:
                    silent_frames += 1
                    recorded_frames.append(chunk.flatten())
                    if silent_frames >= silence_frames and speech_frames >= min_speech_frames:
                        break

        if not recorded_frames:
            return np.zeros(frame_samples, dtype=np.float32)

        audio_int16 = np.concatenate(recorded_frames)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        return audio_float32
