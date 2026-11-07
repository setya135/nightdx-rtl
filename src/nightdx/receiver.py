"""Tune an RTL-SDR dongle, demodulate AM, and measure the signal.

librtlsdr delivers interleaved uint8 IQ; the rtl_sdr tool streams the same
format to stdout. We tune `tune_offset_hz` above the station, shift it back to
0 Hz in software, low-pass, decimate straight to the audio rate, and
envelope-detect. Along the way we average a spectrum to measure the carrier and
the sideband (modulation) level against the local noise floor - useful evidence
for the classifier.

Below 24 MHz the dongle runs in direct sampling mode (the R820T tuner is
bypassed); above it the tuner is used normally.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
from dataclasses import dataclass

import numpy as np
from scipy import signal

from .config import Settings


class ReceiverError(RuntimeError):
    pass


@dataclass
class Capture:
    audio: np.ndarray          # float32 mono, -1..1
    rate: int
    carrier_db: float          # carrier above noise floor, per FFT bin
    modulation_db: float       # sideband (200-4500 Hz) power above noise floor
    carrier_dbfs: float
    fading_db: float = 0.0     # std-dev of the carrier level over the recording (for SINPO "P")
    carriers: list | None = None   # [{offset_hz, rel_db, snr_db}], strongest first (identify.py)


def find_rx(settings: Settings) -> str:
    names = [getattr(settings.receiver, "rtl_sdr", "rtl_sdr"), "rtl_sdr"]
    for name in names:
        exe = shutil.which(name) if name else None
        if exe:
            return exe
    if os.name == "nt":
        folder = os.path.join(os.environ.get("LOCALAPPDATA", ""), "NightDX")
        for name in names:
            if not name:
                continue
            exe = os.path.join(folder, name if name.lower().endswith(".exe") else name + ".exe")
            if os.path.exists(exe):
                return exe
    from .osutil import driver_hint
    raise ReceiverError("rtl_sdr not found. " + driver_hint())


def _device_label(dev: dict) -> str:
    return dev.get("serial") or f"#{dev.get('index', 0)}"


def device_info(settings: Settings) -> str:
    """'RTL-SDR <serial>' (which dongle NightDX will use), via librtlsdr or rtl_test."""
    try:
        from .hfdev import device_list, parse_serial
        devs = device_list()
    except Exception:
        return _device_info_tool(settings)
    if not devs:
        return "not found - is the RTL-SDR plugged in?"
    want = parse_serial(settings.receiver.serial)
    if want:
        dev = next((d for d in devs if d["serial"] == want), None)
        if dev is None:
            return (f"the RTL-SDR {want} chosen in the settings is not connected - found "
                    f"{', '.join(_device_label(d) for d in devs)}")
        extra = f" (selected, {len(devs)} connected)" if len(devs) > 1 else ""
    else:
        idx = int(getattr(settings.receiver, "device_index", 0) or 0)
        dev = devs[idx] if 0 <= idx < len(devs) else devs[0]
        extra = (f" (first of {len(devs)} - choose one in Settings > Setup wizard)"
                 if len(devs) > 1 else "")
    return f"RTL-SDR {_device_label(dev)}{extra}"


def serial_args(rx) -> list[str]:
    """'-d <serial>' (or the device index) for the rtl_sdr tool."""
    from .hfdev import parse_serial
    serial = parse_serial(rx.serial)
    if serial:
        return ["-d", serial]
    idx = int(getattr(rx, "device_index", 0) or 0)
    return ["-d", str(idx)] if idx else []


def _device_info_tool(settings: Settings) -> str:
    try:
        rx = find_rx(settings)
    except Exception as e:
        return str(e)
    sibling = os.path.join(os.path.dirname(rx), "rtl_test")
    exe = sibling if os.path.exists(sibling) else shutil.which("rtl_test")
    if not exe:
        from .osutil import driver_hint
        return "rtl_test not found. " + driver_hint()
    try:
        p = subprocess.run([exe, "-t"], capture_output=True, text=True, timeout=15)
    except Exception as e:
        return f"unavailable ({e})"
    text = (p.stdout or "") + (p.stderr or "")
    for line in text.splitlines():
        if "SN:" in line:
            return f"RTL-SDR {line.split('SN:', 1)[1].strip()}"
    return "not found (is another SDR app using it?)"


def record(khz: float, seconds: float, settings: Settings,
           progress=None, stop: threading.Event | None = None,
           finish: threading.Event | None = None) -> Capture:
    """`stop` aborts and discards; `finish` ends early and keeps what we have."""
    rx = settings.receiver
    sr = rx.sample_rate
    rate = settings.recording.audio_rate
    if sr % rate:
        raise ReceiverError(f"sample rate {sr} is not a multiple of audio rate {rate}")
    dec = sr // rate
    settle = 0.5                                    # drop AGC settling at the start
    total = int((seconds + settle) * sr)
    tune_hz = khz * 1000 + rx.tune_offset_hz

    # filters (stateful, so chunks join seamlessly)
    chan = signal.cheby2(10, 60, 5500, fs=sr, output="sos")
    dc = signal.butter(2, 60, "highpass", fs=rate, output="sos")
    aud = signal.butter(6, settings.recording.audio_bandwidth_hz, fs=rate, output="sos")
    zc = signal.sosfilt_zi(chan) * 0
    zd = signal.sosfilt_zi(dc) * 0
    za = signal.sosfilt_zi(aud) * 0

    nfft = 16384
    spec = np.zeros(nfft)
    nspec = 0
    phase = 0.0
    w = 2 * np.pi * rx.tune_offset_hz / sr
    chunk = (sr // dec) * dec // 4                  # 0.25 s, multiple of `dec`
    out = []
    base = []                                       # complex baseband at the audio rate
    levels: list[float] = []
    got = 0
    early = False

    source = _Source(settings, tune_hz, total)
    carry = np.zeros(0, dtype=np.complex64)
    try:
        for block in source.blocks():
            if stop is not None and stop.is_set():
                raise ReceiverError("recording cancelled")
            if finish is not None and finish.is_set() and got > (settle + 2) * sr:
                early = True
                break
            carry = np.concatenate([carry, block])
            if len(carry) < chunk and got + len(carry) < total:
                continue
            n = len(carry) - len(carry) % dec
            x, carry = carry[:n], carry[n:]
            mix = np.exp(1j * (phase + w * np.arange(n)))
            phase = (phase + w * n) % (2 * np.pi)
            y = x * mix.astype(np.complex64)
            got += n
            if got > settle * sr and len(y) >= nfft:
                spec += np.abs(np.fft.fftshift(np.fft.fft(y[:nfft] * np.hanning(nfft)))) ** 2
                nspec += 1
            y, zc = signal.sosfilt(chan, y, zi=zc)
            yd = y[::dec]
            env = np.abs(yd)
            if got > settle * sr:
                levels.append(20 * np.log10(env.mean() + 1e-12))
                base.append(yd.astype(np.complex64))
            a, zd = signal.sosfilt(dc, env, zi=zd)
            a, za = signal.sosfilt(aud, a, zi=za)
            out.append(a.astype(np.float32))
            if progress:
                progress(min(1.0, got / total))
    finally:
        err = source.close()
    if not early and got < total * 0.9:
        raise ReceiverError(f"the RTL-SDR delivered {got / sr:.1f}s of {total / sr:.1f}s "
                            f"(is another SDR program using it?)\n{err[-400:]}")

    audio = np.concatenate(out)[int(settle * rate):]
    peak = np.percentile(np.abs(audio), 99.9) or 1.0
    audio = np.clip(audio / peak * 0.8, -1, 1).astype(np.float32)

    # signal measurements from the averaged spectrum
    spec /= max(nspec, 1)
    freqs = np.fft.fftshift(np.fft.fftfreq(nfft, 1 / sr))
    af = np.abs(freqs)
    noise = np.median(spec[(af > 6000) & (af < 60000)]) or 1e-20
    carrier = spec[af < 40].max()
    sideband = spec[(af > 200) & (af < 4500)].mean()
    try:
        from .identify import find_carriers
        cs = find_carriers(np.concatenate(base), rate) if base else []
        fix = getattr(rx, "freq_correction_ppm", 0.0) * khz / 1000   # ppm x MHz = Hz
        for c in cs:
            c["offset_hz"] = round(c["offset_hz"] + fix, 2)
    except Exception:
        cs = []
    return Capture(
        audio=audio, rate=rate, carriers=cs,
        carrier_db=float(10 * np.log10(carrier / noise)),
        modulation_db=float(10 * np.log10(sideband / noise)),
        carrier_dbfs=float(10 * np.log10(carrier / np.hanning(nfft).sum() ** 2 + 1e-20)),
        fading_db=float(np.std(levels)) if len(levels) > 4 else 0.0,
    )


class _Source:
    """IQ samples from the RTL-SDR: through librtlsdr directly (every OS), or by
    running the rtl_sdr tool (setting receiver.backend, default 'auto')."""

    def __init__(self, settings: Settings, tune_hz: float, total: int):
        self.settings, self.tune_hz, self.total = settings, tune_hz, total
        self.proc = None
        self.rtl = None
        backend = getattr(settings.receiver, "backend", "auto")
        if backend in ("auto", "library"):
            from .hfdev import RTLDevice, SerialNotFound
            try:
                self.rtl = RTLDevice(settings.receiver.sample_rate,
                                     settings.receiver).__enter__()
            except SerialNotFound as e:              # never fall back to "any radio"
                raise ReceiverError(str(e))
            except Exception as e:
                if backend == "library":
                    raise ReceiverError(str(e))
                self.rtl = None
        if self.rtl is None:
            self._start_tool()

    def _start_tool(self) -> None:
        from .hfdev import direct_sampling_mode
        rx = self.settings.receiver
        sr = rx.sample_rate
        # rtl_sdr counts samples in 32 bits; 0 means "read forever" (live listening)
        n = str(self.total) if self.total < 2 ** 31 else "0"
        cmd = [find_rx(self.settings), "-f", f"{self.tune_hz / 1e6:.6f}", "-s", str(sr),
               "-n", n, "-g", "0" if rx.agc else f"{rx.gain_db:g}"]
        ppm = int(round(getattr(rx, "freq_correction_ppm", 0.0) or 0))
        if ppm:
            cmd += ["-p", str(ppm)]
        if direct_sampling_mode(self.tune_hz, rx):
            cmd += ["-D"]
        cmd += serial_args(rx)
        cmd += ["-"]                                # rtl_sdr: '-' = stdout
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def blocks(self):
        if self.rtl is not None:
            yield from self.rtl.stream(self.tune_hz, self.total)
            return
        from .hfdev import iq_from_uint8
        size = self.settings.receiver.sample_rate // 4 * 2      # 0.25 s, uint8 IQ
        while True:
            buf = self.proc.stdout.read(size)
            if not buf:
                return
            if len(buf) % 2:
                buf = buf[:-1]
            yield iq_from_uint8(buf)

    def close(self) -> str:
        if self.rtl is not None:
            try:
                self.rtl.__exit__(None, None, None)
            except Exception:
                pass
            return ""
        if self.proc.poll() is None:
            self.proc.terminate()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        return self.proc.stderr.read().decode(errors="replace")


# ----------------------------------------------------------------------------- live listening

LIVE_NFFT = 8192


class LiveDemod:
    """The recording chain (offset mix, channel filter, envelope, audio filter) for live
    audio, chunk by chunk, plus the signal level of each chunk and a slow AGC so weak and
    strong stations play at a similar volume."""

    def __init__(self, settings: Settings):
        rx = settings.receiver
        self.sr, self.rate = rx.sample_rate, settings.recording.audio_rate
        if self.sr % self.rate:
            raise ReceiverError(f"sample rate {self.sr} is not a multiple of audio rate {self.rate}")
        self.dec = self.sr // self.rate
        self.w = 2 * np.pi * rx.tune_offset_hz / self.sr
        self.phase = 0.0
        self.chan = signal.cheby2(10, 60, 5500, fs=self.sr, output="sos")
        self.dc = signal.butter(2, 60, "highpass", fs=self.rate, output="sos")
        self.aud = signal.butter(6, settings.recording.audio_bandwidth_hz, fs=self.rate, output="sos")
        self.zc, self.zd, self.za = (signal.sosfilt_zi(f) * 0 for f in (self.chan, self.dc, self.aud))
        self.carry = np.zeros(0, dtype=np.complex64)
        self.gain = None
        self.nfft = LIVE_NFFT
        self.win = np.hanning(self.nfft)
        self.freqs = np.abs(np.fft.fftshift(np.fft.fftfreq(self.nfft, 1 / self.sr)))

    def process(self, block: np.ndarray) -> tuple[np.ndarray, float | None, float | None]:
        """IQ block -> (audio float32, carrier dB above noise, modulation dB above noise).
        The levels are None when the block is too short to measure."""
        x = np.concatenate([self.carry, block])
        n = len(x) - len(x) % self.dec
        x, self.carry = x[:n], x[n:]
        y = x * np.exp(1j * (self.phase + self.w * np.arange(n))).astype(np.complex64)
        self.phase = (self.phase + self.w * n) % (2 * np.pi)
        carrier = mod = None
        if n >= self.nfft:
            spec = np.abs(np.fft.fftshift(np.fft.fft(y[:self.nfft] * self.win))) ** 2
            noise = np.median(spec[(self.freqs > 6000) & (self.freqs < 60000)]) or 1e-20
            carrier = float(10 * np.log10(spec[self.freqs < 40].max() / noise))
            mod = float(10 * np.log10(spec[(self.freqs > 200) & (self.freqs < 4500)].mean() / noise))
        y, self.zc = signal.sosfilt(self.chan, y, zi=self.zc)
        env = np.abs(y[::self.dec])
        a, self.zd = signal.sosfilt(self.dc, env, zi=self.zd)
        a, self.za = signal.sosfilt(self.aud, a, zi=self.za)
        if len(a):
            peak = float(np.percentile(np.abs(a), 99)) or 1e-9
            g = 0.5 / peak
            self.gain = g if self.gain is None else min(g, self.gain * 0.8 + g * 0.2)
            a = np.clip(a * self.gain, -1, 1)
        return a.astype(np.float32), carrier, mod


def listen(settings: Settings, khz: float, stop: threading.Event, retune: threading.Event,
           on_level=None) -> None:
    """Play `khz` live through the computer's speakers until `stop` is set. When `retune`
    is set, return so the caller can start again on a new frequency."""
    import sounddevice as sd
    rx = settings.receiver
    demod = LiveDemod(settings)
    source = _Source(settings, khz * 1000 + rx.tune_offset_hz, rx.sample_rate * 3600 * 12)
    try:
        with sd.OutputStream(samplerate=demod.rate, channels=1, dtype="float32") as out:
            pending, have = [], 0
            for block in source.blocks():
                if stop.is_set() or retune.is_set():
                    break
                pending.append(block)               # USB blocks are small: work in 0.1 s chunks
                have += len(block)
                if have < rx.sample_rate // 10:
                    continue
                chunk, pending, have = np.concatenate(pending), [], 0
                audio, carrier, mod = demod.process(chunk)
                if len(audio):
                    out.write(audio.reshape(-1, 1))
                if on_level and carrier is not None:
                    on_level(carrier, mod)
    finally:
        source.close()
