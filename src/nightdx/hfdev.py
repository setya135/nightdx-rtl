"""Direct access to an RTL-SDR dongle through librtlsdr (ctypes).

rtl_sdr re-opens the device for every run (~1 s). For band scans we open the
dongle once, keep it streaming, and just retune between captures.

RTL-SDR dongles tune 24 MHz - 1.7 GHz with their R820T tuner. Shortwave and
medium wave are below that, so they use *direct sampling* (the tuner is
bypassed); this needs a dongle that exposes the Q branch, such as the RTL-SDR
Blog V3, or a hardware direct-sampling modification.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import re
import threading

import numpy as np

from .osutil import rtlsdr_library_candidates

DIRECT_SAMPLING_MAX_HZ = 24_000_000      # above this the R820T tuner is used
DEFAULT_BUF_NUM = 15
DEFAULT_BUF_LEN = 16 * 16384

# rtlsdr_read_async callback: (uint8_t *buf, uint32_t len, void *ctx) -> void
_CB = ctypes.CFUNCTYPE(None, ctypes.POINTER(ctypes.c_uint8), ctypes.c_uint32,
                       ctypes.c_void_p)

def _lib():
    for p in rtlsdr_library_candidates():
        if p and (os.path.exists(p) or os.sep not in p):
            if os.name == "nt" and os.path.isabs(p):
                # rtlsdr.dll needs libusb-1.0.dll from the same folder
                try:
                    os.add_dll_directory(os.path.dirname(p))
                except (OSError, AttributeError):
                    pass
            try:
                lib = ctypes.CDLL(p)
            except OSError:          # e.g. an Intel build on Apple Silicon
                continue
            try:
                _bind(lib)
            except AttributeError:
                continue
            return lib
    from .osutil import driver_hint
    raise RuntimeError("no usable librtlsdr. " + driver_hint())


def _bind(lib) -> None:
    lib.rtlsdr_open.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint32]
    lib.rtlsdr_open.restype = ctypes.c_int
    lib.rtlsdr_close.argtypes = [ctypes.c_void_p]
    lib.rtlsdr_close.restype = ctypes.c_int
    lib.rtlsdr_get_device_count.argtypes = []
    lib.rtlsdr_get_device_count.restype = ctypes.c_uint32
    lib.rtlsdr_get_device_name.argtypes = [ctypes.c_uint32]
    lib.rtlsdr_get_device_name.restype = ctypes.c_char_p
    lib.rtlsdr_get_device_usb_strings.argtypes = [
        ctypes.c_uint32, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
    lib.rtlsdr_get_device_usb_strings.restype = ctypes.c_int
    lib.rtlsdr_get_index_by_serial.argtypes = [ctypes.c_char_p]
    lib.rtlsdr_get_index_by_serial.restype = ctypes.c_int
    lib.rtlsdr_get_tuner_type.argtypes = [ctypes.c_void_p]
    lib.rtlsdr_get_tuner_type.restype = ctypes.c_int
    lib.rtlsdr_set_sample_rate.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    lib.rtlsdr_set_sample_rate.restype = ctypes.c_int
    lib.rtlsdr_get_sample_rate.argtypes = [ctypes.c_void_p]
    lib.rtlsdr_get_sample_rate.restype = ctypes.c_uint32
    lib.rtlsdr_set_center_freq.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    lib.rtlsdr_set_center_freq.restype = ctypes.c_int
    lib.rtlsdr_set_tuner_gain_mode.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.rtlsdr_set_tuner_gain_mode.restype = ctypes.c_int
    lib.rtlsdr_set_tuner_gain.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.rtlsdr_set_tuner_gain.restype = ctypes.c_int
    lib.rtlsdr_set_agc_mode.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.rtlsdr_set_agc_mode.restype = ctypes.c_int
    lib.rtlsdr_set_direct_sampling.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.rtlsdr_set_direct_sampling.restype = ctypes.c_int
    lib.rtlsdr_set_freq_correction.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.rtlsdr_set_freq_correction.restype = ctypes.c_int
    lib.rtlsdr_reset_buffer.argtypes = [ctypes.c_void_p]
    lib.rtlsdr_reset_buffer.restype = ctypes.c_int
    lib.rtlsdr_read_async.argtypes = [
        ctypes.c_void_p, _CB, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32]
    lib.rtlsdr_read_async.restype = ctypes.c_int
    lib.rtlsdr_cancel_async.argtypes = [ctypes.c_void_p]
    lib.rtlsdr_cancel_async.restype = ctypes.c_int


class SerialNotFound(RuntimeError):
    """The RTL-SDR chosen in the settings is not connected."""


def parse_serial(value) -> str:
    """The settings' serial as people write it: '00000001', 'SN: 00000001',
    'RTL-SDR 00000001'. Empty means the first radio."""
    if value is None:
        return ""
    s = str(value).strip()
    s = re.sub(r"^(?i:sn)\s*:\s*", "", s)
    s = re.sub(r"^(?i:rtl[- ]?sdr)\s+", "", s)
    return s.strip()


def format_serial(value) -> str:
    return parse_serial(value)


def device_list() -> list[dict]:
    """Connected RTL-SDR dongles: [{'index', 'name', 'manufacturer', 'product', 'serial'}]."""
    lib = _lib()
    n = int(lib.rtlsdr_get_device_count())
    out = []
    for i in range(n):
        raw = lib.rtlsdr_get_device_name(i)
        name = raw.decode(errors="replace") if raw else ""
        man = ctypes.create_string_buffer(256)
        prod = ctypes.create_string_buffer(256)
        ser = ctypes.create_string_buffer(256)
        serial = ""
        if lib.rtlsdr_get_device_usb_strings(i, man, prod, ser) == 0:
            serial = ser.value.decode(errors="replace").strip()
        out.append({"index": i, "name": name,
                    "manufacturer": man.value.decode(errors="replace"),
                    "product": prod.value.decode(errors="replace"),
                    "serial": serial})
    return out


def list_serials() -> list[str]:
    """Serial numbers of the connected dongles (e.g. ['00000001'])."""
    return [d["serial"] for d in device_list() if d["serial"]]


def direct_sampling_mode(freq_hz: float, rx=None) -> int:
    """0 = tuner, 1 = I branch, 2 = Q branch. 'auto' uses direct sampling below 24 MHz."""
    setting = str(getattr(rx, "direct_sampling", "auto") or "auto").strip().lower()
    if setting in ("i", "1", "branch-i", "i-branch"):
        return 1
    if setting in ("q", "2", "on", "yes", "true", "branch-q", "q-branch"):
        return 2
    if setting in ("off", "0", "no", "false", "tuner"):
        return 0
    return 2 if freq_hz < DIRECT_SAMPLING_MAX_HZ else 0


def iq_from_uint8(raw) -> np.ndarray:
    """RTL-SDR's interleaved uint8 I/Q -> complex64 around zero."""
    a = (np.asarray(raw, dtype=np.uint8).ravel() if isinstance(raw, np.ndarray)
         else np.frombuffer(raw, dtype=np.uint8))
    if len(a) % 2:
        a = a[:-1]
    if not len(a):
        return np.zeros(0, dtype=np.complex64)
    a = (a.astype(np.float32) - 127.5) * (1.0 / 128.0)
    return (a[0::2] + 1j * a[1::2]).astype(np.complex64)


class RTLDevice:
    """with RTLDevice(768000, settings.receiver) as rtl: iq = rtl.capture(9.6e6, 0.5)"""

    def __init__(self, sample_rate: int, rx=None):
        self.sr = sample_rate
        self.rx = rx
        self.lib = _lib()
        self.dev = ctypes.c_void_p()
        self._lock = threading.Lock()
        self._want = 0
        self._skip = 0
        self._chunks: list[np.ndarray] = []
        self._done = threading.Event()
        self._sink = None
        self._reader: threading.Thread | None = None
        self._reader_done = threading.Event()
        self._error: str | None = None
        self._cb = _CB(self._on_samples)   # keep a reference: the C side calls it

    def __enter__(self):
        serial = parse_serial(self.rx.serial if self.rx else "")
        index = self._index(serial)
        rc = self.lib.rtlsdr_open(ctypes.byref(self.dev), index)
        if (rc != 0 or not self.dev) and serial:
            others = [x for x in list_serials() if x != serial]
            raise SerialNotFound(f"cannot open the RTL-SDR {serial} chosen in the "
                                 f"settings (not plugged in, or another program is using it)"
                                 + (f". Connected: {', '.join(others)}" if others else ""))
        if rc != 0 or not self.dev:
            raise RuntimeError("cannot open the RTL-SDR (not plugged in, or another program "
                               "is using it)")
        if self.lib.rtlsdr_set_sample_rate(self.dev, self.sr) != 0:
            self.lib.rtlsdr_close(self.dev)
            raise RuntimeError(f"the RTL-SDR does not support {self.sr} samples/s")
        self.lib.rtlsdr_set_agc_mode(self.dev, 0)          # keep the IQ linear
        if self.rx is not None:
            ppm = int(round(getattr(self.rx, "freq_correction_ppm", 0.0) or 0))
            if ppm:
                self.lib.rtlsdr_set_freq_correction(self.dev, ppm)
            self._set_gain()
        self.lib.rtlsdr_reset_buffer(self.dev)
        self._reader = threading.Thread(target=self._read_loop, daemon=True,
                                        name="rtlsdr-reader")
        self._reader.start()
        return self

    def __exit__(self, *exc):
        try:
            self.lib.rtlsdr_cancel_async(self.dev)
        except Exception:
            pass
        if self._reader is not None:
            self._reader.join(timeout=3)
        try:
            self.lib.rtlsdr_close(self.dev)
        except Exception:
            pass

    def _index(self, serial: str) -> int:
        if serial:
            idx = int(self.lib.rtlsdr_get_index_by_serial(serial.encode()))
            if idx < 0:
                others = [x for x in list_serials() if x != serial]
                raise SerialNotFound(f"the RTL-SDR {serial} chosen in the settings is not "
                                     f"connected" + (f". Found: {', '.join(others)}" if others else ""))
            return idx
        idx = int(getattr(self.rx, "device_index", 0) or 0) if self.rx is not None else 0
        return idx

    def _set_gain(self) -> None:
        if getattr(self.rx, "agc", True):
            self.lib.rtlsdr_set_tuner_gain_mode(self.dev, 0)       # automatic
        else:
            self.lib.rtlsdr_set_tuner_gain_mode(self.dev, 1)
            self.lib.rtlsdr_set_tuner_gain(self.dev, int(round(self.rx.gain_db * 10)))

    def _read_loop(self) -> None:
        try:
            rc = self.lib.rtlsdr_read_async(self.dev, self._cb, None,
                                            DEFAULT_BUF_NUM, DEFAULT_BUF_LEN)
            if rc != 0:
                self._error = f"the RTL-SDR stopped reading ({rc})"
        except Exception as e:                     # keep the reason for capture/stream
            self._error = str(e)
        finally:
            self._reader_done.set()

    def _on_samples(self, buf, length, ctx) -> None:
        n = int(length) // 2                       # two bytes per complex sample
        if n <= 0:
            return
        with self._lock:
            if self._want <= 0:
                return
            if self._skip > 0:
                self._skip = max(0, self._skip - n)
                return
            block = iq_from_uint8(np.ctypeslib.as_array(buf, shape=(int(length),)))
            if self._sink is not None:
                self._sink.put(block)
            else:
                self._chunks.append(block)
            self._want -= len(block)
            if self._want <= 0:
                self._done.set()

    def _tune(self, freq_hz: float) -> None:
        mode = direct_sampling_mode(freq_hz, self.rx)
        if self.lib.rtlsdr_set_direct_sampling(self.dev, mode) != 0:
            raise RuntimeError("cannot enable direct sampling on this RTL-SDR; set "
                               "receiver.direct_sampling = 'off' and use an upconverter")
        if self.lib.rtlsdr_set_center_freq(self.dev, int(freq_hz)) != 0:
            if mode == 0 and freq_hz < DIRECT_SAMPLING_MAX_HZ:
                raise RuntimeError(f"{freq_hz / 1e6:.3f} MHz is below the R820T tuner's "
                                   f"24 MHz limit - enable direct sampling in the settings")
            raise RuntimeError(f"cannot tune the RTL-SDR to {freq_hz / 1e6:.6f} MHz")

    def stream(self, freq_hz: float, total: int, settle_s: float = 0.0):
        """Yield consecutive IQ blocks (complex64) until `total` samples were delivered."""
        import queue
        q: queue.Queue = queue.Queue()
        with self._lock:
            self._want = 0
            self._skip = 0
            self._chunks = []
        self._sink = q
        self._tune(freq_hz)
        with self._lock:
            self._skip = int(settle_s * self.sr)
            self._want = total
        got = 0
        try:
            while got < total:
                try:
                    block = q.get(timeout=5)
                except queue.Empty:
                    raise RuntimeError(self._error or "timeout waiting for samples")
                got += len(block)
                yield block
        finally:
            with self._lock:
                self._want = 0
            self._sink = None

    def capture(self, freq_hz: float, seconds: float, settle_s: float = 0.15) -> np.ndarray:
        with self._lock:
            self._chunks = []
            self._done.clear()
            self._want = 0
            self._skip = 0
        self._tune(freq_hz)
        with self._lock:
            self._skip = int(settle_s * self.sr)
            self._want = int(seconds * self.sr)
        if not self._done.wait(seconds + settle_s + 3):
            with self._lock:
                self._want = 0
            raise RuntimeError(self._error or "timeout waiting for samples")
        with self._lock:
            chunks, self._chunks = self._chunks, []
        return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.complex64)
