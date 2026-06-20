import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const AUDIO_SR = 44_100;
const OVERSAMPLE = 8;
const SIM_SR = AUDIO_SR * OVERSAMPLE;
const N = 131_072;
const MAX_ANALYSIS_F = 120_000;
const MIN_LOG_F = 10;
const LOG_LO = Math.log10(300);
const LOG_HI = Math.log10(AUDIO_SR);
const EPS = 1e-12;
const SPECTRUM_FLOOR_DB = -120;
function fftInPlace(re, im) {
  const n = re.length;

  for (let i = 1, j = 0; i < n; i += 1) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;

    if (i < j) {
      let t = re[i];
      re[i] = re[j];
      re[j] = t;
      t = im[i];
      im[i] = im[j];
      im[j] = t;
    }
  }

  for (let len = 2; len <= n; len <<= 1) {
    const angle = (-2 * Math.PI) / len;
    const stepRe = Math.cos(angle);
    const stepIm = Math.sin(angle);

    for (let start = 0; start < n; start += len) {
      let twRe = 1;
      let twIm = 0;
      const half = len >> 1;

      for (let j = 0; j < half; j += 1) {
        const even = start + j;
        const odd = even + half;
        const oddRe = re[odd] * twRe - im[odd] * twIm;
        const oddIm = re[odd] * twIm + im[odd] * twRe;
        const evenRe = re[even];
        const evenIm = im[even];

        re[even] = evenRe + oddRe;
        im[even] = evenIm + oddIm;
        re[odd] = evenRe - oddRe;
        im[odd] = evenIm - oddIm;

        const nextRe = twRe * stepRe - twIm * stepIm;
        twIm = twRe * stepIm + twIm * stepRe;
        twRe = nextRe;
      }
    }
  }
}

function computeSpectrum(sig) {
  const n = sig.length;
  const re = new Float64Array(n);
  const im = new Float64Array(n);
  let windowSum = 0;

  for (let i = 0; i < n; i += 1) {
    const phase = (2 * Math.PI * i) / (n - 1);
    const w =
      0.2712203605850388 -
      0.4334446123274422 * Math.cos(phase) +
      0.218004121590025 * Math.cos(2 * phase) -
      0.0657853432956061 * Math.cos(3 * phase) +
      0.010761867305342 * Math.cos(4 * phase) -
      0.000770012710581 * Math.cos(5 * phase) +
      0.000013680883059 * Math.cos(6 * phase);
    re[i] = sig[i] * w;
    windowSum += w;
  }

  fftInPlace(re, im);

  const mag = new Float32Array(n >> 1);
  const scale = 2 / windowSum;

  for (let i = 0; i < mag.length; i += 1) {
    mag[i] = Math.hypot(re[i], im[i]) * scale;
  }

  return mag;
}

const dB = (value) => 20 * Math.log10(Math.max(Math.abs(value), 1e-9));

function buildSpectrumRows(
  spectra,
  sr,
  minF,
  maxF,
  targetBuckets = 1800,
  scaleType = "linear",
) {
  const fftSize = spectra[0].mag.length * 2;
  const hzPerBin = sr / fftSize;
  const minBin = Math.max(1, Math.floor(Math.max(0, minF) / hzPerBin));
  const maxBin = Math.min(
    spectra[0].mag.length - 1,
    Math.max(minBin + 1, Math.ceil(maxF / hzPerBin)),
  );
  const selectedBins = new Set([minBin, maxBin]);

  const addBucket = (start, end) => {
    const first = Math.max(minBin, Math.min(maxBin, start));
    const last = Math.max(first + 1, Math.min(maxBin + 1, end));
    let bestBin = first;
    let bestValue = -Infinity;

    for (let bin = first; bin < last; bin += 1) {
      let value = 0;
      for (const spectrum of spectra) value = Math.max(value, spectrum.mag[bin]);
      if (value > bestValue) {
        bestValue = value;
        bestBin = bin;
      }
    }

    selectedBins.add(bestBin);
  };

  if (scaleType === "log") {
    const low = Math.max(hzPerBin, minF || hzPerBin);
    const high = Math.max(low + hzPerBin, maxF);
    const logLow = Math.log10(low);
    const logHigh = Math.log10(high);

    for (let bucket = 0; bucket < targetBuckets; bucket += 1) {
      const startF = 10 ** (logLow + ((logHigh - logLow) * bucket) / targetBuckets);
      const endF = 10 ** (logLow + ((logHigh - logLow) * (bucket + 1)) / targetBuckets);
      addBucket(Math.floor(startF / hzPerBin), Math.ceil(endF / hzPerBin));
    }
  } else {
    const binsPerBucket = Math.max(1, Math.ceil((maxBin - minBin + 1) / targetBuckets));
    for (let start = minBin; start <= maxBin; start += binsPerBucket) {
      addBucket(start, start + binsPerBucket);
    }
  }

  return Array.from(selectedBins)
    .sort((a, b) => a - b)
    .map((bin) => {
      const row = { f: +(bin * hzPerBin).toFixed(2) };
      for (const spectrum of spectra) {
        row[spectrum.name] = +Math.max(SPECTRUM_FLOOR_DB, dB(spectrum.mag[bin])).toFixed(2);
      }
      return row;
    });
}
function positiveModulo(value, modulus) {
  return ((value % modulus) + modulus) % modulus;
}

function polyBlep(phase, phaseStep) {
  if (phaseStep <= 0 || phaseStep >= 1) return 0;

  if (phase < phaseStep) {
    const t = phase / phaseStep;
    return t + t - t * t - 1;
  }

  if (phase > 1 - phaseStep) {
    const t = (phase - 1) / phaseStep;
    return t * t + t + t + 1;
  }

  return 0;
}


function waveAt(type, frequency, amplitude, time, renderSampleRate = SIM_SR) {
  const phase = positiveModulo(frequency * time, 1);
  const phaseStep = Math.min(0.499, frequency / renderSampleRate);

  if (type === "sin") {
    return amplitude * Math.sin(2 * Math.PI * phase);
  }

  if (type === "sq") {
    let value = phase < 0.5 ? 1 : -1;
    value += polyBlep(phase, phaseStep);
    value -= polyBlep(positiveModulo(phase + 0.5, 1), phaseStep);
    return amplitude * value;
  }

  if (type === "saw") {
    let value = 2 * phase - 1;
    value -= polyBlep(phase, phaseStep);
    return amplitude * value;
  }
  return amplitude * (1 - 4 * Math.abs(phase - 0.5));
}

function renderInput(type, frequency, amplitude, length, sampleRate) {
  const out = new Float32Array(length);
  for (let i = 0; i < length; i += 1) {
    out[i] = waveAt(type, frequency, amplitude, i / sampleRate, sampleRate);
  }
  return out;
}

function renderSaH({
  type,
  frequency,
  amplitude,
  length,
  renderSampleRate,
  holdSampleRate,
  droopTauSeconds = 0,
  apertureFraction = 0,
}) {
  const out = new Float32Array(length);
  const holdPeriod = 1 / holdSampleRate;
  const apertureTime = Math.min(0.999, Math.max(0, apertureFraction)) * holdPeriod;
  const phaseStep = holdSampleRate / renderSampleRate;
  const endHoldAge = Math.max(0, holdPeriod - apertureTime);

  let currentCycle = Number.NaN;
  let sampleStart = 0;
  let holdStart = 0;
  let heldValue = 0;
  let currentStep = 0;
  let nextStep = 0;

  const endValueForCycle = (cycle) => {
    const cycleStart = cycle / holdSampleRate;
    const value = waveAt(
      type,
      frequency,
      amplitude,
      cycleStart + apertureTime,
      renderSampleRate,
    );
    return droopTauSeconds > 0
      ? value * Math.exp(-endHoldAge / droopTauSeconds)
      : value;
  };

  const startValueForCycle = (cycle) =>
    waveAt(
      type,
      frequency,
      amplitude,
      cycle / holdSampleRate,
      renderSampleRate,
    );

  for (let i = 0; i < length; i += 1) {
    const time = i / renderSampleRate;
    const exactCycle = time * holdSampleRate;
    const cycle = Math.floor(exactCycle + EPS);
    const phase = exactCycle - cycle;

    if (cycle !== currentCycle) {
      currentCycle = cycle;
      sampleStart = cycle / holdSampleRate;
      holdStart = sampleStart + apertureTime;
      heldValue = waveAt(
        type,
        frequency,
        amplitude,
        holdStart,
        renderSampleRate,
      );
      currentStep = startValueForCycle(cycle) - endValueForCycle(cycle - 1);
      nextStep = startValueForCycle(cycle + 1) - endValueForCycle(cycle);
    }

    let value;
    if (apertureTime > 0 && time < holdStart) {
      value = waveAt(type, frequency, amplitude, time, renderSampleRate);
    } else if (droopTauSeconds > 0) {
      value = heldValue * Math.exp(-Math.max(0, time - holdStart) / droopTauSeconds);
    } else {
      value = heldValue;
    }

    if (phaseStep > 0 && phaseStep < 0.5) {
      if (phase < phaseStep) {
        const x = phase / phaseStep;
        const blep = x + x - x * x - 1;
        value += 0.5 * currentStep * blep;
      } else if (phase > 1 - phaseStep) {
        const x = (phase - 1) / phaseStep;
        const blep = x * x + x + x + 1;
        value += 0.5 * nextStep * blep;
      }
    }

    out[i] = value;
  }

  return out;
}
function rcLP(inp, sampleRate, cutoff) {
  const a = (2 * Math.PI * cutoff) / (2 * Math.PI * cutoff + sampleRate);
  const out = new Float32Array(inp.length);
  out[0] = inp[0];

  for (let i = 1; i < inp.length; i += 1) {
    out[i] = a * inp[i] + (1 - a) * out[i - 1];
  }

  return out;
}

function bqLP(inp, sampleRate, cutoff, q) {
  const w = (2 * Math.PI * cutoff) / sampleRate;
  const cosW = Math.cos(w);
  const sinW = Math.sin(w);
  const alpha = sinW / (2 * q);
  const norm = 1 / (1 + alpha);
  const b0 = ((1 - cosW) / 2) * norm;
  const b1 = (1 - cosW) * norm;
  const b2 = b0;
  const a1 = -2 * cosW * norm;
  const a2 = (1 - alpha) * norm;
  const out = new Float32Array(inp.length);

  let x1 = 0;
  let x2 = 0;
  let y1 = 0;
  let y2 = 0;

  for (let i = 0; i < inp.length; i += 1) {
    const x0 = inp[i];
    const y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
    out[i] = y0;
    x2 = x1;
    x1 = x0;
    y2 = y1;
    y1 = y0;
  }

  return out;
}

function bw4LP(inp, sampleRate, cutoff) {
  return bqLP(
    bqLP(inp, sampleRate, cutoff, 0.5411961),
    sampleRate,
    cutoff,
    1.306563,
  );
}

function bw8LP(inp, sampleRate, cutoff) {
  let stage = bqLP(inp, sampleRate, cutoff, 0.5097956);
  stage = bqLP(stage, sampleRate, cutoff, 0.6013449);
  stage = bqLP(stage, sampleRate, cutoff, 0.8999762);
  return bqLP(stage, sampleRate, cutoff, 2.5629154);
}

function idealLP(inp, sampleRate, cutoff) {
  const n = inp.length;
  const re = Float64Array.from(inp);
  const im = new Float64Array(n);
  fftInPlace(re, im);

  const cutBin = Math.min(n >> 1, Math.max(0, Math.floor((cutoff / sampleRate) * n)));
  for (let i = cutBin + 1; i < n - cutBin; i += 1) {
    re[i] = 0;
    im[i] = 0;
  }
  for (let i = 0; i < n; i += 1) im[i] = -im[i];
  fftInPlace(re, im);

  const out = new Float32Array(n);
  for (let i = 0; i < n; i += 1) out[i] = re[i] / n;
  return out;
}

function applyLPF(inp, sampleRate, type, cutoff) {
  const f = Math.min(Math.max(1, cutoff), sampleRate * 0.499);
  if (type === "rc1") return rcLP(inp, sampleRate, f);
  if (type === "bw2") return bqLP(inp, sampleRate, f, 0.7071068);
  if (type === "sk2") return bqLP(inp, sampleRate, f, 0.9565);
  if (type === "bw4") return bw4LP(inp, sampleRate, f);
  if (type === "bw8") return bw8LP(inp, sampleRate, f);
  if (type === "ideal") return idealLP(inp, sampleRate, f);
  return inp.slice();
}


function makeAudioPreview(inp) {
  const filtered = bw8LP(inp, SIM_SR, 20_000);
  const outLength = Math.floor(filtered.length / OVERSAMPLE);
  const out = new Float32Array(outLength);
  for (let i = 0; i < outLength; i += 1) out[i] = filtered[i * OVERSAMPLE];
  return out;
}
function zohTF(frequency, holdSampleRate) {
  if (!frequency) return 1;
  const x = (Math.PI * frequency) / holdSampleRate;
  return Math.abs(Math.sin(x) / x);
}

function lpfTF(frequency, type, cutoff) {
  const u = frequency / cutoff;
  if (type === "rc1") return 1 / Math.sqrt(1 + u * u);
  if (type === "bw2") return 1 / Math.sqrt(1 + u ** 4);
  if (type === "sk2") {
    const epsilon = Math.sqrt(10 ** 0.1 - 1);
    const t2 = 2 * u * u - 1;
    return 1 / Math.sqrt(1 + epsilon * epsilon * t2 * t2);
  }
  if (type === "bw4") return 1 / Math.sqrt(1 + u ** 8);
  if (type === "bw8") return 1 / Math.sqrt(1 + u ** 16);
  if (type === "ideal") return u <= 1 ? 1 : 1e-9;
  return 1;
}
const TYPE_ORDER = { fund: 0, harm: 1, alias: 2, zoh: 3 };

function foldToNyquist(frequency, sampleRate) {
  let folded = positiveModulo(frequency, sampleRate);
  if (folded > sampleRate / 2) folded = sampleRate - folded;
  return Math.abs(folded);
}

function harmonicStep(type) {
  return type === "sq" || type === "tri" ? 2 : 1;
}

function firstHarmonic(type) {
  return type === "sin" ? 1 : type === "sq" || type === "tri" ? 1 : 1;
}

function computePeaks(type, f0, holdSampleRate, maxF) {
  const nyquist = holdSampleRate / 2;
  const visibleMax = Math.min(maxF, SIM_SR / 2 - 1);
  const sourceLimit = SIM_SR * 0.45;
  const seen = new Map();

  const add = (frequency, label, peakType, color) => {
    if (!(frequency > 1 && frequency <= visibleMax)) return;
    const key = Math.round(frequency * 10) / 10;
    const existing = seen.get(key);
    if (
      !existing ||
      (TYPE_ORDER[peakType] ?? 9) < (TYPE_ORDER[existing.type] ?? 9)
    ) {
      seen.set(key, { f: frequency, label, type: peakType, color });
    }
  };

  const step = harmonicStep(type);
  const start = firstHarmonic(type);

  for (let k = start; k * f0 <= sourceLimit; k += step) {
    if (type === "sin" && k > 1) break;

    const harmonicFrequency = k * f0;
    const folded = foldToNyquist(harmonicFrequency, holdSampleRate);
    const harmonicLabel = k === 1 ? "f₀" : `${k}f₀`;

    if (harmonicFrequency <= nyquist) {
      add(
        harmonicFrequency,
        k === 1 ? "Grundton" : `${k}. Oberton`,
        k === 1 ? "fund" : "harm",
        k === 1 ? "#60a5fa" : "#34d399",
      );
    } else if (folded > 1) {
      add(
        folded,
        `Alias (${harmonicLabel}=${Math.round(harmonicFrequency)}Hz)`,
        "alias",
        "#f87171",
      );
    }

    if (folded > 1) {
      for (let image = 1; image * holdSampleRate - folded <= visibleMax; image += 1) {
        const lower = image * holdSampleRate - folded;
        const upper = image * holdSampleRate + folded;
        add(
          lower,
          `ZOH ${image}·fs−${harmonicLabel}`,
          "zoh",
          "#c084fc",
        );
        add(
          upper,
          `ZOH ${image}·fs+${harmonicLabel}`,
          "zoh",
          "#c084fc",
        );
      }
    }
  }

  return Array.from(seen.values()).sort((a, b) => a.f - b.f);
}


function harmonicCoefficient(type, harmonic, amplitude) {
  if (harmonic < 1) return null;

  if (type === "sin") {
    return harmonic === 1 ? { re: 0, im: -amplitude / 2 } : null;
  }

  if (type === "sq") {
    if (harmonic % 2 === 0) return null;
    return { re: 0, im: (-2 * amplitude) / (Math.PI * harmonic) };
  }

  if (type === "saw") {
    return { re: 0, im: amplitude / (Math.PI * harmonic) };
  }

  if (harmonic % 2 === 0) return null;
  return {
    re: (-4 * amplitude) / (Math.PI * Math.PI * harmonic * harmonic),
    im: 0,
  };
}

function addComplexComponent(map, frequency, re, im, harmonic, aliased) {
  const key = Math.round(frequency * 1e6) / 1e6;
  const magnitude = Math.hypot(re, im);
  const current = map.get(key) || {
    f: frequency,
    re: 0,
    im: 0,
    dominantHarmonic: harmonic,
    dominantMagnitude: -1,
    hasAlias: false,
    hasDirect: false,
  };

  current.re += re;
  current.im += im;
  current.hasAlias ||= aliased;
  current.hasDirect ||= !aliased;

  if (magnitude > current.dominantMagnitude) {
    current.dominantMagnitude = magnitude;
    current.dominantHarmonic = harmonic;
  }

  map.set(key, current);
}

function analyticalPeakLabel(component, isImage, imageIndex, side) {
  const harmonic = component.dominantHarmonic;

  if (isImage) {
    const base = harmonic === 1 ? "f₀" : `${harmonic}f₀`;
    return `ZOH ${imageIndex}·fs${side}${base}`;
  }

  if (component.hasAlias) {
    return harmonic === 1 ? "Alias f₀" : `Alias ${harmonic}f₀`;
  }

  if (harmonic === 1) return "Grundton";
  return `${harmonic}. Oberton`;
}

function buildSimplifiedSpectrum({
  type,
  f0,
  amplitude,
  holdSampleRate,
  minF,
  maxF,
  scaleType,
  showSaH,
  lpfOn,
  lpfType,
  lpfCut,
}) {
  const sourceLimit = SIM_SR * 0.45;
  const maxHarmonic = Math.min(8192, Math.floor(sourceLimit / f0));
  const nyquist = holdSampleRate / 2;
  const baseComponents = new Map();
  const lines = new Map();

  const addLine = (frequency, series, peakAmplitude, label, typeName, color) => {
    if (!(frequency >= minF && frequency <= maxF) || peakAmplitude <= 0) return;
    const dbValue = Math.max(SPECTRUM_FLOOR_DB, dB(peakAmplitude));
    if (dbValue <= SPECTRUM_FLOOR_DB + 0.01) return;

    const key = Math.round(frequency * 1e6) / 1e6;
    const current = lines.get(key) || {
      f: frequency,
      Eingang: null,
      "S&H": null,
      Ausgang: null,
      label,
      type: typeName,
      color,
    };

    if (current[series] == null || dbValue > current[series]) current[series] = dbValue;

    if ((TYPE_ORDER[typeName] ?? 9) < (TYPE_ORDER[current.type] ?? 9)) {
      current.label = label;
      current.type = typeName;
      current.color = color;
    }

    lines.set(key, current);
  };

  for (let harmonic = 1; harmonic <= maxHarmonic; harmonic += 1) {
    const coefficient = harmonicCoefficient(type, harmonic, amplitude);
    if (!coefficient) continue;

    const harmonicFrequency = harmonic * f0;
    const harmonicPeak = 2 * Math.hypot(coefficient.re, coefficient.im);

    if (harmonicFrequency <= maxF) {
      addLine(
        harmonicFrequency,
        "Eingang",
        harmonicPeak,
        harmonic === 1 ? "Grundton" : `${harmonic}. Oberton`,
        harmonic === 1 ? "fund" : "harm",
        harmonic === 1 ? "#60a5fa" : "#34d399",
      );
    }

    const modulo = positiveModulo(harmonicFrequency, holdSampleRate);
    const tolerance = Math.max(1e-9, holdSampleRate * 1e-10);
    const aliased = harmonicFrequency > nyquist + tolerance;

    if (modulo < tolerance || Math.abs(modulo - holdSampleRate) < tolerance) {
      addComplexComponent(
        baseComponents,
        0,
        2 * coefficient.re,
        0,
        harmonic,
        aliased,
      );
    } else if (Math.abs(modulo - nyquist) < tolerance) {
      addComplexComponent(
        baseComponents,
        nyquist,
        2 * coefficient.re,
        0,
        harmonic,
        aliased,
      );
    } else if (modulo < nyquist) {
      addComplexComponent(
        baseComponents,
        modulo,
        coefficient.re,
        coefficient.im,
        harmonic,
        aliased,
      );
    } else {
      addComplexComponent(
        baseComponents,
        holdSampleRate - modulo,
        coefficient.re,
        -coefficient.im,
        harmonic,
        aliased,
      );
    }
  }

  const addHeldLine = (frequency, component, imageIndex, side, special = false) => {
    if (!(frequency >= minF && frequency <= maxF)) return;
    const coefficientMagnitude = Math.hypot(component.re, component.im);
    const basePeak = special ? coefficientMagnitude : 2 * coefficientMagnitude;
    const heldPeak = basePeak * zohTF(frequency, holdSampleRate);
    const isImage = imageIndex > 0;
    const label = analyticalPeakLabel(component, isImage, imageIndex, side);
    const typeName = isImage ? "zoh" : component.hasAlias ? "alias" : component.dominantHarmonic === 1 ? "fund" : "harm";
    const color = typeName === "zoh" ? "#c084fc" : typeName === "alias" ? "#f87171" : typeName === "fund" ? "#60a5fa" : "#34d399";

    if (showSaH) addLine(frequency, "S&H", heldPeak, label, typeName, color);
    const outputPeak = heldPeak * (lpfOn ? lpfTF(frequency, lpfType, lpfCut) : 1);
    addLine(frequency, "Ausgang", outputPeak, label, typeName, color);
  };

  for (const component of baseComponents.values()) {
    const frequency = component.f;

    if (frequency < 1e-8) {
      addHeldLine(0, component, 0, "", true);
      continue;
    }

    if (Math.abs(frequency - nyquist) < 1e-8) {
      for (let image = 0; frequency + image * holdSampleRate <= maxF; image += 1) {
        addHeldLine(frequency + image * holdSampleRate, component, image, "+", true);
      }
      continue;
    }

    addHeldLine(frequency, component, 0, "");

    for (let image = 1; image * holdSampleRate - frequency <= maxF; image += 1) {
      const lower = image * holdSampleRate - frequency;
      const upper = image * holdSampleRate + frequency;
      addHeldLine(lower, component, image, "−");
      addHeldLine(upper, component, image, "+");
    }
  }

  let components = Array.from(lines.values()).filter((line) =>
    [line.Eingang, line["S&H"], line.Ausgang].some((value) => value != null),
  );

  const maxLines = 700;
  if (components.length > maxLines) {
    components = components
      .sort((a, b) => {
        const aLevel = Math.max(a.Eingang ?? -Infinity, a["S&H"] ?? -Infinity, a.Ausgang ?? -Infinity);
        const bLevel = Math.max(b.Eingang ?? -Infinity, b["S&H"] ?? -Infinity, b.Ausgang ?? -Infinity);
        return bLevel - aLevel;
      })
      .slice(0, maxLines);
  }

  components.sort((a, b) => a.f - b.f);

  const rows = [];
  const domainMin = scaleType === "log" ? Math.max(MIN_LOG_F, minF) : minF;
  const domainSpan = Math.max(1, maxF - domainMin);

  components.forEach((component, index) => {
    const previous = components[index - 1]?.f ?? domainMin;
    const next = components[index + 1]?.f ?? maxF;
    const nominal = scaleType === "log"
      ? Math.max(component.f * 0.0025, 0.02)
      : Math.max(domainSpan / 5000, 0.05);
    const leftGap = Math.max(0.001, component.f - previous);
    const rightGap = Math.max(0.001, next - component.f);
    const delta = Math.max(0.001, Math.min(nominal, leftGap / 3, rightGap / 3));
    const left = Math.max(domainMin, component.f - delta);
    const right = Math.min(maxF, component.f + delta);
    const active = {
      Eingang: component.Eingang != null,
      "S&H": showSaH && component["S&H"] != null,
      Ausgang: component.Ausgang != null,
    };

    rows.push({
      f: left,
      Eingang: active.Eingang ? SPECTRUM_FLOOR_DB : null,
      "S&H": active["S&H"] ? SPECTRUM_FLOOR_DB : null,
      Ausgang: active.Ausgang ? SPECTRUM_FLOOR_DB : null,
    });
    rows.push({
      f: component.f,
      Eingang: component.Eingang,
      "S&H": showSaH ? component["S&H"] : null,
      Ausgang: component.Ausgang,
    });
    rows.push({
      f: right,
      Eingang: active.Eingang ? SPECTRUM_FLOOR_DB : null,
      "S&H": active["S&H"] ? SPECTRUM_FLOOR_DB : null,
      Ausgang: active.Ausgang ? SPECTRUM_FLOOR_DB : null,
    });
    rows.push({
      f: Math.min(maxF, right + Math.max(1e-6, delta * 0.05)),
      Eingang: null,
      "S&H": null,
      Ausgang: null,
    });
  });

  const peaks = components
    .filter((component) => component.f > 1)
    .map((component) => ({
      f: component.f,
      label: component.label,
      type: component.type,
      color: component.color,
    }));

  return { specComb: rows, pkList: peaks };
}

const MO = {
  fontFamily: "system-ui, -apple-system, BlinkMacSystemFont, \"Segoe UI\", sans-serif",
  fontVariantNumeric: "tabular-nums",
};
const GR = "var(--color-border-tertiary)";
const AX = "var(--color-text-secondary)";

function Slider({
  label,
  value,
  unit = "",
  min,
  max,
  step = 1,
  onChange,
  accent = "#60a5fa",
}) {
  return (
    <div style={{ marginBottom: 10 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
        <span style={{ fontSize: 18, color: AX, ...MO }}>{label}</span>
        <span style={{ fontSize: 18, fontWeight: 600, color: "var(--color-text-primary)", ...MO }}>
          {step < 1 ? Number(value).toFixed(2) : value}
          {unit}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) =>
          onChange(step < 1 ? Number.parseFloat(event.target.value) : Number.parseInt(event.target.value, 10))
        }
        style={{ width: "100%", accentColor: accent }}
      />
    </div>
  );
}

function Panel({ title, accent = "var(--color-text-secondary)", children }) {
  return (
    <div
      style={{
        background: "var(--color-background-primary)",
        border: "0.5px solid var(--color-border-tertiary)",
        borderRadius: 8,
        marginBottom: 8,
        overflow: "hidden",
      }}
    >
      <div
        style={{
          padding: "9px 14px",
          fontSize: 18,
          fontWeight: 600,
          color: accent,
          background: "var(--color-background-secondary)",
          borderBottom: "0.5px solid var(--color-border-tertiary)",
          ...MO,
          textTransform: "uppercase",
          letterSpacing: "0.07em",
        }}
      >
        {title}
      </div>
      <div style={{ padding: "14px" }}>{children}</div>
    </div>
  );
}

function ChkBox({ label, checked, onChange, color = "#60a5fa", note }) {
  return (
    <div style={{ marginBottom: 8 }}>
      <label
        style={{
          display: "flex",
          alignItems: "center",
          gap: 6,
          fontSize: 18,
          cursor: "pointer",
          color: "var(--color-text-primary)",
          ...MO,
        }}
      >
        <input
          type="checkbox"
          checked={checked}
          onChange={(event) => onChange(event.target.checked)}
          style={{ accentColor: color }}
        />
        {label}
      </label>
      {note && (
        <div style={{ fontSize: 18, color: AX, marginTop: 3, marginLeft: 20, lineHeight: 1.55 }}>
          {note}
        </div>
      )}
    </div>
  );
}

function TipBox({ active, payload, label, fmtLabel = (v) => v, fmtVal = (v) => v.toFixed(3) }) {
  if (!active || !payload?.length) return null;

  return (
    <div
      style={{
        background: "var(--color-background-secondary)",
        border: "0.5px solid var(--color-border-secondary)",
        borderRadius: 6,
        padding: "6px 10px",
        fontSize: 18,
        ...MO,
      }}
    >
      <div style={{ color: AX, marginBottom: 3 }}>{fmtLabel(label)}</div>
      {payload.map((item, index) => (
        <div key={`${item.dataKey}-${index}`} style={{ color: item.stroke || item.color }}>
          {item.name || item.dataKey}: {fmtVal(item.value)}
        </div>
      ))}
    </div>
  );
}

function TabBtn({ active, onClick, children }) {
  return (
    <button
      type="button"
      onClick={onClick}
      style={{
        padding: "10px 16px",
        fontSize: 18,
        cursor: "pointer",
        border: "none",
        background: active ? "var(--color-background-primary)" : "transparent",
        color: active ? "var(--color-text-primary)" : AX,
        borderBottom: active ? "2px solid #60a5fa" : "2px solid transparent",
        ...MO,
      }}
    >
      {children}
    </button>
  );
}

const FILT = {
  rc1: "RC 1. Ordnung (−6 dB/Okt)",
  bw2: "Butterworth 2. Ordnung (−12 dB/Okt)",
  sk2: "Sallen-Key / Tschebyscheff-ähnlich",
  bw4: "Butterworth 4. Ordnung (−24 dB/Okt)",
  bw8: "Butterworth 8. Ordnung (−48 dB/Okt)",
  ideal: "Idealfilter / Brickwall (Referenz)",
};
export default function SaHSimulator() {
  const [wtype, setWtype] = useState("sin");
  const [freq, setFreq] = useState(440);
  const [amp, setAmp] = useState(0.8);
  const [fs, setFs] = useState(8_000);
  const [fsText, setFsText] = useState("8000");
  const [lpfOn, setLpfOn] = useState(true);
  const [lpfType, setLpfType] = useState("bw2");
  const [lpfCut, setLpfCut] = useState(3_800);
  const [efxOpen, setEfxOpen] = useState(false);
  const [droopOn, setDroopOn] = useState(false);
  const [droopTau, setDroopTau] = useState(50);
  const [aperOn, setAperOn] = useState(false);
  const [aperPct, setAperPct] = useState(5);
  const [tab, setTab] = useState("time");
  const [inPlay, setInPlay] = useState(false);
  const [outPlay, setOutPlay] = useState(false);
  const [showSaH, setShowSaH] = useState(true);
  const [specMin, setSpecMin] = useState(MIN_LOG_F);
  const [specMax, setSpecMax] = useState(60_000);
  const [specScale, setSpecScale] = useState("log");
  const [spectrumMode, setSpectrumMode] = useState("simplified");

  const audioCtxRef = useRef(null);
  const playersRef = useRef({ input: null, output: null });
  const playRequestRef = useRef({ input: 0, output: 0 });

  const stopPlayer = useCallback((key, updateState = true) => {
    playRequestRef.current[key] += 1;
    const player = playersRef.current[key];

    if (player) {
      player.source.onended = null;
      try {
        player.source.stop();
      } catch (_) {
      }
      try {
        player.source.disconnect();
        player.gain.disconnect();
      } catch (_) {
      }
      playersRef.current[key] = null;
    }

    if (updateState) {
      if (key === "input") setInPlay(false);
      else setOutPlay(false);
    }
  }, []);

  const stopAllPlayers = useCallback((updateState = true) => {
    stopPlayer("input", updateState);
    stopPlayer("output", updateState);
  }, [stopPlayer]);

  const startPlayer = useCallback(async (key, sig) => {
    stopPlayer(key);
    const requestId = playRequestRef.current[key];

    let context = audioCtxRef.current;
    if (!context || context.state === "closed") {
      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      context = new AudioContextClass({ latencyHint: "interactive" });
      audioCtxRef.current = context;
    }

    if (context.state === "suspended") await context.resume();
    if (requestId !== playRequestRef.current[key]) return;

    const preview = makeAudioPreview(sig);
    const buffer = context.createBuffer(1, preview.length, AUDIO_SR);
    buffer.copyToChannel(preview, 0);

    const source = context.createBufferSource();
    const gain = context.createGain();
    gain.gain.value = 0.8;
    source.buffer = buffer;
    source.loop = true;
    source.connect(gain).connect(context.destination);

    source.onended = () => {
      if (playersRef.current[key]?.source === source) {
        playersRef.current[key] = null;
        if (key === "input") setInPlay(false);
        else setOutPlay(false);
      }
    };

    playersRef.current[key] = { source, gain };
    source.start();

    if (key === "input") setInPlay(true);
    else setOutPlay(true);
  }, [stopPlayer]);

  useEffect(() => () => {
    stopAllPlayers(false);
    const context = audioCtxRef.current;
    audioCtxRef.current = null;
    if (context && context.state !== "closed") {
      void context.close();
    }
  }, [stopAllPlayers]);

  const nyquist = fs / 2;
  const fundamentalAliasing = freq > nyquist;
  const fsLog = Math.log10(fs);
  const maxLpfCut = Math.max(100, Math.min(20_000, Math.floor(nyquist * 0.98)));

  useEffect(() => {
    setLpfCut((current) => Math.min(current, maxLpfCut));
  }, [maxLpfCut]);

  const { inSig, sahSig, outSig } = useMemo(() => {
    const input = renderInput(wtype, freq, amp, N, SIM_SR);
    const sah = renderSaH({
      type: wtype,
      frequency: freq,
      amplitude: amp,
      length: N,
      renderSampleRate: SIM_SR,
      holdSampleRate: fs,
      droopTauSeconds: droopOn ? droopTau / 1000 : 0,
      apertureFraction: aperOn ? aperPct / 100 : 0,
    });
    const output = lpfOn ? applyLPF(sah, SIM_SR, lpfType, lpfCut) : sah.slice();
    return { inSig: input, sahSig: sah, outSig: output };
  }, [wtype, freq, amp, fs, droopOn, droopTau, aperOn, aperPct, lpfOn, lpfType, lpfCut]);
  useEffect(() => {
    stopAllPlayers();
  }, [inSig, outSig, stopAllPlayers]);

  const timeData = useMemo(() => {
    if (tab !== "time") return [];

    const samplesForFourPeriods = Math.min(
      Math.ceil((SIM_SR / freq) * 4),
      Math.floor(SIM_SR * 0.12),
      N,
    );
    const displayStep = Math.max(1, Math.floor(samplesForFourPeriods / 1800));
    const rows = [];

    for (let i = 0; i < samplesForFourPeriods; i += displayStep) {
      const row = {
        t: +((i / SIM_SR) * 1000).toFixed(5),
        Eingang: +inSig[i].toFixed(5),
        Ausgang: +outSig[i].toFixed(5),
      };
      if (showSaH) row["S&H"] = +sahSig[i].toFixed(5);
      rows.push(row);
    }

    return rows;
  }, [tab, inSig, sahSig, outSig, freq, showSaH]);

  const spectrumMagnitudes = useMemo(() => {
    if (tab !== "spec" || spectrumMode !== "fft") return null;

    return {
      input: computeSpectrum(inSig),
      sah: computeSpectrum(sahSig),
      output: computeSpectrum(outSig),
    };
  }, [tab, spectrumMode, inSig, sahSig, outSig]);

  const simplifiedSpectrum = useMemo(() => {
    if (tab !== "spec" || spectrumMode !== "simplified") return null;

    return buildSimplifiedSpectrum({
      type: wtype,
      f0: freq,
      amplitude: amp,
      holdSampleRate: fs,
      minF: specMin,
      maxF: specMax,
      scaleType: specScale,
      showSaH,
      lpfOn,
      lpfType,
      lpfCut,
    });
  }, [
    tab,
    spectrumMode,
    wtype,
    freq,
    amp,
    fs,
    specMin,
    specMax,
    specScale,
    showSaH,
    lpfOn,
    lpfType,
    lpfCut,
  ]);

  const { specComb, pkList } = useMemo(() => {
    if (spectrumMode === "simplified") {
      return simplifiedSpectrum || { specComb: [], pkList: [] };
    }

    if (!spectrumMagnitudes) return { specComb: [], pkList: [] };

    const spectra = [
      { name: "Eingang", mag: spectrumMagnitudes.input },
      ...(showSaH ? [{ name: "S&H", mag: spectrumMagnitudes.sah }] : []),
      { name: "Ausgang", mag: spectrumMagnitudes.output },
    ];

    return {
      specComb: buildSpectrumRows(
        spectra,
        SIM_SR,
        specMin,
        specMax,
        2400,
        specScale,
      ),
      pkList: computePeaks(wtype, freq, fs, specMax).filter(
        (peak) => peak.f >= specMin,
      ),
    };
  }, [
    spectrumMode,
    simplifiedSpectrum,
    spectrumMagnitudes,
    wtype,
    freq,
    fs,
    showSaH,
    specMin,
    specMax,
    specScale,
  ]);

  const tfData = useMemo(() => {
    if (tab !== "tf") return [];

    const maxF = Math.min(specMax, fs * 2.5, SIM_SR / 2 - 1);
    return Array.from({ length: 900 }, (_, index) => {
      const f = Math.max(1, ((index + 1) / 900) * maxF);
      const zoh = zohTF(f, fs);
      const lpf = lpfOn ? lpfTF(f, lpfType, lpfCut) : 1;
      return {
        f: +f.toFixed(2),
        ZOH: +dB(zoh).toFixed(2),
        LPF: +dB(lpf).toFixed(2),
        Gesamt: +dB(zoh * lpf).toFixed(2),
      };
    });
  }, [tab, fs, lpfOn, lpfType, lpfCut, specMax]);

  const selectedPeaks = useMemo(
    () => [
      ...pkList.filter((peak) => peak.type === "fund" || peak.type === "harm").slice(0, 5),
      ...pkList.filter((peak) => peak.type === "alias").slice(0, 4),
      ...pkList.filter((peak) => peak.type === "zoh").slice(0, 4),
    ],
    [pkList],
  );

  const togglePlayer = useCallback((key, sig, playing) => {
    if (playing) stopPlayer(key);
    else void startPlayer(key, sig);
  }, [startPlayer, stopPlayer]);

  const setFsFromLog = (value) => {
    const hz = Math.round(10 ** Number.parseFloat(value));
    setFs(hz);
    setFsText(String(hz));
  };

  const setSpectrumRange = (minimum, maximum) => {
    const safeMax = Math.max(20, Math.min(MAX_ANALYSIS_F, Number(maximum) || 20));
    const safeMin = Math.max(0, Math.min(safeMax - 1, Number(minimum) || 0));
    setSpecMin(safeMin);
    setSpecMax(safeMax);
  };


  const handleFsText = (event) => {
    setFsText(event.target.value);
    const value = Number.parseFloat(event.target.value);
    if (Number.isFinite(value) && value >= 300 && value <= AUDIO_SR) {
      setFs(Math.round(value));
    }
  };

  const selectStyle = {
    background: "var(--color-background-secondary)",
    border: "0.5px solid var(--color-border-secondary)",
    borderRadius: 6,
    color: "var(--color-text-primary)",
    padding: "7px 10px",
    fontSize: 18,
    width: "100%",
    cursor: "pointer",
    ...MO,
  };
  const chartTick = { fontSize: 18, ...MO };

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100vh", fontSize: 18 }}>
      <div
        style={{
          background: "var(--color-background-secondary)",
          borderBottom: "0.5px solid var(--color-border-tertiary)",
          padding: "8px 16px",
          display: "flex",
          alignItems: "center",
          gap: 12,
          flexWrap: "wrap",
        }}
      >
        <span style={{ fontWeight: 600, fontSize: 22, ...MO }}>S&amp;H Simulator</span>
        {fundamentalAliasing && (
          <span
            style={{
              marginLeft: "auto",
              background: "var(--color-background-danger)",
              border: "0.5px solid var(--color-border-danger)",
              color: "var(--color-text-danger)",
              fontSize: 18,
              padding: "2px 8px",
              borderRadius: 4,
              ...MO,
            }}
          >
            ⚠ Grundton-Alias: f₀={freq} Hz &gt; Nyquist={nyquist} Hz
          </span>
        )}
      </div>

      <div style={{ display: "flex", flex: 1, overflow: "hidden" }}>
        <div
          style={{
            width: 390,
            flexShrink: 0,
            background: "var(--color-background-secondary)",
            borderRight: "0.5px solid var(--color-border-tertiary)",
            overflowY: "auto",
            padding: 10,
          }}
        >
          <Panel title="Eingang" accent="#60a5fa">
            <div style={{ display: "flex", gap: 3, marginBottom: 10 }}>
              {[
                ["sin", "Sinus"],
                ["sq", "Rechteck"],
                ["saw", "Sägezahn"],
                ["tri", "Dreieck"],
              ].map(([value, label]) => (
                <button
                  type="button"
                  key={value}
                  onClick={() => setWtype(value)}
                  style={{
                    flex: 1,
                    fontSize: 18,
                    padding: "8px 4px",
                    borderRadius: 4,
                    cursor: "pointer",
                    ...MO,
                    border: `0.5px solid ${wtype === value ? "#3b82f6" : "var(--color-border-tertiary)"}`,
                    background: wtype === value ? "#1d4ed8" : "var(--color-background-tertiary)",
                    color: wtype === value ? "#fff" : AX,
                  }}
                >
                  {label}
                </button>
              ))}
            </div>
            <Slider label="Frequenz" value={freq} unit=" Hz" min={50} max={8_000} onChange={setFreq} />
            <Slider label="Amplitude" value={amp} min={0.1} max={1} step={0.01} onChange={setAmp} />
          </Panel>

          <Panel title="Abtastung" accent="#f97316">
            <div style={{ fontSize: 18, color: AX, marginBottom: 5, ...MO }}>
              S&amp;H-Abtastrate (logarithmisch)
            </div>
            <div style={{ display: "flex", gap: 6, alignItems: "center", marginBottom: 6 }}>
              <input
                value={fsText}
                onChange={handleFsText}
                onBlur={() => setFsText(String(fs))}
                style={{ ...selectStyle, width: 88, padding: "4px 6px" }}
              />
              <span style={{ fontSize: 18, color: AX }}>Hz</span>
            </div>
            <input
              type="range"
              min={LOG_LO}
              max={LOG_HI}
              step={0.001}
              value={fsLog}
              onChange={(event) => setFsFromLog(event.target.value)}
              style={{ width: "100%", accentColor: "#f97316" }}
            />
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 18, color: AX, marginTop: 3, ...MO }}>
              {[
                "300",
                "1k",
                "4k",
                "16k",
                "44k",
              ].map((value) => <span key={value}>{value}</span>)}
            </div>
            <div
              style={{
                padding: "3px 8px",
                borderRadius: 4,
                fontSize: 18,
                marginTop: 7,
                ...MO,
                background: fundamentalAliasing ? "var(--color-background-danger)" : "var(--color-background-success)",
                color: fundamentalAliasing ? "var(--color-text-danger)" : "var(--color-text-success)",
                border: `0.5px solid ${fundamentalAliasing ? "var(--color-border-danger)" : "var(--color-border-success)"}`,
              }}
            >
              Grundton: Nyquist {nyquist.toFixed(0)} Hz {fundamentalAliasing ? "→ Alias" : "✓"}
            </div>
          </Panel>

          {tab === "spec" && (
            <Panel title="Spektrum">
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, marginBottom: 12 }}>
                <button
                  type="button"
                  onClick={() => setSpectrumMode("simplified")}
                  style={{
                    ...selectStyle,
                    background: spectrumMode === "simplified" ? "var(--color-background-tertiary)" : "var(--color-background-secondary)",
                    border: `1px solid ${spectrumMode === "simplified" ? "#60a5fa" : "var(--color-border-secondary)"}`,
                    fontWeight: spectrumMode === "simplified" ? 700 : 500,
                  }}
                >
                  Vereinfacht
                </button>
                <button
                  type="button"
                  onClick={() => setSpectrumMode("fft")}
                  style={{
                    ...selectStyle,
                    background: spectrumMode === "fft" ? "var(--color-background-tertiary)" : "var(--color-background-secondary)",
                    border: `1px solid ${spectrumMode === "fft" ? "#60a5fa" : "var(--color-border-secondary)"}`,
                    fontWeight: spectrumMode === "fft" ? 700 : 500,
                  }}
                >
                  FFT
                </button>
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, marginBottom: 12 }}>
                <button
                  type="button"
                  onClick={() => {
                    setSpecScale("linear");
                    setSpecMin(0);
                  }}
                  style={{
                    ...selectStyle,
                    background: specScale === "linear" ? "var(--color-background-tertiary)" : "var(--color-background-secondary)",
                    border: `1px solid ${specScale === "linear" ? "#60a5fa" : "var(--color-border-secondary)"}`,
                    fontWeight: specScale === "linear" ? 700 : 500,
                  }}
                >
                  Linear
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setSpecScale("log");
                    if (specMin < MIN_LOG_F) setSpecMin(MIN_LOG_F);
                  }}
                  style={{
                    ...selectStyle,
                    background: specScale === "log" ? "var(--color-background-tertiary)" : "var(--color-background-secondary)",
                    border: `1px solid ${specScale === "log" ? "#60a5fa" : "var(--color-border-secondary)"}`,
                    fontWeight: specScale === "log" ? 700 : 500,
                  }}
                >
                  Logarithmisch
                </button>
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, marginBottom: 10 }}>
                <label style={{ display: "grid", gap: 5, color: AX, ...MO }}>
                  Start (Hz)
                  <input
                    type="number"
                    min={specScale === "log" ? MIN_LOG_F : 0}
                    max={Math.max(0, specMax - 1)}
                    value={Math.round(specMin)}
                    onChange={(event) => setSpectrumRange(event.target.value, specMax)}
                    style={selectStyle}
                  />
                </label>
                <label style={{ display: "grid", gap: 5, color: AX, ...MO }}>
                  Ende (Hz)
                  <input
                    type="number"
                    min={20}
                    max={MAX_ANALYSIS_F}
                    value={Math.round(specMax)}
                    onChange={(event) => setSpectrumRange(specMin, event.target.value)}
                    style={selectStyle}
                  />
                </label>
              </div>
              <input
                type="range"
                min={Math.log10(500)}
                max={Math.log10(MAX_ANALYSIS_F)}
                step={0.001}
                value={Math.log10(Math.max(500, specMax))}
                onChange={(event) => setSpectrumRange(specMin, 10 ** Number.parseFloat(event.target.value))}
                style={{ width: "100%", accentColor: "#60a5fa", marginBottom: 12 }}
              />
              <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                {[1_000, 2_000, 5_000, 10_000, 20_000, 60_000, 120_000].map((value) => (
                  <button
                    type="button"
                    key={value}
                    onClick={() => setSpectrumRange(specScale === "log" ? MIN_LOG_F : 0, value)}
                    style={{ ...selectStyle, width: "auto", padding: "7px 10px" }}
                  >
                    {value >= 1_000 ? `${value / 1_000} kHz` : `${value} Hz`}
                  </button>
                ))}
              </div>
            </Panel>
          )}

          <Panel title="Tiefpass" accent="#4ade80">
            <ChkBox label="Tiefpass aktiv" checked={lpfOn} onChange={setLpfOn} color="#4ade80" />
            {lpfOn && (
              <>
                <select
                  value={lpfType}
                  onChange={(event) => setLpfType(event.target.value)}
                  style={{ ...selectStyle, marginTop: 4, marginBottom: 8 }}
                >
                  {Object.entries(FILT).map(([key, label]) => (
                    <option key={key} value={key}>{label}</option>
                  ))}
                </select>
                <Slider
                  label="Grenzfrequenz"
                  value={lpfCut}
                  unit=" Hz"
                  min={100}
                  max={maxLpfCut}
                  onChange={setLpfCut}
                  accent="#4ade80"
                />
              </>
            )}
          </Panel>

          <div style={{ border: "0.5px solid var(--color-border-tertiary)", borderRadius: 8, overflow: "hidden", marginBottom: 8 }}>
            <button
              type="button"
              onClick={() => setEfxOpen((open) => !open)}
              style={{
                width: "100%",
                display: "flex",
                justifyContent: "space-between",
                padding: "7px 12px",
                background: "var(--color-background-secondary)",
                color: AX,
                fontSize: 18,
                border: "none",
                cursor: "pointer",
                ...MO,
                textTransform: "uppercase",
                letterSpacing: "0.07em",
                fontWeight: 500,
              }}
            >
              <span>Schaltung</span>
              <span>{efxOpen ? "▲" : "▼"}</span>
            </button>
            {efxOpen && (
              <div style={{ padding: 12, borderTop: "0.5px solid var(--color-border-tertiary)" }}>
                <ChkBox
                  label="Kondensator-Droop"
                  checked={droopOn}
                  onChange={setDroopOn}
                  color="#fbbf24"
                />
                {droopOn && (
                  <Slider
                    label="Zeitkonstante τ"
                    value={droopTau}
                    unit=" ms"
                    min={1}
                    max={1_000}
                    onChange={setDroopTau}
                    accent="#fbbf24"
                  />
                )}
                <ChkBox
                  label="Track-/Aperturzeit"
                  checked={aperOn}
                  onChange={setAperOn}
                  color="#fbbf24"
                />
                {aperOn && (
                  <Slider
                    label="Track-Anteil"
                    value={aperPct}
                    unit="%"
                    min={1}
                    max={50}
                    onChange={setAperPct}
                    accent="#fbbf24"
                  />
                )}
              </div>
            )}
          </div>



          <Panel title="Audio">
            <div style={{ display: "flex", gap: 6, marginBottom: 8 }}>
              <button
                type="button"
                onClick={() => togglePlayer("input", inSig, inPlay)}
                style={{
                  flex: 1,
                  padding: "8px 6px",
                  borderRadius: 4,
                  cursor: "pointer",
                  fontSize: 18,
                  ...MO,
                  border: `0.5px solid ${inPlay ? "#60a5fa" : "var(--color-border-secondary)"}`,
                  background: inPlay ? "#1d4ed820" : "var(--color-background-secondary)",
                  color: inPlay ? "#60a5fa" : AX,
                }}
              >
                {inPlay ? "⏹" : "▶"} Eingang
              </button>
              <button
                type="button"
                onClick={() => togglePlayer("output", outSig, outPlay)}
                style={{
                  flex: 1,
                  padding: "8px 6px",
                  borderRadius: 4,
                  cursor: "pointer",
                  fontSize: 18,
                  ...MO,
                  border: `0.5px solid ${outPlay ? "#4ade80" : "var(--color-border-secondary)"}`,
                  background: outPlay ? "#16a34a20" : "var(--color-background-secondary)",
                  color: outPlay ? "#4ade80" : AX,
                }}
              >
                {outPlay ? "⏹" : "▶"} Ausgang
              </button>
            </div>
            <ChkBox label="S&H-Signal anzeigen" checked={showSaH} onChange={setShowSaH} color="#f97316" />
          </Panel>
        </div>

        <div style={{ flex: 1, display: "flex", flexDirection: "column", minWidth: 0 }}>
          <div style={{ display: "flex", background: "var(--color-background-secondary)", borderBottom: "0.5px solid var(--color-border-tertiary)" }}>
            {[
              ["time", "Zeitbereich"],
              ["spec", "Spektrum"],
              ["tf", "Übertragung"],
            ].map(([key, label]) => (
              <TabBtn key={key} active={tab === key} onClick={() => setTab(key)}>
                {label}
              </TabBtn>
            ))}
          </div>

          <div style={{ flex: 1, overflowY: "auto", padding: "18px 22px" }}>
            {tab === "time" && (
              <div>
                <div style={{ display: "flex", gap: 16, marginBottom: 10, flexWrap: "wrap", fontSize: 18 }}>
                  <span style={{ color: "#60a5fa" }}>— Eingang</span>
                  {showSaH && <span style={{ color: "#f97316" }}>— S&amp;H</span>}
                  <span style={{ color: "#4ade80" }}>— Filterausgang</span>
                </div>
                <div style={{ background: "var(--color-background-secondary)", borderRadius: 8, border: "0.5px solid var(--color-border-tertiary)", paddingTop: 8 }}>
                  <ResponsiveContainer width="100%" height={360}>
                    <LineChart data={timeData} margin={{ top: 18, right: 28, bottom: 36, left: 66 }}>
                      <CartesianGrid strokeDasharray="2 6" stroke={GR} />
                      <XAxis
                        dataKey="t"
                        stroke={AX}
                        tick={chartTick}
                        type="number"
                        domain={["dataMin", "dataMax"]}
                        tickFormatter={(value) => Number(value).toFixed(2)}
                        label={{ value: "Zeit (ms)", position: "insideBottomRight", dy: 18, fontSize: 18, fill: AX }}
                      />
                      <YAxis
                        stroke={AX}
                        tick={chartTick}
                        domain={[-1.3, 1.3]}
                        label={{ value: "Amplitude", angle: -90, position: "insideLeft", dx: -10, fontSize: 18, fill: AX }}
                      />
                      <Tooltip content={<TipBox fmtLabel={(value) => `${Number(value).toFixed(3)} ms`} fmtVal={(value) => Number(value).toFixed(5)} />} />
                      <Line isAnimationActive={false} type="linear" dataKey="Eingang" stroke="#60a5fa" dot={false} strokeWidth={1.4} />
                      {showSaH && <Line isAnimationActive={false} type="linear" dataKey="S&H" stroke="#f97316" dot={false} strokeWidth={1.5} />}
                      <Line isAnimationActive={false} type="linear" dataKey="Ausgang" stroke="#4ade80" dot={false} strokeWidth={2} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}

            {tab === "spec" && (
              <div>
                <div style={{ display: "flex", gap: 10, marginBottom: 8, flexWrap: "wrap", fontSize: 18 }}>
                  <span style={{ color: "#60a5fa" }}>— Eingang</span>
                  {showSaH && <span style={{ color: "#f97316" }}>— S&amp;H</span>}
                  <span style={{ color: "#4ade80" }}>— Ausgang</span>
                  <span style={{ color: "#f87171", marginLeft: 8 }}>│ Alias</span>
                  <span style={{ color: "#c084fc" }}>│ ZOH-Bild</span>
                  <span style={{ color: "#fbbf24" }}>│ S&amp;H-Nyquist</span>
                </div>
                <div style={{ background: "var(--color-background-secondary)", borderRadius: 8, border: "0.5px solid var(--color-border-tertiary)", paddingTop: 8 }}>
                  <ResponsiveContainer width="100%" height={540}>
                    <LineChart data={specComb} margin={{ top: 96, right: 42, bottom: 50, left: 90 }}>
                      <CartesianGrid strokeDasharray="2 6" stroke={GR} />
                      <XAxis
                        dataKey="f"
                        stroke={AX}
                        tick={chartTick}
                        type="number"
                        scale={specScale === "log" ? "log" : "linear"}
                        domain={[specScale === "log" ? Math.max(MIN_LOG_F, specMin) : specMin, specMax]}
                        allowDataOverflow
                        tickFormatter={(value) => (value >= 1000 ? `${(value / 1000).toFixed(value < 10_000 ? 1 : 0)}k` : String(Math.round(value)))}
                        label={{ value: "Frequenz (Hz)", position: "insideBottomRight", dy: 20, fontSize: 18, fill: AX }}
                      />
                      <YAxis
                        stroke={AX}
                        tick={chartTick}
                        domain={[SPECTRUM_FLOOR_DB, 5]}
                        label={{ value: "dBFS", angle: -90, position: "insideLeft", dx: -16, fontSize: 18, fill: AX }}
                      />
                      <Tooltip content={<TipBox fmtLabel={(value) => `${Number(value).toFixed(1)} Hz`} fmtVal={(value) => `${Number(value).toFixed(1)} dBFS`} />} />

                      {selectedPeaks.map((peak) => (
                        <ReferenceLine
                          key={`${peak.type}-${peak.f.toFixed(2)}-${peak.label}`}
                          x={peak.f}
                          stroke={peak.color}
                          strokeDasharray={peak.type === "alias" ? "3 2" : "4 3"}
                          strokeOpacity={0.82}
                          strokeWidth={1.25}
                          label={{ value: peak.label.slice(0, 16), position: "insideTopRight", fontSize: 16, fill: peak.color, angle: -45, dy: 38, dx: 8 }}
                        />
                      ))}
                      {nyquist >= specMin && nyquist <= specMax && (
                        <ReferenceLine
                          x={nyquist}
                          stroke="#fbbf24"
                          strokeWidth={1.5}
                          strokeDasharray="6 2"
                          label={{ value: `Nyquist ${nyquist} Hz`, position: "insideTopRight", fontSize: 18, fill: "#fbbf24", dy: 6, ...MO }}
                        />
                      )}
                      {lpfOn && lpfCut >= specMin && lpfCut <= specMax && (
                        <ReferenceLine
                          x={lpfCut}
                          stroke="#4ade80"
                          strokeWidth={1}
                          strokeDasharray="4 3"
                          label={{ value: `fc=${lpfCut} Hz`, position: "insideTopRight", fontSize: 18, fill: "#4ade80", dy: 6, ...MO }}
                        />
                      )}

                      <Line isAnimationActive={false} connectNulls={false} type="linear" dataKey="Eingang" stroke="#60a5fa" dot={false} strokeWidth={1.2} strokeOpacity={0.85} />
                      {showSaH && <Line isAnimationActive={false} connectNulls={false} type="linear" dataKey="S&H" stroke="#f97316" dot={false} strokeWidth={1.2} strokeOpacity={0.82} />}
                      <Line isAnimationActive={false} connectNulls={false} type="linear" dataKey="Ausgang" stroke="#4ade80" dot={false} strokeWidth={1.8} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}

            {tab === "tf" && (
              <div>
                <div style={{ display: "flex", gap: 14, marginBottom: 8, flexWrap: "wrap", fontSize: 18 }}>
                  <span style={{ color: "#f97316" }}>— ZOH-Sinc</span>
                  <span style={{ color: "#4ade80" }}>— Tiefpass</span>
                  <span style={{ color: "#60a5fa" }}>— Gesamt</span>
                </div>
                <div style={{ background: "var(--color-background-secondary)", borderRadius: 8, border: "0.5px solid var(--color-border-tertiary)", paddingTop: 8 }}>
                  <ResponsiveContainer width="100%" height={390}>
                    <LineChart data={tfData} margin={{ top: 42, right: 28, bottom: 36, left: 68 }}>
                      <CartesianGrid strokeDasharray="2 6" stroke={GR} />
                      <XAxis
                        dataKey="f"
                        stroke={AX}
                        tick={chartTick}
                        type="number"
                        domain={["dataMin", "dataMax"]}
                        tickFormatter={(value) => (value >= 1000 ? `${(value / 1000).toFixed(0)}k` : String(value))}
                        label={{ value: "Frequenz (Hz)", position: "insideBottomRight", dy: 18, fontSize: 18, fill: AX }}
                      />
                      <YAxis
                        stroke={AX}
                        tick={chartTick}
                        domain={[-100, 5]}
                        label={{ value: "dB", angle: -90, position: "insideLeft", dx: -14, fontSize: 18, fill: AX }}
                      />
                      <Tooltip content={<TipBox fmtLabel={(value) => `${Number(value).toFixed(0)} Hz`} fmtVal={(value) => `${Number(value).toFixed(1)} dB`} />} />
                      <ReferenceLine y={-3.92} stroke="#f97316" strokeDasharray="2 4" strokeOpacity={0.5} />
                      <ReferenceLine x={nyquist} stroke="#fbbf24" strokeWidth={1.3} strokeDasharray="6 2" label={{ value: "Nyquist", position: "insideTopRight", fontSize: 18, fill: "#fbbf24", dy: 6 }} />
                      <ReferenceLine x={fs} stroke={AX} strokeDasharray="3 3" label={{ value: "fs", position: "insideTopRight", fontSize: 18, fill: AX, dy: 6 }} />
                      {lpfOn && <ReferenceLine x={lpfCut} stroke="#4ade80" strokeDasharray="4 3" label={{ value: "fc", position: "insideTopRight", fontSize: 18, fill: "#4ade80", dy: 6 }} />}
                      <ReferenceLine x={freq} stroke="#60a5fa" strokeDasharray="2 4" label={{ value: "f₀", position: "insideTopRight", fontSize: 18, fill: "#60a5fa", dy: 6 }} />
                      <Line isAnimationActive={false} type="linear" dataKey="ZOH" stroke="#f97316" dot={false} strokeWidth={1.5} />
                      <Line isAnimationActive={false} type="linear" dataKey="LPF" stroke="#4ade80" dot={false} strokeWidth={1.5} />
                      <Line isAnimationActive={false} type="linear" dataKey="Gesamt" stroke="#60a5fa" dot={false} strokeWidth={2} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}
          </div>

          <div
            style={{
              background: "var(--color-background-secondary)",
              borderTop: "0.5px solid var(--color-border-tertiary)",
              padding: "4px 14px",
              display: "flex",
              flexWrap: "wrap",
              gap: 14,
              fontSize: 18,
              color: AX,
              ...MO,
            }}
          >
            <span style={{ color: "#f97316" }}>S&amp;H fs={fs} Hz</span>
            <span>Nyquist={nyquist} Hz</span>
            <span style={{ color: "#60a5fa" }}>f₀={freq} Hz</span>
            {lpfOn && <span style={{ color: "#4ade80" }}>fc={lpfCut} Hz ({lpfType})</span>}
            {droopOn && <span style={{ color: "#fbbf24" }}>τ={droopTau} ms</span>}
            {aperOn && <span style={{ color: "#fbbf24" }}>Track={aperPct} %</span>}
            {fundamentalAliasing && <span style={{ color: "var(--color-text-danger)" }}>⚠ GRUNDTON-ALIAS</span>}
          </div>
        </div>
      </div>
    </div>
  );
}
