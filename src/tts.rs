//! Kokoro та Piper через ONNX Runtime (CUDA EP, ліміт арени ~1.5 ГБ VRAM).
use ort::{
    execution_providers::{CUDAExecutionProvider, ExecutionProvider},
    session::Session,
    value::Tensor,
};
use std::{collections::HashMap, fs::File, path::{Path, PathBuf}, process::Command};
use unicode_normalization::UnicodeNormalization;

fn es<E: std::fmt::Display>(e: E) -> String {
    e.to_string()
}

pub fn data_dir() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(".local/share/tts-overlay")
}

fn session(path: &Path) -> Result<Session, String> {
    Session::builder()
        .map_err(es)?
        .with_execution_providers([CUDAExecutionProvider::default()
            .with_device_id(0)
            .with_memory_limit(1536 * 1024 * 1024)
            .build()])
        .map_err(es)?
        .commit_from_file(path)
        .map_err(|e| format!("{}: {e}", path.display()))
}

pub fn cuda_ok() -> bool {
    matches!(CUDAExecutionProvider::default().is_available(), Ok(true))
}

fn espeak(voice: &str, text: &str) -> Result<String, String> {
    let o = Command::new("espeak-ng")
        .args(["-q", "--ipa", "-v", voice, text.trim_start_matches('-')])
        .output()
        .map_err(|e| format!("espeak-ng: {e}"))?;
    Ok(String::from_utf8_lossy(&o.stdout).split_whitespace().collect::<Vec<_>>().join(" "))
}

fn t_i64(shape: &[usize], v: Vec<i64>) -> Result<Tensor<i64>, String> {
    Tensor::from_array((shape.to_vec(), v)).map_err(es)
}
fn t_f32(shape: &[usize], v: Vec<f32>) -> Result<Tensor<f32>, String> {
    Tensor::from_array((shape.to_vec(), v)).map_err(es)
}


/// Мінімальний читач .npz (numpy.savez, без стиснення): (ім'я, f32-дані).
fn read_npz(path: &Path, with_data: bool) -> Result<Vec<(String, Vec<f32>)>, String> {
    let b = std::fs::read(path).map_err(es)?;
    let (mut pos, mut out) = (0usize, vec![]);
    let u16at = |i: usize| u16::from_le_bytes([b[i], b[i + 1]]) as usize;
    let u32at = |i: usize| u32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]]) as u64;
    while pos + 30 <= b.len() && u32at(pos) == 0x04034b50 {
        let (flags, method) = (u16at(pos + 6), u16at(pos + 8));
        let (mut csize, nlen, xlen) = (u32at(pos + 18), u16at(pos + 26), u16at(pos + 28));
        let name = String::from_utf8_lossy(&b[pos + 30..pos + 30 + nlen]).to_string();
        // zip64: реальні розміри лежать в extra-полі 0x0001
        if csize == 0xFFFF_FFFF {
            let mut x = pos + 30 + nlen;
            let end = x + xlen;
            while x + 4 <= end {
                let (id, len) = (u16at(x), u16at(x + 2));
                if id == 1 && len >= 16 {
                    csize = u64::from_le_bytes(b[x + 12..x + 20].try_into().unwrap());
                }
                x += 4 + len;
            }
        }
        if method != 0 || flags & 8 != 0 {
            return Err("voices.bin must be an uncompressed npz".into());
        }
        let d0 = pos + 30 + nlen + xlen;
        let d1 = d0 + csize as usize;
        let key = name.trim_end_matches(".npy").to_string();
        if with_data {
            let npy = &b[d0..d1];
            let hlen = if npy[6] == 1 { 2 + u16::from_le_bytes([npy[8], npy[9]]) as usize } else { 4 + u32::from_le_bytes([npy[8], npy[9], npy[10], npy[11]]) as usize };
            let data = &npy[8 + hlen..];
            out.push((key, data.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()));
        } else {
            out.push((key, vec![]));
        }
        pos = d1;
    }
    Ok(out)
}

// ───────────── каталоги голосів ─────────────
pub fn kokoro_voice_names() -> Vec<String> {
    let Ok(e) = read_npz(&data_dir().join("voices.bin"), false) else { return vec![] };
    let mut v: Vec<String> = e.into_iter().map(|(n, _)| n).filter(|n| n.starts_with('a') || n.starts_with('b')).collect();
    v.sort();
    v
}

pub struct PiperEntry {
    pub label: String, // "name" або "name#speaker"
    pub lang: String,  // "uk", "en", ...
}

/// Будь-який Piper-голос (також власний) з ~/.local/share/tts-overlay/piper/*.onnx(+.json)
pub fn piper_catalog() -> Vec<PiperEntry> {
    let mut out = vec![];
    if let Ok(rd) = std::fs::read_dir(data_dir().join("piper")) {
        for e in rd.flatten() {
            let n = e.file_name().to_string_lossy().to_string();
            let Some(stem) = n.strip_suffix(".onnx.json") else { continue };
            let Ok(f) = File::open(e.path()) else { continue };
            let Ok(j) = serde_json::from_reader::<_, serde_json::Value>(f) else { continue };
            let lang = j["language"]["code"].as_str().unwrap_or("").split('_').next().unwrap_or("").to_string();
            let ns = j["num_speakers"].as_i64().unwrap_or(1);
            if ns > 1 {
                for k in 0..ns {
                    out.push(PiperEntry { label: format!("{stem}#{k}"), lang: lang.clone() });
                }
            } else {
                out.push(PiperEntry { label: stem.to_string(), lang });
            }
        }
    }
    out.sort_by(|a, b| a.label.cmp(&b.label));
    out
}

// ───────────── Kokoro ─────────────
struct Kokoro {
    sess: Session,
    vocab: HashMap<char, i64>,
    voices: HashMap<String, Vec<f32>>, // [510,1,256] -> плоский вектор
}

impl Kokoro {
    fn load() -> Result<Self, String> {
        let d = data_dir();
        let sess = session(&d.join("kokoro.onnx"))?;
        let cfg: serde_json::Value =
            serde_json::from_reader(File::open(d.join("kokoro-config.json")).map_err(es)?).map_err(es)?;
        let vocab = cfg["vocab"]
            .as_object()
            .ok_or("kokoro-config.json has no vocab")?
            .iter()
            .filter_map(|(k, v)| Some((k.chars().next()?, v.as_i64()?)))
            .collect();
        let voices = read_npz(&d.join("voices.bin"), true)?.into_iter().collect();
        Ok(Self { sess, vocab, voices })
    }

    fn style(&self, a: &str, b: Option<&str>, mix: f32, n: usize) -> Result<Vec<f32>, String> {
        let row = |name: &str| -> Result<&[f32], String> {
            let v = self.voices.get(name).ok_or(format!("voice not found: {name}"))?;
            Ok(&v[n * 256..(n + 1) * 256])
        };
        let ra = row(a)?;
        Ok(match b {
            Some(b) => {
                let rb = row(b)?;
                ra.iter().zip(rb).map(|(x, y)| x * (1.0 - mix) + y * mix).collect()
            }
            None => ra.to_vec(),
        })
    }

    fn synth(&mut self, text: &str, a: &str, b: Option<&str>, mix: f32, speed: f32) -> Result<Vec<f32>, String> {
        let ph = espeak(if a.starts_with('b') { "en-gb" } else { "en-us" }, text)?;
        let mut ids: Vec<i64> = ph.chars().filter_map(|c| self.vocab.get(&c).copied()).collect();
        ids.truncate(509);
        if ids.is_empty() {
            return Ok(vec![]);
        }
        let style = self.style(a, b, mix, ids.len())?;
        let mut toks = vec![0i64];
        toks.extend(&ids);
        toks.push(0);
        let out = self
            .sess
            .run(ort::inputs![
                "tokens" => t_i64(&[1, toks.len()], toks)?,
                "style" => t_f32(&[1, 256], style)?,
                "speed" => t_f32(&[1], vec![speed])?
            ])
            .map_err(es)?;
        let (_, data) = out["audio"].try_extract_tensor::<f32>().map_err(es)?;
        Ok(data.to_vec())
    }
}

// ───────────── Piper ─────────────
struct Piper {
    sess: Session,
    pub rate: u32,
    espeak: String,
    map: HashMap<String, Vec<i64>>,
    scales: [f32; 3],
    multi: bool,
    text_mode: bool, // phoneme_type == "text": таблиця за літерами, без espeak
}

impl Piper {
    fn load(file: &str) -> Result<Self, String> {
        let base = data_dir().join("piper").join(file);
        let sess = session(&base.with_extension("onnx"))?;
        let cfg: serde_json::Value =
            serde_json::from_reader(File::open(format!("{}.onnx.json", base.display())).map_err(es)?).map_err(es)?;
        let map = cfg["phoneme_id_map"]
            .as_object()
            .ok_or("no phoneme_id_map in voice json")?
            .iter()
            .map(|(k, v)| (k.clone(), v.as_array().unwrap().iter().filter_map(|x| x.as_i64()).collect()))
            .collect();
        let f = |k: &str, d: f64| cfg["inference"][k].as_f64().unwrap_or(d) as f32;
        Ok(Self {
            sess,
            rate: cfg["audio"]["sample_rate"].as_u64().unwrap_or(22050) as u32,
            espeak: cfg["espeak"]["voice"].as_str().unwrap_or("en-us").to_string(),
            map,
            scales: [f("noise_scale", 0.667), f("length_scale", 1.0), f("noise_w", 0.8)],
            multi: cfg["num_speakers"].as_i64().unwrap_or(1) > 1,
            text_mode: cfg["phoneme_type"].as_str() == Some("text"),
        })
    }

    fn synth(&mut self, text: &str, speed: f32, sid: i64) -> Result<Vec<f32>, String> {
        let ph: String = if self.text_mode { text.nfd().collect() } else { espeak(&self.espeak, text)?.nfd().collect() };
        let get = |k: &str| self.map.get(k).cloned().unwrap_or_default();
        let (bos, pad, eos) = (get("^"), get("_"), get("$"));
        let mut ids = bos;
        for ch in ph.chars() {
            if let Some(v) = self.map.get(&ch.to_string()).or_else(|| self.map.get(&ch.to_lowercase().to_string())) {
                ids.extend(v);
                ids.extend(&pad);
            }
        }
        ids.extend(eos);
        let n = ids.len();
        let scales = vec![self.scales[0], self.scales[1] / speed.max(0.1), self.scales[2]];
        let out = if self.multi {
            self.sess.run(ort::inputs![
                "input" => t_i64(&[1, n], ids)?,
                "input_lengths" => t_i64(&[1], vec![n as i64])?,
                "scales" => t_f32(&[3], scales)?,
                "sid" => t_i64(&[1], vec![sid])?
            ])
        } else {
            self.sess.run(ort::inputs![
                "input" => t_i64(&[1, n], ids)?,
                "input_lengths" => t_i64(&[1], vec![n as i64])?,
                "scales" => t_f32(&[3], scales)?
            ])
        }
        .map_err(es)?;
        let (_, data) = out["output"].try_extract_tensor::<f32>().map_err(es)?;
        Ok(data.to_vec())
    }
}

// ───────────── диспетчер ─────────────
#[derive(Clone, Copy, PartialEq)]
pub enum Engine {
    Kokoro,
    Piper,
}

#[derive(Clone)]
pub struct Params {
    pub engine: Engine,
    pub voice: String,
    pub voice_b: Option<String>,
    pub mix: f32,
    pub speed: f32,
    pub text: String,
}

#[derive(Default)]
pub struct Engines {
    kokoro: Option<Kokoro>,
    piper: HashMap<String, Piper>,
}

impl Engines {
    /// Повертає (семпли, частота).
    pub fn synth(&mut self, p: &Params, s: &str) -> Result<(Vec<f32>, u32), String> {
        match p.engine {
            Engine::Kokoro => {
                if self.kokoro.is_none() {
                    self.kokoro = Some(Kokoro::load()?);
                }
                let k = self.kokoro.as_mut().unwrap();
                Ok((k.synth(s, &p.voice, p.voice_b.as_deref(), p.mix, p.speed)?, 24000))
            }
            Engine::Piper => {
                let (file, sid) = match p.voice.split_once('#') {
                    Some((f, k)) => (f.to_string(), k.parse().unwrap_or(0)),
                    None => (p.voice.clone(), 0),
                };
                if !self.piper.contains_key(&file) {
                    self.piper.insert(file.clone(), Piper::load(&file)?);
                }
                let v = self.piper.get_mut(&file).unwrap();
                let x = v.synth(s, p.speed, sid)?;
                Ok((x, v.rate))
            }
        }
    }
}
