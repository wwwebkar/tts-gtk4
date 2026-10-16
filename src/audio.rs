//! Вивід звуку через cpal (PipeWire/ALSA) + буфер для візуалізатора.
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

pub const SCOPE: usize = 2048;

#[derive(Clone)]
pub struct AudioOut {
    q: Arc<Mutex<VecDeque<f32>>>,
    pub scope: Arc<Mutex<Vec<f32>>>,
    pub rate: u32,
}

impl AudioOut {
    /// Потік cpal не Send, тому повертаємо його окремо (його треба тримати живим).
    pub fn new() -> Result<(Self, cpal::Stream), String> {
        let dev = cpal::default_host().default_output_device().ok_or("немає пристрою виводу")?;
        let cfg = dev.default_output_config().map_err(|e| e.to_string())?;
        if cfg.sample_format() != cpal::SampleFormat::F32 {
            return Err("потрібен f32-вивід".into());
        }
        let (rate, ch) = (cfg.sample_rate().0, cfg.channels() as usize);
        let q = Arc::new(Mutex::new(VecDeque::<f32>::new()));
        let scope = Arc::new(Mutex::new(vec![0.0f32; SCOPE]));
        let (q2, s2) = (q.clone(), scope.clone());
        let stream = dev
            .build_output_stream(
                &cfg.into(),
                move |out: &mut [f32], _| {
                    let mut q = q2.lock().unwrap();
                    let mut sp = s2.lock().unwrap();
                    for frame in out.chunks_mut(ch) {
                        let s = q.pop_front().unwrap_or(0.0);
                        frame.fill(s);
                        sp.push(s);
                    }
                    let l = sp.len();
                    if l > SCOPE {
                        sp.drain(..l - SCOPE);
                    }
                },
                |e| eprintln!("audio: {e}"),
                None,
            )
            .map_err(|e| e.to_string())?;
        stream.play().map_err(|e| e.to_string())?;
        Ok((Self { q, scope, rate }, stream))
    }

    /// Приймає семпли на 24 кГц.
    pub fn push(&self, x24k: &[f32]) {
        let x = crate::dsp::resample(x24k, crate::dsp::SR, self.rate);
        self.q.lock().unwrap().extend(x);
    }

    pub fn clear(&self) {
        self.q.lock().unwrap().clear();
    }
}
