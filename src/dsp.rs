//! Прості DSP-ефекти: ресемплінг, обрізання тиші, "якість", 8-bit.
pub const SR: u32 = 24000;

pub fn resample(x: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to || x.len() < 2 {
        return x.to_vec();
    }
    let n = ((x.len() as u64 * to as u64) / from as u64).max(2) as usize;
    let k = (x.len() - 1) as f32 / (n - 1) as f32;
    (0..n)
        .map(|i| {
            let p = i as f32 * k;
            let i0 = p as usize;
            let i1 = (i0 + 1).min(x.len() - 1);
            let f = p - i0 as f32;
            x[i0] * (1.0 - f) + x[i1] * f
        })
        .collect()
}

pub fn trim(x: &[f32]) -> &[f32] {
    let a = x.iter().position(|v| v.abs() > 0.01);
    let b = x.iter().rposition(|v| v.abs() > 0.01);
    match (a, b) {
        (Some(a), Some(b)) => &x[a..=b],
        _ => &x[..0],
    }
}

/// "Якість" = ефективна частота дискретизації (8–24 кГц).
pub fn quality(x: &[f32], rate: u32) -> Vec<f32> {
    if rate >= SR {
        x.to_vec()
    } else {
        resample(&resample(x, SR, rate), rate, SR)
    }
}

/// 8-bit: sample&hold + менше біт + м'який кліп.
pub fn pixel(x: &[f32], amt: f32) -> Vec<f32> {
    if amt <= 0.0 {
        return x.to_vec();
    }
    let bits = (16.0 - amt * 13.0).round();
    let hold = 1 + (amt * 9.0) as usize;
    let lv = 2f32.powf(bits - 1.0);
    let drive = 1.0 + amt * 5.0;
    let mut y: Vec<f32> = (0..x.len())
        .map(|i| {
            let q = (x[i / hold * hold] * lv).round() / lv;
            (q * drive).tanh() / drive.tanh()
        })
        .collect();
    let pk = y.iter().fold(1.0f32, |m, v| m.max(v.abs()));
    y.iter_mut().for_each(|v| *v /= pk);
    y
}

pub fn split_sentences(t: &str) -> Vec<String> {
    let (mut out, mut cur) = (vec![], String::new());
    let mut it = t.chars().peekable();
    while let Some(c) = it.next() {
        if c == '\n' {
            out.push(std::mem::take(&mut cur));
            continue;
        }
        cur.push(c);
        if ".!?…".contains(c) && it.peek().map_or(true, |n| n.is_whitespace()) {
            out.push(std::mem::take(&mut cur));
        }
    }
    out.push(cur);
    out.into_iter().map(|s| s.trim().to_string()).filter(|s| s.chars().any(|c| c.is_alphanumeric())).collect()
}
