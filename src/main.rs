#![allow(deprecated)]
mod audio;
mod dsp;
mod tts;

use audio::{AudioOut, SCOPE};
use gtk4::prelude::*;
use gtk4::{self as gtk, gdk, gio, glib};
use gtk4_layer_shell::{KeyboardMode, Layer, LayerShell};
use rustfft::{num_complex::Complex, FftPlanner};
use std::{
    cell::RefCell,
    rc::Rc,
    sync::{
        atomic::{AtomicU64, Ordering::SeqCst},
        Arc, Mutex,
    },
    time::Instant,
};
use tts::{Engine, Engines, Params};

const CSS: &str = r#"
window.tts { background: alpha(@theme_bg_color, 0.90); color: @theme_fg_color;
  border: 2px solid @theme_selected_bg_color; border-radius: 0; }
window.tts * { border-radius: 0; box-shadow: none; }
.tts .title { font-weight: 800; font-size: 18px; letter-spacing: 2px; }
.tts .card { background: mix(@theme_bg_color, @theme_selected_bg_color, 0.10);
  border: 1px solid alpha(@theme_fg_color, 0.18); padding: 10px; }
.tts button, .tts dropdown > button { background: mix(@theme_bg_color, @theme_selected_bg_color, 0.18);
  border: 1px solid alpha(@theme_fg_color, 0.22); color: @theme_fg_color; padding: 6px 12px; min-height: 0; }
.tts button:hover { background: mix(@theme_bg_color, @theme_selected_bg_color, 0.38); }
.tts button:checked, .tts button.suggested { background: @theme_selected_bg_color;
  color: @theme_selected_fg_color; border-color: @theme_selected_bg_color; }
.tts textview, .tts textview text { background: alpha(@theme_base_color, 0.75); color: @theme_fg_color; }
.tts scale trough { min-height: 6px; background: alpha(@theme_fg_color, 0.20); }
.tts scale highlight { background: @theme_selected_bg_color; min-height: 6px; }
.tts scale slider { min-width: 14px; min-height: 14px; margin: -5px;
  background: @theme_selected_bg_color; border: 2px solid @theme_bg_color; }
.tts .dim { opacity: 0.65; font-size: 11px; }
"#;

const TEXT_EN: &str = "Hello! This is a test of the overlay.";

fn t(lang: &str, k: &str) -> &'static str {
    let (en, uk) = match k {
        "speak" => ("▶  Speak", "▶  Говорити"),
        "mix" => ("Mix with", "Мікс з"),
        "speed" => ("Speed", "Швидкість"),
        "quality" => ("Quality", "Якість"),
        "pause" => ("Pause", "Пауза"),
        "no_pause" => ("no pause", "без паузи"),
        "off" => ("off", "вимк."),
        "khz" => ("kHz", "кГц"),
        "ms" => ("ms", "мс"),
        "gen" => ("Generating… (first run loads the model)", "Генерація… (перший запуск завантажує модель)"),
        "done" => ("Done", "Готово"),
        "stopped" => ("Stopped", "Зупинено"),
        "error" => ("Error", "Помилка"),
        "nomodels" => ("No models/voices found — check ~/.local/share/tts-overlay", "Немає моделей/голосів — перевір ~/.local/share/tts-overlay"),
        "save" => ("💾  Save WAV", "💾  Зберегти WAV"),
        "saved" => ("Saved", "Збережено"),
        "nothing" => ("Nothing to save yet — press Speak first", "Ще нічого зберігати — спершу натисни «Говорити»"),
        "cuda_ok" => ("CUDA available (GPU, ~1.5 GB VRAM limit)", "CUDA доступна (GPU, ліміт ~1.5 ГБ VRAM)"),
        "cuda_cpu" => ("⚠ CUDA unavailable — running on CPU", "⚠ CUDA недоступна — працює CPU"),
        _ => ("", ""),
    };
    if lang == "uk" { uk } else { en }
}

fn fmt_for(lang: &str, kind: usize) -> Box<dyn Fn(f64) -> String> {
    let l = lang.to_string();
    match kind {
        0 => Box::new(|v| format!("{v:.2}×")),
        1 => Box::new(move |v| format!("{:.0} {}", v / 1000.0, t(&l, "khz"))),
        2 => Box::new(move |v| if v == 0.0 { t(&l, "no_pause").to_string() } else { format!("{v:.0} {}", t(&l, "ms")) }),
        _ => Box::new(move |v| if v == 0.0 { t(&l, "off").to_string() } else { format!("{:.0}%", v * 100.0) }),
    }
}

fn accent(w: &impl IsA<gtk::Widget>) -> (f64, f64, f64) {
    w.style_context()
        .lookup_color("theme_selected_bg_color")
        .map(|c| (c.red() as f64, c.green() as f64, c.blue() as f64))
        .unwrap_or((0.4, 0.6, 1.0))
}

fn selected(dd: &gtk::DropDown) -> Option<String> {
    dd.selected_item().and_downcast::<gtk::StringObject>().map(|s| s.string().to_string())
}

fn set_items(dd: &gtk::DropDown, items: &[String]) {
    let r: Vec<&str> = items.iter().map(|s| s.as_str()).collect();
    dd.set_model(Some(&gtk::StringList::new(&r)));
}

fn slider(parent: &gtk::Box, lo: f64, hi: f64, val: f64, step: f64) -> (gtk::Label, gtk::Scale) {
    let row = gtk::Box::new(gtk::Orientation::Horizontal, 8);
    let l = gtk::Label::new(None);
    l.set_xalign(0.0);
    l.set_width_chars(10);
    row.append(&l);
    let s = gtk::Scale::with_range(gtk::Orientation::Horizontal, lo, hi, step);
    s.set_hexpand(true);
    s.set_value(val);
    s.set_draw_value(true);
    s.set_value_pos(gtk::PositionType::Right);
    row.append(&s);
    parent.append(&row);
    (l, s)
}

struct Catalog {
    kokoro: Vec<String>,
    piper: Vec<tts::PiperEntry>,
}

#[derive(Clone)]
struct W {
    lang: Rc<RefCell<String>>,
    cat: Rc<Catalog>,
    dd_engine: gtk::DropDown,
    dd_voice: gtk::DropDown,
    dd_voice_b: gtk::DropDown,
    s_mix: gtk::Scale,
    mix_row: gtk::Box,
}

impl W {
    fn engine(&self) -> Option<Engine> {
        selected(&self.dd_engine).map(|n| if n.starts_with("Kokoro") { Engine::Kokoro } else { Engine::Piper })
    }

    fn refresh_engines(&self) {
        let mut names = vec![];
        if !self.cat.kokoro.is_empty() {
            names.push("Kokoro".to_string());
        }
        if !self.cat.piper.is_empty() {
            names.push("Piper".to_string());
        }
        set_items(&self.dd_engine, &names);
        self.refresh_voices();
    }

    fn refresh_voices(&self) {
        let kokoro = self.engine() == Some(Engine::Kokoro);
        let list: Vec<String> = if kokoro { self.cat.kokoro.clone() } else { self.cat.piper.iter().map(|p| p.label.clone()).collect() };
        set_items(&self.dd_voice, &list);
        let mut b = vec!["—".to_string()];
        b.extend(list);
        set_items(&self.dd_voice_b, &b);
        self.mix_row.set_visible(kokoro);
    }
}

fn write_wav(path: &std::path::Path, x: &[f32]) -> Result<(), String> {
    let n = (x.len() * 2) as u32;
    let mut b: Vec<u8> = Vec::with_capacity(44 + x.len() * 2);
    b.extend(b"RIFF");
    b.extend((36 + n).to_le_bytes());
    b.extend(b"WAVEfmt ");
    b.extend(16u32.to_le_bytes());
    b.extend(1u16.to_le_bytes()); // PCM
    b.extend(1u16.to_le_bytes()); // mono
    b.extend(dsp::SR.to_le_bytes());
    b.extend((dsp::SR * 2).to_le_bytes());
    b.extend(2u16.to_le_bytes());
    b.extend(16u16.to_le_bytes());
    b.extend(b"data");
    b.extend(n.to_le_bytes());
    for s in x {
        b.extend(((s.clamp(-1.0, 1.0) * 32767.0) as i16).to_le_bytes());
    }
    std::fs::write(path, b).map_err(|e| e.to_string())
}

fn run_job(engines: &Mutex<Engines>, audio: &AudioOut, gen: &AtomicU64, gid: u64, p: &Params, sh: (u32, f32, f32), rec: &Mutex<Vec<f32>>) -> Result<bool, String> {
    let (rate, pause_ms, pix) = sh;
    let mut eng = engines.lock().unwrap();
    for s in dsp::split_sentences(&p.text) {
        if gen.load(SeqCst) != gid {
            return Ok(false);
        }
        let (x, sr) = eng.synth(p, &s)?;
        let x = dsp::resample(&x, sr, dsp::SR);
        let x = dsp::pixel(&dsp::quality(dsp::trim(&x), rate), pix);
        let mut out = x;
        out.extend(std::iter::repeat(0.0).take((dsp::SR as f32 * pause_ms / 1000.0) as usize));
        if gen.load(SeqCst) == gid {
            rec.lock().unwrap().extend_from_slice(&out);
            audio.push(&out);
        }
    }
    Ok(true)
}

fn build(app: &gtk::Application, audio: AudioOut, hidden: bool, use_layer: bool) -> gtk::ApplicationWindow {
    let win = gtk::ApplicationWindow::builder().application(app).title("TTS Overlay").default_width(560).default_height(700).decorated(false).build();
    win.add_css_class("tts");
    if use_layer {
        win.init_layer_shell();
        win.set_layer(Layer::Overlay);
        win.set_keyboard_mode(KeyboardMode::OnDemand);
        win.set_namespace("tts-overlay");
    }

    let cat = Rc::new(Catalog { kokoro: tts::kokoro_voice_names(), piper: tts::piper_catalog() });
    let root = gtk::Box::new(gtk::Orientation::Vertical, 10);
    root.set_margin_top(16);
    root.set_margin_bottom(16);
    root.set_margin_start(16);
    root.set_margin_end(16);

    // шапка: назва + прапори
    let head = gtk::Box::new(gtk::Orientation::Horizontal, 6);
    let title = gtk::Label::new(Some("TTS"));
    title.set_xalign(0.0);
    title.set_hexpand(true);
    title.add_css_class("title");
    let btn_en = gtk::ToggleButton::with_label("🇬🇧");
    btn_en.set_active(true);
    let btn_uk = gtk::ToggleButton::with_label("🇺🇦");
    btn_uk.set_group(Some(&btn_en));
    let close = gtk::Button::with_label("✕");
    {
        let win = win.clone();
        close.connect_clicked(move |_| win.set_visible(false));
    }
    for w in [title.upcast_ref::<gtk::Widget>(), btn_en.upcast_ref(), btn_uk.upcast_ref(), close.upcast_ref()] {
        head.append(w);
    }
    root.append(&head);

    // двигун, голос, текст
    let card = gtk::Box::new(gtk::Orientation::Vertical, 8);
    card.add_css_class("card");
    let r1 = gtk::Box::new(gtk::Orientation::Horizontal, 6);
    let dd_engine = gtk::DropDown::from_strings(&[]);
    let dd_voice = gtk::DropDown::from_strings(&[]);
    dd_engine.set_hexpand(true);
    dd_voice.set_hexpand(true);
    r1.append(&dd_engine);
    r1.append(&dd_voice);
    card.append(&r1);
    let mix_row = gtk::Box::new(gtk::Orientation::Horizontal, 8);
    let ml = gtk::Label::new(None);
    ml.set_width_chars(10);
    ml.set_xalign(0.0);
    let dd_voice_b = gtk::DropDown::from_strings(&[]);
    let s_mix = gtk::Scale::with_range(gtk::Orientation::Horizontal, 0.0, 1.0, 0.05);
    s_mix.set_value(0.5);
    s_mix.set_hexpand(true);
    mix_row.append(&ml);
    mix_row.append(&dd_voice_b);
    mix_row.append(&s_mix);
    card.append(&mix_row);
    let tv = gtk::TextView::new();
    tv.set_wrap_mode(gtk::WrapMode::WordChar);
    tv.buffer().set_text(TEXT_EN);
    let sw = gtk::ScrolledWindow::new();
    sw.set_vexpand(true);
    sw.set_min_content_height(110);
    sw.set_child(Some(&tv));
    card.append(&sw);
    root.append(&card);

    let w = W { lang: Rc::new(RefCell::new("en".into())), cat, dd_engine: dd_engine.clone(), dd_voice, dd_voice_b, s_mix, mix_row };
    {
        let w = w.clone();
        dd_engine.connect_selected_notify(move |_| w.refresh_voices());
    }
    w.refresh_engines();

    // слайдери
    let sl = gtk::Box::new(gtk::Orientation::Vertical, 6);
    sl.add_css_class("card");
    let (l_speed, s_speed) = slider(&sl, 0.5, 2.0, 1.0, 0.05);
    let (l_rate, s_rate) = slider(&sl, 8000.0, 24000.0, 24000.0, 1000.0);
    let (l_pause, s_pause) = slider(&sl, 0.0, 1500.0, 250.0, 50.0);
    let (l_pix, s_pix) = slider(&sl, 0.0, 1.0, 0.0, 0.05);
    let l_pix_text = "8-bit";
    root.append(&sl);

    let cuda_ok = tts::cuda_ok();
    let cuda_lbl = gtk::Label::new(None);
    cuda_lbl.set_xalign(0.0);
    cuda_lbl.add_css_class("dim");
    let status = gtk::Label::new(None);
    status.set_xalign(0.0);
    status.set_wrap(true);
    status.add_css_class("dim");

    // озвучення
    let engines = Arc::new(Mutex::new(Engines::default()));
    let gen = Arc::new(AtomicU64::new(0));
    let rec = Arc::new(Mutex::new(Vec::<f32>::new()));
    let (tx, rx) = async_channel::unbounded::<String>();
    {
        let status = status.clone();
        glib::spawn_future_local(async move {
            while let Ok(s) = rx.recv().await {
                status.set_text(&s);
            }
        });
    }
    let stop = {
        let (gen, audio) = (gen.clone(), audio.clone());
        Rc::new(move || {
            gen.fetch_add(1, SeqCst);
            audio.clear();
        })
    };
    let speak: Rc<dyn Fn()> = {
        let (w, tv, status, stop) = (w.clone(), tv.clone(), status.clone(), stop.clone());
        let (audio, engines, gen, rec) = (audio.clone(), engines.clone(), gen.clone(), rec.clone());
        let (s_speed, s_rate, s_pause, s_pix) = (s_speed.clone(), s_rate.clone(), s_pause.clone(), s_pix.clone());
        Rc::new(move || {
            stop();
            let (Some(engine), Some(voice)) = (w.engine(), selected(&w.dd_voice)) else {
                status.set_text(t(&w.lang.borrow(), "nomodels"));
                return;
            };
            let buf = tv.buffer();
            let vb = selected(&w.dd_voice_b).filter(|v| v != "—" && engine == Engine::Kokoro);
            let p = Params { engine, voice, voice_b: vb, mix: w.s_mix.value() as f32, speed: s_speed.value() as f32, text: buf.text(&buf.start_iter(), &buf.end_iter(), false).to_string() };
            let sh = (s_rate.value() as u32, s_pause.value() as f32, s_pix.value() as f32);
            let gid = gen.load(SeqCst);
            rec.lock().unwrap().clear();
            let lg = w.lang.borrow().clone();
            status.set_text(t(&lg, "gen"));
            let (audio, engines, gen, tx, rec) = (audio.clone(), engines.clone(), gen.clone(), tx.clone(), rec.clone());
            std::thread::spawn(move || {
                let msg = match run_job(&engines, &audio, &gen, gid, &p, sh, &rec) {
                    Ok(true) => t(&lg, "done").to_string(),
                    Ok(false) => t(&lg, "stopped").to_string(),
                    Err(e) => format!("{}: {e}", t(&lg, "error")),
                };
                let _ = tx.send_blocking(msg);
            });
        })
    };
    let btns = gtk::Box::new(gtk::Orientation::Horizontal, 6);
    let go = gtk::Button::with_label("");
    go.set_hexpand(true);
    go.add_css_class("suggested");
    {
        let speak = speak.clone();
        go.connect_clicked(move |_| speak());
    }
    let save_btn = gtk::Button::with_label("");
    let stop_btn = gtk::Button::with_label("■");
    {
        let stop = stop.clone();
        stop_btn.connect_clicked(move |_| stop());
    }
    btns.append(&go);
    btns.append(&stop_btn);
    btns.append(&save_btn);
    {
        let (rec, win, status, w) = (rec.clone(), win.clone(), status.clone(), w.clone());
        save_btn.connect_clicked(move |_| {
            let lg = w.lang.borrow().clone();
            if rec.lock().unwrap().is_empty() {
                status.set_text(t(&lg, "nothing"));
                return;
            }
            let dlg = gtk::FileDialog::new();
            dlg.set_initial_name(Some("tts.wav"));
            let (rec, status) = (rec.clone(), status.clone());
            dlg.save(Some(&win), gio::Cancellable::NONE, move |res| {
                if let Some(path) = res.ok().and_then(|f| f.path()) {
                    status.set_text(&match write_wav(&path, &rec.lock().unwrap()) {
                        Ok(()) => format!("{}: {}", t(&lg, "saved"), path.display()),
                        Err(e) => format!("{}: {e}", t(&lg, "error")),
                    });
                }
            });
        });
    }
    root.append(&btns);

    // мова ІНТЕРФЕЙСУ (голос/модель від неї не залежать)
    let apply_lang: Rc<dyn Fn(&str)> = {
        let (ml, go, save_btn, cuda_lbl) = (ml.clone(), go.clone(), save_btn.clone(), cuda_lbl.clone());
        let (l_speed, l_rate, l_pause, l_pix) = (l_speed.clone(), l_rate.clone(), l_pause.clone(), l_pix.clone());
        let (s_speed, s_rate, s_pause, s_pix) = (s_speed.clone(), s_rate.clone(), s_pause.clone(), s_pix.clone());
        Rc::new(move |lg: &str| {
            ml.set_text(t(lg, "mix"));
            go.set_label(t(lg, "speak"));
            save_btn.set_label(t(lg, "save"));
            cuda_lbl.set_text(t(lg, if cuda_ok { "cuda_ok" } else { "cuda_cpu" }));
            l_speed.set_text(t(lg, "speed"));
            l_rate.set_text(t(lg, "quality"));
            l_pause.set_text(t(lg, "pause"));
            l_pix.set_text(l_pix_text);
            for (i, sc) in [&s_speed, &s_rate, &s_pause, &s_pix].into_iter().enumerate() {
                let f = fmt_for(lg, i);
                sc.set_format_value_func(move |_, v| f(v));
            }
        })
    };
    apply_lang("en");
    for (btn, lang) in [(&btn_en, "en"), (&btn_uk, "uk")] {
        let (w, ap) = (w.clone(), apply_lang.clone());
        btn.connect_toggled(move |b| {
            if b.is_active() {
                *w.lang.borrow_mut() = lang.to_string();
                ap(lang);
            }
        });
    }
    root.append(&status);
    root.append(&cuda_lbl);

    // шари: візуалізатор -> UI -> хвіст курсора
    let overlay = gtk::Overlay::new();
    let vis = gtk::DrawingArea::new();
    let tail = gtk::DrawingArea::new();
    tail.set_can_target(false);
    overlay.set_child(Some(&vis));
    overlay.add_overlay(&root);
    overlay.add_overlay(&tail);
    win.set_child(Some(&overlay));

    {
        let (scope, rate) = (audio.scope.clone(), audio.rate as f32);
        let fft = FftPlanner::<f32>::new().plan_fft_forward(SCOPE);
        let hann: Vec<f32> = (0..SCOPE).map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / SCOPE as f32).cos()).collect();
        let bars = RefCell::new(Vec::<f32>::new());
        vis.set_draw_func(move |a, cr, w, h| {
            let (r, g, b) = accent(a);
            let (cell, gap) = (12, 3);
            let step = cell + gap;
            let n = ((w / step).max(1)) as usize;
            let rows = (h / step).max(1) as f32;
            let mut bars = bars.borrow_mut();
            bars.resize(n, 0.0);
            let sc = scope.lock().unwrap().clone();
            if sc.len() == SCOPE && sc.iter().any(|x| x.abs() > 1e-4) {
                let mut buf: Vec<Complex<f32>> = sc.iter().zip(&hann).map(|(s, w)| Complex::new(s * w, 0.0)).collect();
                fft.process(&mut buf);
                let hz = rate / SCOPE as f32;
                let edge = |i: usize| 60.0 * (8000.0f32 / 60.0).powf(i as f32 / n as f32);
                for i in 0..n {
                    let lo = ((edge(i) / hz) as usize).min(SCOPE / 2 - 2);
                    let hi = ((edge(i + 1) / hz) as usize).clamp(lo + 1, SCOPE / 2);
                    let m = buf[lo..hi].iter().map(|c| c.norm()).sum::<f32>() / (hi - lo) as f32;
                    bars[i] = ((m / 6.0).ln_1p() / 3.0).clamp(0.0, 1.0).max(bars[i] * 0.85);
                }
            } else {
                bars.iter_mut().for_each(|b| *b *= 0.88);
            }
            for (i, v) in bars.iter().enumerate() {
                let k = (v * rows * 0.85) as i32 + 1;
                for j in 0..k {
                    cr.rectangle((i as i32 * step) as f64, (h - (j + 1) * step) as f64, cell as f64, cell as f64);
                }
                cr.set_source_rgba(r, g, b, 0.10 + 0.25 * *v as f64);
                let _ = cr.fill();
            }
        });
    }

    let trail: Rc<RefCell<Vec<(f64, f64, Instant)>>> = Rc::new(RefCell::new(vec![]));
    {
        let trail = trail.clone();
        tail.set_draw_func(move |a, cr, _, _| {
            let (r, g, b) = accent(a);
            let now = Instant::now();
            let life = 0.6;
            trail.borrow_mut().retain(|t| now.duration_since(t.2).as_secs_f64() < life);
            for (x, y, born) in trail.borrow().iter() {
                let k = 1.0 - now.duration_since(*born).as_secs_f64() / life;
                let s = 4.0 + 14.0 * k;
                cr.set_source_rgba(r, g, b, 0.85 * k);
                cr.rectangle(x - s / 2.0, y - s / 2.0, s, s);
                let _ = cr.fill();
            }
        });
    }
    {
        let trail = trail.clone();
        let last = RefCell::new((-99.0f64, -99.0f64));
        let mc = gtk::EventControllerMotion::new();
        mc.set_propagation_phase(gtk::PropagationPhase::Capture);
        mc.connect_motion(move |_, x, y| {
            let mut l = last.borrow_mut();
            if (x - l.0).powi(2) + (y - l.1).powi(2) >= 36.0 {
                trail.borrow_mut().push((x, y, Instant::now()));
                *l = (x, y);
            }
        });
        overlay.add_controller(mc);
    }
    {
        let (vis, tail, trail) = (vis.clone(), tail.clone(), trail.clone());
        win.add_tick_callback(move |_, _| {
            vis.queue_draw();
            if !trail.borrow().is_empty() {
                tail.queue_draw();
            }
            glib::ControlFlow::Continue
        });
    }

    // Esc — сховати, Ctrl+Enter — говорити
    let kc = gtk::EventControllerKey::new();
    kc.set_propagation_phase(gtk::PropagationPhase::Capture);
    {
        let (win, speak) = (win.clone(), speak.clone());
        kc.connect_key_pressed(move |_, key, _, st| {
            if key == gdk::Key::Escape {
                win.set_visible(false);
                return glib::Propagation::Stop;
            }
            if (key == gdk::Key::Return || key == gdk::Key::KP_Enter) && st.contains(gdk::ModifierType::CONTROL_MASK) {
                speak();
                return glib::Propagation::Stop;
            }
            glib::Propagation::Proceed
        });
    }
    win.add_controller(kc);
    win.connect_close_request(|w| {
        w.set_visible(false);
        glib::Propagation::Stop
    });
    if !hidden {
        win.present();
    }
    win
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let overlay = args.iter().any(|a| a == "--overlay"); // layer-shell лише за бажанням

    // gtk4-layer-shell має бути завантажений до libwayland -> LD_PRELOAD і re-exec
    let ls = "/usr/lib/libgtk4-layer-shell.so";
    let pre = std::env::var("LD_PRELOAD").unwrap_or_default();
    if overlay && std::path::Path::new(ls).exists() && !pre.contains(ls) {
        use std::os::unix::process::CommandExt;
        let err = std::process::Command::new(std::env::current_exe().unwrap())
            .args(&args[1..])
            .env("LD_PRELOAD", format!("{ls} {pre}").trim())
            .exec();
        eprintln!("re-exec: {err}");
    }
    if std::env::var_os("ORT_DYLIB_PATH").is_none() {
        std::env::set_var("ORT_DYLIB_PATH", "/usr/lib/libonnxruntime.so");
    }

    let app = gtk::Application::builder().application_id("dev.local.TtsOverlayRs").flags(gio::ApplicationFlags::FLAGS_NONE).build();
    let win: Rc<RefCell<Option<gtk::ApplicationWindow>>> = Rc::new(RefCell::new(None));
    let hidden = args.iter().any(|a| a == "--hidden");
    app.connect_startup(|app| {
        std::mem::forget(app.hold());
        let p = gtk::CssProvider::new();
        p.load_from_data(CSS);
        gtk::style_context_add_provider_for_display(&gdk::Display::default().unwrap(), &p, gtk::STYLE_PROVIDER_PRIORITY_APPLICATION);
    });
    app.connect_activate(move |app| {
        let mut slot = win.borrow_mut();
        match slot.as_ref() {
            None => {
                let (audio, stream) = AudioOut::new().expect("audio output");
                std::mem::forget(stream); // потік cpal живе до виходу
                let layer = overlay && gtk4_layer_shell::is_supported();
                *slot = Some(build(app, audio, hidden, layer));
            }
            Some(w) if w.is_visible() => w.set_visible(false),
            Some(w) => w.present(),
        }
    });
    app.run_with_args::<&str>(&[]);
}
