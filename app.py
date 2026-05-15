import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from skimage import color as skcolor
import re

st.set_page_config(page_title="Paint Color Matcher", page_icon="🎨", layout="centered")

st.markdown('''<style>
  html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }
  .header-banner {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 16px; padding: 24px 20px 20px;
    text-align: center; margin-bottom: 24px;
  }
  .header-banner h1 { color: white; font-size: 1.8rem; margin: 0 0 4px; }
  .header-banner p  { color: rgba(255,255,255,0.85); font-size: 0.9rem; margin: 0; }
  .color-card {
    background: white; border-radius: 14px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.10);
    padding: 16px; margin-bottom: 12px;
    display: flex; align-items: center; gap: 16px;
    border-left: 6px solid #667eea;
  }
  .color-swatch { width:60px; height:60px; border-radius:10px; flex-shrink:0; box-shadow:0 2px 6px rgba(0,0,0,0.15); }
  .color-info h3 { margin: 0 0 4px; font-size: 1.05rem; color: #1e1b4b; }
  .color-info p  { margin: 0; font-size: 0.82rem; color: #6b7280; }
  .badge { display:inline-block; background:#ede9fe; color:#5b21b6; border-radius:20px; padding:2px 10px; font-size:0.75rem; font-weight:600; margin-top:4px; }
  .dom-color-row { display:flex; align-items:center; gap:12px; background:#f3f4f6; border-radius:12px; padding:12px 16px; margin:12px 0 20px; }
  .dom-swatch { width:42px; height:42px; border-radius:8px; flex-shrink:0; box-shadow:0 1px 4px rgba(0,0,0,0.18); }
  .dom-text { font-size:0.9rem; color:#374151; }
  .dom-text b { color:#111827; }
  .section-title { font-size:1rem; font-weight:700; color:#4c1d95; margin:4px 0 12px; border-left:4px solid #7c3aed; padding-left:10px; }
  .rank-1 { border-left-color:#f59e0b; }
  .rank-badge { font-size:0.7rem; background:#fef3c7; color:#92400e; border-radius:20px; padding:1px 8px; font-weight:700; margin-left:8px; }
  [data-testid="stMetricValue"] { font-size: 0.75rem !important; }
  #MainMenu, footer { visibility:hidden; }
</style>''', unsafe_allow_html=True)

@st.cache_data
def load_colors():
    df = pd.read_excel("kode_warna_avian.xlsx")
    def parse_rgb(s):
        nums = re.findall(r"\d+", str(s))
        return tuple(int(x) for x in nums[:3]) if len(nums) >= 3 else None
    df["RGB"] = df["Estimasi RGB"].apply(parse_rgb)
    df = df.dropna(subset=["RGB"])
    rgb_norm = np.array(df["RGB"].tolist(), dtype=np.float32) / 255.0
    lab = skcolor.rgb2lab(rgb_norm.reshape(1, -1, 3)).reshape(-1, 3)
    df["LAB"] = [tuple(lab[i]) for i in range(len(lab))]
    return df

df_colors = load_colors()
lab_array = np.array(df_colors["LAB"].tolist(), dtype=np.float64)

def find_closest(rgb_query, top_n=5):
    q_norm = np.array(rgb_query, dtype=np.float32) / 255.0
    q_lab = skcolor.rgb2lab(q_norm.reshape(1, 1, 3)).reshape(1, 3)
    q_tiled = np.tile(q_lab, (len(lab_array), 1))
    delta_e = skcolor.deltaE_ciede2000(q_tiled, lab_array)
    idx = np.argsort(delta_e)[:top_n]
    return [{"kode": df_colors.iloc[i]["Kode"], "nama": df_colors.iloc[i]["Nama Warna"],
             "rgb": df_colors.iloc[i]["RGB"], "hex": df_colors.iloc[i]["Estimasi Hex"],
             "delta_e": float(delta_e[i])} for i in idx]

def get_dominant_color(img, n_clusters=4):
    img_rgb = np.array(img.convert("RGB").resize((120, 120), Image.LANCZOS), dtype=np.float32) / 255.0
    img_lab = skcolor.rgb2lab(img_rgb).reshape(-1, 3)
    L = img_lab[:, 0]
    mask = (L > 8) & (L < 95)
    pixels = img_lab[mask] if mask.sum() > 50 else img_lab
    np.random.seed(42)
    centers = pixels[np.random.choice(len(pixels), n_clusters, replace=False)]
    labels = np.zeros(len(pixels), dtype=int)
    for _ in range(20):
        dists = np.linalg.norm(pixels[:, None] - centers[None], axis=2)
        labels = np.argmin(dists, axis=1)
        new_centers = np.array([
            pixels[labels == k].mean(axis=0) if (labels == k).sum() > 0 else centers[k]
            for k in range(n_clusters)
        ])
        if np.allclose(centers, new_centers, atol=0.5): break
        centers = new_centers
    dominant_lab = centers[np.argmax(np.bincount(labels, minlength=n_clusters))]
    rgb = skcolor.lab2rgb(dominant_lab.reshape(1, 1, 3)).reshape(3)
    return tuple(int(np.clip(v * 255, 0, 255)) for v in rgb)

def rgb_to_hex(r, g, b): return f"#{r:02X}{g:02X}{b:02X}"

def delta_e_to_sim(de):
    return max(0.0, 100.0 - de * 2.0)

def color_card_html(result, rank=None):
    r, g, b = result["rgb"]
    de = result["delta_e"]
    sim = delta_e_to_sim(de)
    badge = '<span class="rank-badge">⭐ Terbaik</span>' if rank == 1 else ''
    return f'''
    <div class="color-card {'rank-1' if rank==1 else ''}">
      <div class="color-swatch" style="background:{result['hex']};"></div>
      <div class="color-info">
        <h3>{result['nama']} {badge}</h3>
        <p>Kode: <b>{result['kode']}</b> &nbsp;|&nbsp; Hex: <b>{result['hex']}</b></p>
        <p>RGB: ({r}, {g}, {b})</p>
        <span class="badge">ΔE = {de:.2f} &nbsp;|&nbsp; Kemiripan {sim:.1f}%</span>
      </div>
    </div>'''

st.markdown('''<div class="header-banner"><h1>🎨 Paint Color Matcher</h1><p>Upload foto warna → temukan warna yang paling cocok</p></div>''', unsafe_allow_html=True)
c1,c2,c3 = st.columns(3)
c1.metric("Total Warna", f"{len(df_colors):,}")
c2.metric("Metode", "Delta E CIEDE2000")
c3.metric("Brand", "Avian Brands")
st.divider()

st.markdown('<div class="section-title">📸 Input Foto</div>', unsafe_allow_html=True)
st.caption("Upload dari galeri atau foto langsung lewat kamera.")

tab1, tab2 = st.tabs(["📁 Upload dari Galeri", "📷 Ambil Foto"])
img = None
with tab1:
    uploaded = st.file_uploader("Pilih gambar", type=["jpg","jpeg","png","webp"], label_visibility="collapsed")
    if uploaded:
        img = Image.open(uploaded)
with tab2:
    camera = st.camera_input("Ambil foto", label_visibility="collapsed")
    if camera:
        img = Image.open(camera)

if img:
    st.image(img, caption="Gambar yang diinput", use_container_width=True)
    img_crop = img
    with st.expander("✂️ Fokus ke area tertentu (opsional)", expanded=False):
        st.caption("Geser slider untuk memotong area gambar sebelum analisis.")
        w, h = img.size
        l_pct = st.slider("Kiri (%)", 0, 45, 10, key="l")
        r_pct = st.slider("Kanan (%)", 55, 100, 90, key="r")
        t_pct = st.slider("Atas (%)", 0, 45, 10, key="t")
        b_pct = st.slider("Bawah (%)", 55, 100, 90, key="b")
        img_crop = img.crop((int(w*l_pct/100), int(h*t_pct/100), int(w*r_pct/100), int(h*b_pct/100)))
        st.image(img_crop, caption="Area yang dipilih", use_container_width=True)
    top_n = st.select_slider("Tampilkan berapa warna terdekat?", options=[3,5,8,10], value=5)
    if st.button("🔍 Analisis Warna", type="primary", use_container_width=True):
        with st.spinner("Menganalisis warna..."):
            dom = get_dominant_color(img_crop)
            r, g, b = dom
            results = find_closest(dom, top_n=top_n)
        st.markdown('<div class="section-title">🎯 Warna Dominan Terdeteksi</div>', unsafe_allow_html=True)
        st.markdown(f'''<div class="dom-color-row"><div class="dom-swatch" style="background:{rgb_to_hex(r,g,b)};"></div><div class="dom-text"><b>RGB ({r}, {g}, {b})</b><br>{rgb_to_hex(r,g,b)}</div></div>''', unsafe_allow_html=True)
        st.markdown(f'<div class="section-title">🏆 {top_n} Warna Avian Paling Cocok</div>', unsafe_allow_html=True)
        for i, res in enumerate(results, 1):
            st.markdown(color_card_html(res, rank=i), unsafe_allow_html=True)
        df_out = pd.DataFrame([{"Rank":i+1,"Kode":r["kode"],"Nama Warna":r["nama"],"Hex":r["hex"],"RGB":str(r["rgb"]),"Delta E":f"{r['delta_e']:.2f}","Kemiripan (%)":f"{delta_e_to_sim(r['delta_e']):.1f}"} for i,r in enumerate(results)])
        st.download_button("⬇️ Download Hasil (.csv)", df_out.to_csv(index=False).encode(), "hasil_warna.csv", "text/csv", use_container_width=True)
else:
    st.markdown('''<div style="text-align:center;color:#9ca3af;padding:40px 20px;"><div style="font-size:3rem;">📷</div><p>Upload foto atau ambil gambar untuk memulai deteksi warna</p></div>''', unsafe_allow_html=True)
    with st.expander("ℹ️ Cara Penggunaan"):
        st.markdown("1. Pilih tab **Upload dari Galeri** atau **Ambil Foto**\n2. *(Opsional)* Gunakan **crop** untuk fokus area\n3. Pilih jumlah hasil\n4. Klik **Analisis Warna**!")
    with st.expander("🔬 Cara Kerja Algoritma"):
        st.markdown("- **K-Means Clustering** di ruang LAB untuk ekstraksi warna dominan\n- **Konversi LAB color space** untuk representasi persepsi manusia\n- **Delta E CIEDE2000** untuk pencocokan warna paling akurat secara persepsi")
st.divider()
st.caption("🎨 Avian Color Matcher · Tugas Besar Pengolahan Citra · Data: Avian Brands")