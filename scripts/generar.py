# -*- coding: utf-8 -*-
import os
import csv
import time
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager


# ============================================================
# CONFIGURACIÓN
# ============================================================
USER = "ckkrisalas@gmail.com"
PASSWORD = os.getenv("MI_PASSWORD")

if not PASSWORD:
    raise ValueError("No se encontró la variable de entorno MI_PASSWORD")

ESTACION_ID = 360019
ELEMENTO_ID = 28
TARGET_YEAR = 2026
TARGET_MONTH = 3   # marzo 2026

THR20 = 20.0
THR15 = 15.0

STAR20_FACE = "gold"
STAR20_EDGE = "black"
STAR20_SIZE = 120

STAR15_FACE = "silver"
STAR15_EDGE = "dimgray"
STAR15_SIZE = 80

DATA_DIR = Path("data")
SITE_DIR = Path("site")
DATA_DIR.mkdir(parents=True, exist_ok=True)
SITE_DIR.mkdir(parents=True, exist_ok=True)

CSV_MENSUAL = DATA_DIR / f"viento_{TARGET_YEAR}_{TARGET_MONTH:02d}.csv"
PNG_OUT = SITE_DIR / f"serie_viento_direccion_{TARGET_YEAR}_{TARGET_MONTH:02d}.png"
HTML_OUT = SITE_DIR / "index.html"


# ============================================================
# 1) DESCARGAR SOLO EL MES OBJETIVO
# ============================================================
def descargar_mes():
    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--window-size=1600,1200")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/114.0.5735.199 Safari/537.36"
    )
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options
    )
    driver.execute_script(
        "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
    )

    try:
        print("Abriendo login MeteoChile...")
        driver.get("https://climatologia.meteochile.gob.cl/application/usuario/loginUsuario")

        email_box = WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.ID, "correo"))
        )
        password_box = driver.find_element(By.ID, "clave")
        acceder_btn = driver.find_element(By.XPATH, "//button[contains(., 'Acceder')]")

        actions = ActionChains(driver)

        actions.move_to_element(email_box).click().perform()
        for char in USER:
            email_box.send_keys(char)
            time.sleep(0.05)

        actions.move_to_element(password_box).click().perform()
        for char in PASSWORD:
            password_box.send_keys(char)
            time.sleep(0.05)

        actions.move_to_element(acceder_btn).pause(0.5).click().perform()
        print("Login enviado.")
        time.sleep(3)

        datos_url = (
            "https://climatologia.meteochile.gob.cl/application/informacion/"
            f"datosMensualesDelElemento/{ESTACION_ID}/{TARGET_YEAR}/{TARGET_MONTH}/{ELEMENTO_ID}"
        )
        print(f"Descargando mes {TARGET_MONTH:02d}-{TARGET_YEAR}...")
        driver.get(datos_url)
        time.sleep(3)

        tabla = driver.find_element(By.XPATH, '//*[@id="excel"]/div/table')
        filas = tabla.find_elements(By.XPATH, "./tbody/tr")
        # Forzar encabezados fijos
        encabezados = ["fecha", "hora", "direccion", "intensidad"]

        datos = []
        for fila in filas:
        celdas = fila.find_elements(By.TAG_NAME, "td")
        if len(celdas) >= 4:
            fila_txt = [cel.text.strip() for cel in celdas[:4]]

            # Evitar una fila de encabezado incrustada dentro del tbody
            fila_norm = [x.strip().lower() for x in fila_txt]
        if fila_norm == ["fecha", "hora", "direccion", "intensidad"]:
            continue

        datos.append(fila_txt)

if not datos:
    raise RuntimeError("No se descargaron datos para el mes solicitado.")

with open(CSV_MENSUAL, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(encabezados)
    writer.writerows(datos)

        print(f"CSV mensual guardado en: {CSV_MENSUAL}")
        print(f"Filas descargadas: {len(datos)}")

    finally:
        driver.quit()
        print("Navegador cerrado.")


# ============================================================
# 2) LEER CSV MENSUAL Y NORMALIZAR
# ============================================================
def cargar_y_preparar():
    if not CSV_MENSUAL.exists():
        raise FileNotFoundError(f"No existe el archivo esperado: {CSV_MENSUAL}")

    df = pd.read_csv(CSV_MENSUAL)
    df.columns = [str(c).strip().lower() for c in df.columns]

    required = ["fecha", "hora", "direccion", "intensidad"]
    faltantes = [c for c in required if c not in df.columns]
    if faltantes:
        raise ValueError(
            f"El CSV no tiene las columnas esperadas {required}. "
            f"Faltan: {faltantes}. "
            f"Columnas encontradas: {list(df.columns)}"
        )

    fecha = pd.to_datetime(df["fecha"], errors="coerce", dayfirst=True)

    hora_txt = df["hora"].astype(str).str.strip()
    hora_dt = pd.to_timedelta(hora_txt, errors="coerce")

    mask_bad = hora_dt.isna()
    if mask_bad.any():
        hora_num = pd.to_numeric(
            hora_txt[mask_bad].str.replace(",", ".", regex=False),
            errors="coerce"
        )
        hora_alt = pd.to_timedelta(hora_num, unit="h")
        hora_dt.loc[mask_bad] = hora_alt

    df["datetime"] = fecha + hora_dt

    df["wind_kt"] = pd.to_numeric(
        df["intensidad"].astype(str).str.replace(",", ".", regex=False),
        errors="coerce"
    )
    df["wind_dir"] = pd.to_numeric(
        df["direccion"].astype(str).str.replace(",", ".", regex=False),
        errors="coerce"
    )

    df = df.dropna(subset=["datetime", "wind_kt"])
    df = df.sort_values("datetime").set_index("datetime")

    df["wind_kt"] = df["wind_kt"].where(df["wind_kt"] <= 30.0, other=np.nan)
    df = df.dropna(subset=["wind_kt"])

    df = df[(df.index.year == TARGET_YEAR) & (df.index.month == TARGET_MONTH)].copy()

    if df.empty:
        raise RuntimeError("El DataFrame quedó vacío después de filtrar marzo 2026.")

    return df


# ============================================================
# 3) GRAFICAR SUBPLOT
# ============================================================
def to_month_ref(idx, month=TARGET_MONTH, year_ref=2000):
    idx = pd.DatetimeIndex(idx)
    return pd.to_datetime({
        "year":   np.full(len(idx), year_ref, dtype=int),
        "month":  np.full(len(idx), month, dtype=int),
        "day":    idx.day.values,
        "hour":   idx.hour.values,
        "minute": idx.minute.values,
        "second": idx.second.values,
    })


def generar_figura(df_plot):
    t_ref = to_month_ref(df_plot.index)
    xmin = pd.Timestamp(2000, TARGET_MONTH, 1)
    xmax = xmin + pd.offsets.MonthBegin(1)

    dir_plot = df_plot["wind_dir"].copy()
    dir_plot[df_plot["wind_kt"] == 0] = np.nan

    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(20, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.06}
    )

    ax1.plot(
        t_ref, df_plot["wind_kt"],
        color="tab:red",
        linewidth=1.5,
        label=str(TARGET_YEAR),
        zorder=3
    )

    ax1.axhline(THR15, color="silver", linestyle="--", linewidth=1.2, label="15 kt")
    ax1.axhline(THR20, color="gold", linestyle="--", linewidth=1.2, label="20 kt")

    daily_max = df_plot["wind_kt"].resample("1D").max()

    for d in daily_max[daily_max >= THR20].index:
        gd = df_plot.loc[d.normalize(): d.normalize() + pd.Timedelta(days=1)]
        if gd.empty:
            continue
        ts_max = gd["wind_kt"].idxmax()
        val_max = gd.loc[ts_max, "wind_kt"]
        ax1.scatter(
            to_month_ref([ts_max]), [val_max],
            marker="*", s=STAR20_SIZE,
            facecolor=STAR20_FACE, edgecolor=STAR20_EDGE,
            linewidth=1.0, zorder=5
        )

    for d in daily_max[(daily_max >= THR15) & (daily_max < THR20)].index:
        gd = df_plot.loc[d.normalize(): d.normalize() + pd.Timedelta(days=1)]
        if gd.empty:
            continue
        ts_max = gd["wind_kt"].idxmax()
        val_max = gd.loc[ts_max, "wind_kt"]
        ax1.scatter(
            to_month_ref([ts_max]), [val_max],
            marker="*", s=STAR15_SIZE,
            facecolor=STAR15_FACE, edgecolor=STAR15_EDGE,
            linewidth=1.0, zorder=4
        )

    n20 = int((daily_max >= THR20).sum())
    n15 = int(((daily_max >= THR15) & (daily_max < THR20)).sum())

    ax1.text(
        0.01, 0.97,
        f"$\\bf{{{TARGET_YEAR}}}$\n★ ≥20 kt: {n20} días\n☆ ≥15 kt: {n15} días",
        transform=ax1.transAxes,
        fontsize=9,
        verticalalignment="top",
        color="tab:red",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="tab:red", alpha=0.85)
    )

    h_line = plt.Line2D([0], [0], color="tab:red", linewidth=1.5, label=str(TARGET_YEAR))
    h_thr15 = plt.Line2D([0], [0], color="silver", linestyle="--", linewidth=1.2, label="15 kt")
    h_thr20 = plt.Line2D([0], [0], color="gold", linestyle="--", linewidth=1.2, label="20 kt")
    h15 = ax1.scatter([], [], marker="*", s=STAR15_SIZE,
                      facecolor=STAR15_FACE, edgecolor=STAR15_EDGE,
                      linewidth=1.0, label="≥15 kt")
    h20 = ax1.scatter([], [], marker="*", s=STAR20_SIZE,
                      facecolor=STAR20_FACE, edgecolor=STAR20_EDGE,
                      linewidth=1.0, label="≥20 kt")

    ax1.legend(handles=[h_line, h_thr15, h_thr20, h15, h20],
               frameon=False, loc="upper right")

    ax1.set_ylabel("Intensidad [kt]")
    ax1.set_title(f"Viento en Carriel Sur — {TARGET_YEAR} (mes {TARGET_MONTH:02d})", fontsize=12)
    ax1.grid(True, which="major", alpha=0.3)
    ax1.grid(True, which="minor", alpha=0.1, linestyle=":")
    ax1.set_xlim(xmin, xmax)
    ax1.xaxis.set_minor_locator(mdates.HourLocator(byhour=[0, 6, 12, 18]))

    sc = ax2.scatter(
        t_ref, dir_plot,
        c=dir_plot, cmap="hsv",
        vmin=0, vmax=360,
        s=18, zorder=3
    )

    for deg in [0, 90, 180, 270, 360]:
        ax2.axhline(deg, color="grey", linestyle=":", linewidth=0.7, alpha=0.6)

    ax2.set_yticks([0, 90, 180, 270, 360])
    ax2.set_yticklabels(["N (0°)", "E (90°)", "S (180°)", "O (270°)", "N (360°)"])
    ax2.set_ylabel("Dirección")
    ax2.set_ylim(-10, 370)
    ax2.grid(True, alpha=0.2)

    cbar = plt.colorbar(sc, ax=ax2, orientation="vertical", pad=0.01, fraction=0.02)
    cbar.set_label("Dir. [°]")
    cbar.set_ticks([0, 90, 180, 270, 360])
    cbar.set_ticklabels(["N", "E", "S", "O", "N"])

    ax2.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d"))
    ax2.xaxis.set_minor_locator(mdates.HourLocator(byhour=[0, 6, 12, 18]))
    ax2.set_xlim(xmin, xmax)
    ax2.set_xlabel(f"Día de mes {TARGET_MONTH:02d}")

    plt.tight_layout()
    plt.savefig(PNG_OUT, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Figura guardada en: {PNG_OUT}")
    return n15, n20


# ============================================================
# 4) CREAR HTML
# ============================================================
def generar_html(n15, n20):
    fecha = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    nombre_png = PNG_OUT.name

    html = f"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <title>Viento Carriel Sur Marzo 2026</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body {{
      font-family: Arial, sans-serif;
      margin: 30px;
      background: #f7f7f7;
      color: #222;
    }}
    .box {{
      max-width: 1200px;
      margin: auto;
      background: white;
      padding: 24px;
      border-radius: 12px;
      box-shadow: 0 2px 10px rgba(0,0,0,0.08);
    }}
    img {{
      max-width: 100%;
      height: auto;
      border: 1px solid #ddd;
    }}
  </style>
</head>
<body>
  <div class="box">
    <h1>Viento en Carriel Sur — Marzo 2026</h1>
    <p><strong>Última actualización:</strong> {fecha}</p>
    <p><strong>Días con ≥15 kt:</strong> {n15} &nbsp;&nbsp; <strong>Días con ≥20 kt:</strong> {n20}</p>
    <img src="{nombre_png}" alt="Serie viento dirección">
    <p><a href="{nombre_png}">Descargar PNG</a></p>
  </div>
</body>
</html>
"""
    HTML_OUT.write_text(html, encoding="utf-8")
    print(f"HTML guardado en: {HTML_OUT}")


# ============================================================
# MAIN
# ============================================================
def main():
    descargar_mes()
    df = cargar_y_preparar()
    n15, n20 = generar_figura(df)
    generar_html(n15, n20)
    print("Proceso terminado OK.")


if __name__ == "__main__":
    main()
