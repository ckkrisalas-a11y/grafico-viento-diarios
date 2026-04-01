# -*- coding: utf-8 -*-
import os
import csv
import time
import calendar
from pathlib import Path
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

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

THR20 = 20.0
THR15 = 15.0

STAR20_FACE = "gold"
STAR20_EDGE = "black"
STAR20_SIZE = 120

STAR15_FACE = "silver"
STAR15_EDGE = "dimgray"
STAR15_SIZE = 80

TZ_LOCAL = ZoneInfo("America/Santiago")

DATA_DIR = Path("data")
SITE_DIR = Path("site")
DATA_DIR.mkdir(parents=True, exist_ok=True)
SITE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# UTILIDADES GENERALES
# ============================================================
def nombre_mes_es(month: int) -> str:
    meses = [
        "", "enero", "febrero", "marzo", "abril", "mayo", "junio",
        "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"
    ]
    return meses[month]


def meses_objetivo():
    """
    Devuelve [(anio_mes_anterior), (anio_mes_actual)] usando hora de Chile.
    """
    now_local = datetime.now(TZ_LOCAL)
    primero_actual = datetime(now_local.year, now_local.month, 1, tzinfo=TZ_LOCAL)
    ultimo_anterior = primero_actual - timedelta(days=1)

    return [
        (ultimo_anterior.year, ultimo_anterior.month),
        (now_local.year, now_local.month),
    ]


def csv_path(year: int, month: int) -> Path:
    return DATA_DIR / f"viento_{year}_{month:02d}.csv"


def png_path(year: int, month: int) -> Path:
    return SITE_DIR / f"serie_viento_direccion_{year}_{month:02d}.png"


def es_mes_parcial(df_plot: pd.DataFrame, year: int, month: int) -> bool:
    if df_plot.empty:
        return False
    ultimo_dia_con_datos = int(df_plot.index.max().day)
    ultimo_dia_mes = calendar.monthrange(year, month)[1]
    return ultimo_dia_con_datos < ultimo_dia_mes


# ============================================================
# SELENIUM
# ============================================================
def crear_driver():
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
    return driver


def login_meteochile(driver):
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
        time.sleep(0.03)

    actions.move_to_element(password_box).click().perform()
    for char in PASSWORD:
        password_box.send_keys(char)
        time.sleep(0.03)

    actions.move_to_element(acceder_btn).pause(0.4).click().perform()
    print("Login enviado.")
    time.sleep(3)


def descargar_mes(driver, year: int, month: int):
    salida = csv_path(year, month)

    datos_url = (
        "https://climatologia.meteochile.gob.cl/application/informacion/"
        f"datosMensualesDelElemento/{ESTACION_ID}/{year}/{month}/{ELEMENTO_ID}"
    )
    print(f"Descargando {year}-{month:02d} ...")
    driver.get(datos_url)
    time.sleep(3)

    tabla = driver.find_element(By.XPATH, '//*[@id="excel"]/div/table')
    filas = tabla.find_elements(By.XPATH, ".//tbody/tr")

    encabezados = ["fecha", "hora", "direccion", "intensidad"]
    datos = []

    for fila in filas:
        celdas = fila.find_elements(By.TAG_NAME, "td")
        if len(celdas) >= 4:
            fila_txt = [cel.text.strip() for cel in celdas[:4]]

            if not any(fila_txt):
                continue

            fila_norm = [x.strip().lower() for x in fila_txt]
            if fila_norm == encabezados:
                continue

            datos.append(fila_txt)

    if not datos:
        raise RuntimeError(f"No se descargaron datos para {year}-{month:02d}")

    with open(salida, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(encabezados)
        writer.writerows(datos)

    print(f"CSV guardado en: {salida}")
    print(f"Filas descargadas en {year}-{month:02d}: {len(datos)}")


def descargar_meses(meses):
    driver = crear_driver()
    try:
        login_meteochile(driver)
        for year, month in meses:
            descargar_mes(driver, year, month)
    finally:
        driver.quit()
        print("Navegador cerrado.")


# ============================================================
# LECTURA / PREPARACIÓN
# ============================================================
def cargar_y_preparar(year: int, month: int) -> pd.DataFrame:
    archivo = csv_path(year, month)
    if not archivo.exists():
        raise FileNotFoundError(f"No existe el archivo esperado: {archivo}")

    expected = ["fecha", "hora", "direccion", "intensidad"]

    df = pd.read_csv(archivo, dtype=str)
    df.columns = [str(c).strip().lower() for c in df.columns]

    if list(df.columns) != expected:
        if len(df.columns) == 4:
            df = pd.read_csv(archivo, dtype=str, header=None, names=expected)
        else:
            raise ValueError(
                f"El CSV no tiene las columnas esperadas {expected}. "
                f"Columnas encontradas: {list(df.columns)}"
            )

    for c in expected:
        df[c] = df[c].astype(str).str.strip()

    df = df.replace({"": np.nan})
    df = df.dropna(subset=expected, how="all").copy()

    mask_header = (
        df["fecha"].str.lower().eq("fecha") &
        df["hora"].str.lower().eq("hora") &
        df["direccion"].str.lower().eq("direccion") &
        df["intensidad"].str.lower().eq("intensidad")
    )
    df = df.loc[~mask_header].copy()

    # Parseo robusto de fecha+hora como UTC
    fh = df["fecha"].astype(str) + " " + df["hora"].astype(str)
    dt = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")

    formatos = [
        "%d-%m-%Y %H:%M",
        "%d-%m-%Y %H:%M:%S",
        "%d/%m/%Y %H:%M",
        "%d/%m/%Y %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d %H:%M:%S",
    ]

    for fmt in formatos:
        m = dt.isna()
        if m.any():
            dt.loc[m] = pd.to_datetime(fh.loc[m], format=fmt, errors="coerce")

    m = dt.isna()
    if m.any():
        dt.loc[m] = pd.to_datetime(fh.loc[m], errors="coerce", dayfirst=True)

    # IMPORTANTE:
    # Interpretamos la hora descargada como UTC y luego la convertimos a Chile
    dt = pd.DatetimeIndex(dt)
    dt = dt.tz_localize("UTC").tz_convert("America/Santiago")

    df["datetime"] = dt

    df["wind_kt"] = pd.to_numeric(
        df["intensidad"].str.replace(",", ".", regex=False),
        errors="coerce"
    )
    df["wind_dir"] = pd.to_numeric(
        df["direccion"].str.replace(",", ".", regex=False),
        errors="coerce"
    )

    df = df.dropna(subset=["datetime", "wind_kt"]).copy()
    df = df.sort_values("datetime").set_index("datetime")

    df["wind_kt"] = df["wind_kt"].where(df["wind_kt"] <= 30.0, other=np.nan)
    df = df.dropna(subset=["wind_kt"])

    # Filtrar por mes local de Chile
    df = df[(df.index.year == year) & (df.index.month == month)].copy()

    if df.empty:
        raise RuntimeError(f"El DataFrame quedó vacío en {year}-{month:02d}")

    return df


# ============================================================
# FIGURA
# ============================================================
def to_month_ref(idx, month: int, year_ref: int = 2000):
    idx = pd.DatetimeIndex(idx)
    return pd.to_datetime({
        "year": np.full(len(idx), year_ref, dtype=int),
        "month": np.full(len(idx), month, dtype=int),
        "day": idx.day.values,
        "hour": idx.hour.values,
        "minute": idx.minute.values,
        "second": idx.second.values,
    })


def generar_figura(df_plot: pd.DataFrame, year: int, month: int):
    t_ref = to_month_ref(df_plot.index, month=month)
    xmin = pd.Timestamp(2000, month, 1)
    xmax = xmin + pd.offsets.MonthBegin(1)

    dir_plot = df_plot["wind_dir"].copy()
    dir_plot[df_plot["wind_kt"] == 0] = np.nan

    # GridSpec para que ambos paneles tengan exactamente el mismo ancho
    fig = plt.figure(figsize=(20, 8))
    gs = fig.add_gridspec(
        2, 2,
        height_ratios=[1, 1],
        width_ratios=[50, 1.2],
        hspace=0.06,
        wspace=0.03
    )

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    cax = fig.add_subplot(gs[1, 1])

    # ── Panel superior: Intensidad ───────────────────────────
    ax1.plot(
        t_ref, df_plot["wind_kt"],
        color="tab:red",
        linewidth=1.5,
        label=str(year),
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
            to_month_ref([ts_max], month=month),
            [val_max],
            marker="*",
            s=STAR20_SIZE,
            facecolor=STAR20_FACE,
            edgecolor=STAR20_EDGE,
            linewidth=1.0,
            zorder=5
        )

    for d in daily_max[(daily_max >= THR15) & (daily_max < THR20)].index:
        gd = df_plot.loc[d.normalize(): d.normalize() + pd.Timedelta(days=1)]
        if gd.empty:
            continue
        ts_max = gd["wind_kt"].idxmax()
        val_max = gd.loc[ts_max, "wind_kt"]
        ax1.scatter(
            to_month_ref([ts_max], month=month),
            [val_max],
            marker="*",
            s=STAR15_SIZE,
            facecolor=STAR15_FACE,
            edgecolor=STAR15_EDGE,
            linewidth=1.0,
            zorder=4
        )

    n20 = int((daily_max >= THR20).sum())
    n15 = int(((daily_max >= THR15) & (daily_max < THR20)).sum())

    ax1.text(
        0.01, 0.97,
        f"$\\bf{{{year}}}$\n★ ≥20 kt: {n20} días\n☆ ≥15 kt: {n15} días",
        transform=ax1.transAxes,
        fontsize=9,
        verticalalignment="top",
        color="tab:red",
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            edgecolor="tab:red",
            alpha=0.85
        )
    )

    h_line = plt.Line2D([0], [0], color="tab:red", linewidth=1.5, label=str(year))
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

    parcial_txt = " (parcial)" if es_mes_parcial(df_plot, year, month) else ""
    ax1.set_title(
        f"Viento en Carriel Sur — {nombre_mes_es(month).capitalize()} {year}{parcial_txt}",
        fontsize=12
    )
    ax1.set_ylabel("Intensidad [kt]")
    ax1.set_xlim(xmin, xmax)
    ax1.grid(True, which="major", alpha=0.3)
    ax1.grid(True, which="minor", alpha=0.1, linestyle=":")
    ax1.tick_params(axis="x", labelbottom=False)

    # ── Panel inferior: Dirección ───────────────────────────
    sc = ax2.scatter(
        t_ref, dir_plot,
        c=dir_plot, cmap="hsv",
        vmin=0, vmax=360,
        s=22, zorder=3
    )

    for deg in [0, 90, 180, 270, 360]:
        ax2.axhline(deg, color="grey", linestyle=":", linewidth=0.7, alpha=0.6)

    ax2.set_yticks([0, 90, 180, 270, 360])
    ax2.set_yticklabels(["N (0°)", "E (90°)", "S (180°)", "O (270°)", "N (360°)"])
    ax2.set_ylabel("Dirección")
    ax2.set_ylim(-10, 370)
    ax2.grid(True, which="major", alpha=0.2)
    ax2.grid(True, which="minor", alpha=0.1, linestyle=":")

    cbar = fig.colorbar(sc, cax=cax)
    cbar.set_label("Dir. [°]")
    cbar.set_ticks([0, 90, 180, 270, 360])
    cbar.set_ticklabels(["N", "E", "S", "O", "N"])

    # Mismo eje x en ambos paneles
    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax1.xaxis.set_minor_locator(mdates.HourLocator(byhour=[0, 6, 12, 18]))

    ax2.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d"))
    ax2.xaxis.set_minor_locator(mdates.HourLocator(byhour=[0, 6, 12, 18]))
    ax2.set_xlim(xmin, xmax)
    ax2.set_xlabel(f"Día de {nombre_mes_es(month)}")

    out_png = png_path(year, month)
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Figura guardada en: {out_png}")

    fecha_ultima = df_plot.index.max().strftime("%d-%m-%Y %H:%M")
    return {
        "year": year,
        "month": month,
        "month_name": nombre_mes_es(month),
        "png_name": out_png.name,
        "n15": n15,
        "n20": n20,
        "nrows": len(df_plot),
        "ultima_fecha": fecha_ultima,
        "parcial": es_mes_parcial(df_plot, year, month),
    }


# ============================================================
# HTML
# ============================================================
def generar_html(resumenes):
    fecha_web = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    cards = []
    for r in resumenes:
        parcial_txt = "parcial" if r["parcial"] else "completo"
        cards.append(f"""
        <section class="card">
          <h2>{r["month_name"].capitalize()} {r["year"]}</h2>
          <p class="meta">
            <strong>Estado:</strong> {parcial_txt} &nbsp;|&nbsp;
            <strong>Último dato:</strong> {r["ultima_fecha"]} &nbsp;|&nbsp;
            <strong>≥15 kt:</strong> {r["n15"]} días &nbsp;|&nbsp;
            <strong>≥20 kt:</strong> {r["n20"]} días
          </p>
          <img src="{r["png_name"]}" alt="Viento {r["month_name"]} {r["year"]}">
          <p><a href="{r["png_name"]}">Descargar PNG</a></p>
        </section>
        """)

    html = f"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <title>Viento Carriel Sur — mes actual y anterior</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body {{
      font-family: Arial, sans-serif;
      margin: 28px;
      background: #f5f5f5;
      color: #222;
    }}
    .wrap {{
      max-width: 1500px;
      margin: auto;
    }}
    .top {{
      background: white;
      border-radius: 14px;
      padding: 22px;
      box-shadow: 0 2px 12px rgba(0,0,0,0.08);
      margin-bottom: 18px;
    }}
    .card {{
      background: white;
      border-radius: 14px;
      padding: 22px;
      box-shadow: 0 2px 12px rgba(0,0,0,0.08);
      margin-bottom: 18px;
    }}
    img {{
      width: 100%;
      height: auto;
      border: 1px solid #ddd;
      border-radius: 8px;
    }}
    .meta {{
      color: #444;
      margin-bottom: 14px;
    }}
    a {{
      color: #0b57d0;
      text-decoration: none;
    }}
    a:hover {{
      text-decoration: underline;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="top">
      <h1>Viento en Carriel Sur</h1>
      <p><strong>Actualización automática:</strong> {fecha_web}</p>
      <p>Se muestran siempre dos meses: el mes actual y el mes anterior.</p>
    </div>
    {''.join(cards)}
  </div>
</body>
</html>
"""
    (SITE_DIR / "index.html").write_text(html, encoding="utf-8")
    print(f"HTML guardado en: {SITE_DIR / 'index.html'}")


# ============================================================
# MAIN
# ============================================================
def main():
    targets = meses_objetivo()
    print("Meses objetivo:", targets)

    # Descarga en una sola sesión
    descargar_meses(targets)

    resumenes = []
    for year, month in targets:
        df = cargar_y_preparar(year, month)
        resumen = generar_figura(df, year, month)
        resumenes.append(resumen)

    generar_html(resumenes)
    print("Proceso terminado OK.")


if __name__ == "__main__":
    main()
