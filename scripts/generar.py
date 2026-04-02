# -*- coding: utf-8 -*-
import os
import csv
import time
import calendar
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker

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

# ── Umbrales de intensidad [kt] ──────────────────────────────
THR10 = 10.0   # umbral para persistencia sur/norte
THR15 = 15.0
THR20 = 20.0

# ── Sectores angulares sugeridos ─────────────────────────────
# Convención:
#   0°/360° = Norte, 90° = Este, 180° = Sur, 270° = Oeste
SUR_DIR_MIN = 135.0
SUR_DIR_MAX = 225.0

NORTH_DIR_MIN_1 = 315.0
NORTH_DIR_MAX_1 = 360.0
NORTH_DIR_MIN_2 = 0.0
NORTH_DIR_MAX_2 = 45.0

# ── Persistencia ─────────────────────────────────────────────
# Si quieres filtrar eventos muy cortos, súbelo por ejemplo a 3 o 6 h
MIN_EVENT_HOURS = 0.0

# ── Factor de conversión ─────────────────────────────────────
KT_TO_MS = 0.5144   # 1 kt = 0.5144 m/s

# ── Marcadores ───────────────────────────────────────────────
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
    now_local = datetime.now(TZ_LOCAL)
    primero_actual = datetime(now_local.year, now_local.month, 1, tzinfo=TZ_LOCAL)
    ultimo_anterior = primero_actual - timedelta(days=1)
    return [
        (now_local.year, now_local.month),
        (ultimo_anterior.year, ultimo_anterior.month),
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


def to_month_ref(idx, month: int, year_ref: int = 2000):
    idx = pd.DatetimeIndex(idx)
    return pd.to_datetime({
        "year":   np.full(len(idx), year_ref, dtype=int),
        "month":  np.full(len(idx), month,    dtype=int),
        "day":    idx.day.values,
        "hour":   idx.hour.values,
        "minute": idx.minute.values,
        "second": idx.second.values,
    })


def format_horas(h):
    if h is None or np.isnan(h):
        return "0 h"
    if abs(h - round(h)) < 1e-6:
        return f"{int(round(h))} h"
    return f"{h:.1f} h"


def estado_evento_actual(h):
    if h is None or h <= 0:
        return "sin evento actual"
    return f"en curso: {format_horas(h)}"


def infer_step_hours(idx: pd.DatetimeIndex) -> float:
    if len(idx) < 2:
        return 1.0
    diffs_h = pd.Series(idx).diff().dt.total_seconds().dropna() / 3600.0
    diffs_h = diffs_h[diffs_h > 0]
    if diffs_h.empty:
        return 1.0
    return float(diffs_h.median())


def mask_viento_sur_favorable(wind_dir, wind_kt, thr_kt=THR10):
    return (
        np.isfinite(wind_dir) &
        np.isfinite(wind_kt) &
        (wind_kt >= thr_kt) &
        (wind_dir >= SUR_DIR_MIN) &
        (wind_dir <= SUR_DIR_MAX)
    )


def mask_viento_norte_favorable(wind_dir, wind_kt, thr_kt=THR10):
    return (
        np.isfinite(wind_dir) &
        np.isfinite(wind_kt) &
        (wind_kt >= thr_kt) &
        (
            ((wind_dir >= NORTH_DIR_MIN_1) & (wind_dir <= NORTH_DIR_MAX_1)) |
            ((wind_dir >= NORTH_DIR_MIN_2) & (wind_dir <= NORTH_DIR_MAX_2))
        )
    )


def resumir_eventos(mask_bool, idx, step_hours, min_event_hours=0.0):
    """
    Detecta eventos contiguos True en mask_bool.
    Corta evento si aparece un gap temporal mayor a 1.5*step_hours.
    """
    idx = pd.DatetimeIndex(idx)
    mask_bool = np.asarray(mask_bool, dtype=bool)

    eventos_raw = []
    start = None
    last_true_time = None
    npts = 0

    for t, ok in zip(idx, mask_bool):
        if ok:
            if start is None:
                start = t
                last_true_time = t
                npts = 1
            else:
                gap_h = (t - last_true_time).total_seconds() / 3600.0
                if gap_h > 1.5 * step_hours:
                    dur_h = npts * step_hours
                    eventos_raw.append({
                        "start": start,
                        "end": last_true_time,
                        "npts": npts,
                        "duration_h": dur_h
                    })
                    start = t
                    last_true_time = t
                    npts = 1
                else:
                    last_true_time = t
                    npts += 1
        else:
            if start is not None:
                dur_h = npts * step_hours
                eventos_raw.append({
                    "start": start,
                    "end": last_true_time,
                    "npts": npts,
                    "duration_h": dur_h
                })
                start = None
                last_true_time = None
                npts = 0

    if start is not None:
        dur_h = npts * step_hours
        eventos_raw.append({
            "start": start,
            "end": last_true_time,
            "npts": npts,
            "duration_h": dur_h
        })

    eventos = [ev for ev in eventos_raw if ev["duration_h"] >= min_event_hours]

    n_eventos = len(eventos)
    total_h = float(sum(ev["duration_h"] for ev in eventos)) if eventos else 0.0
    max_h = float(max((ev["duration_h"] for ev in eventos), default=0.0))

    current_h = 0.0
    if len(mask_bool) > 0 and mask_bool[-1]:
        if eventos_raw:
            ev_last = eventos_raw[-1]
            if ev_last["end"] == idx[-1] and ev_last["duration_h"] >= min_event_hours:
                current_h = float(ev_last["duration_h"])

    return {
        "events": eventos,
        "n_eventos": n_eventos,
        "total_h": total_h,
        "max_h": max_h,
        "current_h": current_h,
    }


def plot_event_spans(ax, eventos, month, color, alpha=0.10, step_hours=1.0):
    for ev in eventos:
        x0 = to_month_ref([ev["start"]], month=month)[0]
        x1 = to_month_ref([ev["end"] + pd.Timedelta(hours=step_hours)], month=month)[0]
        ax.axvspan(x0, x1, color=color, alpha=alpha, lw=0, zorder=1)


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

    fh = df["fecha"].astype(str) + " " + df["hora"].astype(str)
    dt = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")

    formatos = [
        "%d-%m-%Y %H:%M", "%d-%m-%Y %H:%M:%S",
        "%d/%m/%Y %H:%M", "%d/%m/%Y %H:%M:%S",
        "%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S",
    ]

    for fmt in formatos:
        m = dt.isna()
        if m.any():
            dt.loc[m] = pd.to_datetime(fh.loc[m], format=fmt, errors="coerce")

    m = dt.isna()
    if m.any():
        dt.loc[m] = pd.to_datetime(fh.loc[m], errors="coerce", dayfirst=True)

    dt = pd.DatetimeIndex(dt)
    dt = dt.tz_localize("UTC").tz_convert("America/Santiago")
    df["datetime"] = dt

    df["wind_kt"] = pd.to_numeric(
        df["intensidad"].str.replace(",", ".", regex=False), errors="coerce"
    )
    df["wind_dir"] = pd.to_numeric(
        df["direccion"].str.replace(",", ".", regex=False), errors="coerce"
    )

    df = df.dropna(subset=["datetime", "wind_kt"]).copy()
    df = df.sort_values("datetime").set_index("datetime")

    df["wind_kt"] = df["wind_kt"].where(df["wind_kt"] <= 30.0, other=np.nan)
    df = df.dropna(subset=["wind_kt"])

    df = df[(df.index.year == year) & (df.index.month == month)].copy()
    if df.empty:
        raise RuntimeError(f"El DataFrame quedó vacío en {year}-{month:02d}")

    return df


# ============================================================
# FIGURA
# ============================================================
def generar_figura(df_plot: pd.DataFrame, year: int, month: int):
    t_ref = to_month_ref(df_plot.index, month=month)
    xmin = pd.Timestamp(2000, month, 1)
    xmax = xmin + pd.offsets.MonthBegin(1)

    dir_plot = df_plot["wind_dir"].copy()
    dir_plot[df_plot["wind_kt"] == 0] = np.nan

    # --------------------------------------------------------
    # Persistencia de viento sur/norte favorable
    # --------------------------------------------------------
    step_hours = infer_step_hours(df_plot.index)

    mask_sur = mask_viento_sur_favorable(
        df_plot["wind_dir"].to_numpy(),
        df_plot["wind_kt"].to_numpy(),
        thr_kt=THR10
    )
    mask_norte = mask_viento_norte_favorable(
        df_plot["wind_dir"].to_numpy(),
        df_plot["wind_kt"].to_numpy(),
        thr_kt=THR10
    )

    resumen_sur = resumir_eventos(
        mask_sur, df_plot.index, step_hours, min_event_hours=MIN_EVENT_HOURS
    )
    resumen_norte = resumir_eventos(
        mask_norte, df_plot.index, step_hours, min_event_hours=MIN_EVENT_HOURS
    )

    # --------------------------------------------------------
    # Figura
    # --------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(20, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.06}
    )

    # ── Panel superior: intensidad ───────────────────────────
    plot_event_spans(
        ax1, resumen_sur["events"], month=month,
        color="royalblue", alpha=0.09, step_hours=step_hours
    )
    plot_event_spans(
        ax1, resumen_norte["events"], month=month,
        color="darkorange", alpha=0.10, step_hours=step_hours
    )

    ax1.plot(
        t_ref, df_plot["wind_kt"],
        color="tab:red", linewidth=1.5, zorder=3
    )

    ax1.axhline(
        THR10, color="steelblue", linestyle=(0, (5, 4)),
        linewidth=1.0, alpha=0.8, zorder=2
    )
    ax1.axhline(
        THR15, color="gray", linestyle="--",
        linewidth=1.2, zorder=2
    )
    ax1.axhline(
        THR20, color="#DAA520", linestyle="--",
        linewidth=1.4, zorder=2
    )

    # Máximos diarios y marcadores
    daily_max = df_plot["wind_kt"].resample("1D").max()

    for d in daily_max[daily_max >= THR20].index:
        gd = df_plot.loc[d.normalize(): d.normalize() + pd.Timedelta(days=1)]
        if gd.empty:
            continue
        ts_max = gd["wind_kt"].idxmax()
        val_max = gd.loc[ts_max, "wind_kt"]
        ax1.scatter(
            to_month_ref([ts_max], month=month), [val_max],
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
            to_month_ref([ts_max], month=month), [val_max],
            marker="*", s=STAR15_SIZE,
            facecolor=STAR15_FACE, edgecolor=STAR15_EDGE,
            linewidth=1.0, zorder=4
        )

    # Contadores de días por umbral
    n20 = int((daily_max >= THR20).sum())
    n15 = int(((daily_max >= THR15) & (daily_max < THR20)).sum())
    n10 = int(((daily_max >= THR10) & (daily_max < THR15)).sum())

    texto_stats = (
        f"★ ≥20 kt: {n20} días\n"
        f"☆ ≥15 kt: {n15} días\n"
        f"· ≥10 kt: {n10} días\n\n"
        f"Viento Sur favorable a surgencia ({int(SUR_DIR_MIN)}°–{int(SUR_DIR_MAX)}°, ≥{int(THR10)} kt): "
        f"{resumen_sur['n_eventos']} eventos | {format_horas(resumen_sur['total_h'])} | "
        f"máx {format_horas(resumen_sur['max_h'])} | {estado_evento_actual(resumen_sur['current_h'])}\n"
        f"Viento Norte favorable a convergencia ({int(NORTH_DIR_MIN_1)}°–360° y 0°–{int(NORTH_DIR_MAX_2)}°, ≥{int(THR10)} kt): "
        f"{resumen_norte['n_eventos']} eventos | {format_horas(resumen_norte['total_h'])} | "
        f"máx {format_horas(resumen_norte['max_h'])} | {estado_evento_actual(resumen_norte['current_h'])}"
    )

    ax1.text(
        0.01, 0.97,
        texto_stats,
        transform=ax1.transAxes,
        fontsize=8.6,
        verticalalignment="top",
        color="tab:red",
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="white",
            edgecolor="tab:red",
            alpha=0.88
        )
    )

    ax1.text(
        0.99, 0.03,
        "Sombreado azul: evento Sur favorable a surgencia | "
        "Sombreado naranjo: evento Norte favorable a convergencia",
        transform=ax1.transAxes,
        ha="right", va="bottom",
        fontsize=8,
        color="0.25",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="0.8", alpha=0.75)
    )

    # Eje derecho: m/s
    ymax_kt = max(1.0, ax1.get_ylim()[1])
    ax1_r = ax1.twinx()
    ax1_r.set_ylim(0, ymax_kt * KT_TO_MS)
    ax1_r.set_ylabel("Intensidad [m/s]", labelpad=8)
    ax1_r.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{x:.1f}")
    )
    ax1_r.yaxis.set_minor_locator(ticker.NullLocator())

    # Leyenda superior simplificada
    h_line = plt.Line2D([0], [0], color="tab:red", lw=1.5, label="Intensidad")
    h_thr10 = plt.Line2D([0], [0], color="steelblue", lw=1.0, ls=(0, (5, 4)), label="10 kt")
    h_thr15 = plt.Line2D([0], [0], color="gray", lw=1.2, ls="--", label="15 kt")
    h_thr20 = plt.Line2D([0], [0], color="#DAA520", lw=1.4, ls="--", label="20 kt")
    h15 = ax1.scatter([], [], marker="*", s=STAR15_SIZE,
                      facecolor=STAR15_FACE, edgecolor=STAR15_EDGE, lw=1.0, label="máx diario ≥15 kt")
    h20 = ax1.scatter([], [], marker="*", s=STAR20_SIZE,
                      facecolor=STAR20_FACE, edgecolor=STAR20_EDGE, lw=1.0, label="máx diario ≥20 kt")

    ax1.legend(
        handles=[h_line, h_thr10, h_thr15, h_thr20, h15, h20],
        frameon=False, loc="upper right", fontsize=8
    )

    # Momento actual si cae dentro del mes mostrado
    now_local = pd.Timestamp.now(tz=TZ_LOCAL)
    if (now_local.year == year) and (now_local.month == month):
        t_now = pd.Timestamp(
            year=2000, month=month, day=now_local.day,
            hour=now_local.hour, minute=now_local.minute, second=now_local.second
        )
        ax1.axvline(t_now, color="black", linestyle="--", linewidth=1.0, alpha=0.8, zorder=6)
        ax2.axvline(t_now, color="black", linestyle="--", linewidth=1.0, alpha=0.8, zorder=6)
        ax1.text(
            t_now, ax1.get_ylim()[1] * 0.98, "Ahora",
            rotation=90, va="top", ha="right", fontsize=8, color="black"
        )

    parcial_txt = " (parcial)" if es_mes_parcial(df_plot, year, month) else ""
    ax1.set_title(
        f"Viento en Carriel Sur — {nombre_mes_es(month).capitalize()} {year}{parcial_txt}",
        fontsize=12
    )
    ax1.set_ylabel("Intensidad [kt]")
    ax1.set_xlim(xmin, xmax)
    ax1.set_ylim(bottom=0)
    ax1.grid(True, which="major", alpha=0.3)
    ax1.grid(True, which="minor", alpha=0.1, linestyle=":")
    ax1.tick_params(axis="x", labelbottom=False)

    # Sincronizar eje derecho después de fijar ylim
    ax1_r.set_ylim(0, ax1.get_ylim()[1] * KT_TO_MS)

    # ── Panel inferior: dirección ─────────────────────────────
    ax2.scatter(
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

    # Nota angular en vez de colorbar derecha
    ax2.text(
        0.01, 0.02,
        f"Sector Sur favorable: {int(SUR_DIR_MIN)}°–{int(SUR_DIR_MAX)}°  |  "
        f"Sector Norte favorable: {int(NORTH_DIR_MIN_1)}°–360° y 0°–{int(NORTH_DIR_MAX_2)}°",
        transform=ax2.transAxes,
        ha="left", va="bottom",
        fontsize=8.3, color="0.25",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.8", alpha=0.78)
    )

    # Formato eje X
    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax1.xaxis.set_minor_locator(mdates.HourLocator(byhour=[0, 6, 12, 18]))

    ax2.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d"))
    ax2.xaxis.set_minor_locator(mdates.HourLocator(byhour=[0, 6, 12, 18]))
    ax2.set_xlim(xmin, xmax)
    ax2.set_xlabel(f"Día de {nombre_mes_es(month)} {year}")

    # Guardar
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
        "n10": n10,
        "n15": n15,
        "n20": n20,
        "sur_eventos": resumen_sur["n_eventos"],
        "sur_total_h": format_horas(resumen_sur["total_h"]),
        "sur_max_h": format_horas(resumen_sur["max_h"]),
        "sur_actual": estado_evento_actual(resumen_sur["current_h"]),
        "norte_eventos": resumen_norte["n_eventos"],
        "norte_total_h": format_horas(resumen_norte["total_h"]),
        "norte_max_h": format_horas(resumen_norte["max_h"]),
        "norte_actual": estado_evento_actual(resumen_norte["current_h"]),
        "nrows": len(df_plot),
        "ultima_fecha": fecha_ultima,
        "parcial": es_mes_parcial(df_plot, year, month),
    }


# ============================================================
# HTML
# ============================================================
def generar_html(resumenes):
    ahora_local = datetime.now(TZ_LOCAL)
    fecha_web = ahora_local.strftime("%Y-%m-%d %H:%M %Z")
    version = ahora_local.strftime("%Y%m%d%H%M%S")

    resumenes = sorted(resumenes, key=lambda r: (r["year"], r["month"]), reverse=True)

    cards = []
    for r in resumenes:
        parcial_txt = "parcial" if r["parcial"] else "completo"
        img_url = f'{r["png_name"]}?v={version}'
        cards.append(f"""
        <section class="card">
          <h2>{r["month_name"].capitalize()} {r["year"]}</h2>
          <p class="meta">
            <strong>Estado:</strong> {parcial_txt} &nbsp;|&nbsp;
            <strong>Último dato:</strong> {r["ultima_fecha"]} &nbsp;|&nbsp;
            <strong>≥10 kt:</strong> {r["n10"]} días &nbsp;|&nbsp;
            <strong>≥15 kt:</strong> {r["n15"]} días &nbsp;|&nbsp;
            <strong>≥20 kt:</strong> {r["n20"]} días
          </p>

          <p class="meta">
            <strong>Viento Sur favorable a surgencia:</strong>
            {r["sur_eventos"]} eventos | {r["sur_total_h"]} | máx {r["sur_max_h"]} | {r["sur_actual"]}
            <br>
            <strong>Viento Norte favorable a convergencia:</strong>
            {r["norte_eventos"]} eventos | {r["norte_total_h"]} | máx {r["norte_max_h"]} | {r["norte_actual"]}
          </p>

          <img src="{img_url}" alt="Viento {r["month_name"]} {r["year"]}">
          <p><a href="{img_url}">Descargar PNG</a></p>
        </section>
        """)

    html = f"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <title>Viento Carriel Sur — mes actual y anterior</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
  <meta http-equiv="Pragma" content="no-cache">
  <meta http-equiv="Expires" content="0">
  <style>
    body {{ font-family: Arial, sans-serif; margin: 28px; background: #f5f5f5; color: #222; }}
    .wrap {{ max-width: 1500px; margin: auto; }}
    .top, .card {{
      background: white; border-radius: 14px; padding: 22px;
      box-shadow: 0 2px 12px rgba(0,0,0,0.08); margin-bottom: 18px;
    }}
    img {{ width: 100%; height: auto; border: 1px solid #ddd; border-radius: 8px; }}
    .meta {{ color: #444; margin-bottom: 14px; line-height: 1.5; }}
    a {{ color: #0b57d0; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="top">
      <h1>Viento en Carriel Sur</h1>
      <p><strong>Actualización automática:</strong> {fecha_web}</p>
      <p>Se muestran siempre dos meses: el mes actual y el mes anterior.</p>
      <p>
        Criterios mostrados en la figura:
        Viento Sur favorable a surgencia = 135°–225° y ≥{int(THR10)} kt;
        Viento Norte favorable a convergencia = 315°–360° y 0°–45° y ≥{int(THR10)} kt.
      </p>
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
