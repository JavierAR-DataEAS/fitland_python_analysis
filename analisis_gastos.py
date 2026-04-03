import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        90,
})

AZUL    = "#3498DB"
VERDE   = "#2ECC71"
ROJO    = "#E74C3C"
NARANJA = "#E67E22"
MORADO  = "#8E44AD"
GRIS    = "#95A5A6"
PALETA  = [AZUL, NARANJA, ROJO, VERDE, MORADO]

# ════════════════════════════════════════════════════════════════════════════
# CARGA Y LIMPIEZA
# ════════════════════════════════════════════════════════════════════════════
g_feb = pd.read_csv(os.path.join(DATA_DIR, "gastros_febrero.csv"))
g_mar = pd.read_csv(os.path.join(DATA_DIR, "gastos_marzo.csv"))
g_feb["Mes"] = "Febrero"; g_mar["Mes"] = "Marzo"

gastos = pd.concat([g_feb, g_mar], ignore_index=True)
gastos.columns = gastos.columns.str.strip()
gastos["Fecha"]    = pd.to_datetime(gastos["Fecha"], format="%m/%d/%Y")
gastos["Tipo"]     = gastos["Tipo"].str.strip()
gastos["Concepto"] = gastos["Concepto"].str.strip()
gastos["Monto"]    = (gastos["Monto"].astype(str)
                      .str.replace(r"[\$,]", "", regex=True).str.strip().astype(float))

ventas = pd.read_csv(os.path.join(DATA_DIR, "ventas_limpio.csv"))
ventas["Fecha"] = pd.to_datetime(ventas["Fecha"])

def gan(mes): return ventas[ventas["Mes"] == mes]["Ganancia"].sum()
def vta(mes): return ventas[ventas["Mes"] == mes]["Total venta"].sum()

gan_feb, gan_mar, gan_total = gan("2026-02"), gan("2026-03"), ventas["Ganancia"].sum()
vta_feb, vta_mar, vta_total = vta("2026-02"), vta("2026-03"), ventas["Total venta"].sum()

# ════════════════════════════════════════════════════════════════════════════
# CLASIFICACION CORRECTA
#   Gastos operativos reales  = Prestamo
#   Reinversion en inventario = Inversion   (capital, no gasto)
#   Distribucion de utilidades= Retiro socios (no es gasto operativo)
# ════════════════════════════════════════════════════════════════════════════
def sumar(mes, tipo):
    m = gastos[(gastos["Mes"] == mes) & (gastos["Tipo"] == tipo)]["Monto"]
    return m.sum() if len(m) else 0.0

op_feb   = sumar("Febrero", "Prestamo")  if sumar("Febrero", "Prestamo") else sumar("Febrero", "Pr\u00e9stamo")
op_mar   = sumar("Marzo",   "Prestamo")  if sumar("Marzo",   "Prestamo") else sumar("Marzo",   "Pr\u00e9stamo")

# Busqueda robusta ignorando tildes y mayusculas
def sumar_tipo(mes, tipo_keyword):
    mask = (gastos["Mes"] == mes) & (gastos["Tipo"].str.lower().str.contains(tipo_keyword.lower()))
    return gastos[mask]["Monto"].sum()

op_feb    = sumar_tipo("Febrero", "stamo")   # Prestamo / Préstamo
op_mar    = sumar_tipo("Marzo",   "stamo")
op_total  = op_feb + op_mar

inv_feb   = sumar_tipo("Febrero", "nversi")  # Inversion / Inversión
inv_mar   = sumar_tipo("Marzo",   "nversi")
inv_total = inv_feb + inv_mar

ret_feb   = sumar_tipo("Febrero", "Retiro")
ret_mar   = sumar_tipo("Marzo",   "Retiro")
ret_total = ret_feb + ret_mar

# Ganancia neta = Ganancia bruta - SOLO gastos operativos (Prestamos)
neta_feb   = gan_feb   - op_feb
neta_mar   = gan_mar   - op_mar
neta_total = gan_total - op_total

# ════════════════════════════════════════════════════════════════════════════
# REPORTE EN CONSOLA
# ════════════════════════════════════════════════════════════════════════════
SEP  = "=" * 60
sep2 = "-" * 60

print(f"\n{SEP}")
print("  ANALISIS FINANCIERO CORREGIDO — FIT LAND  (Feb–Mar 2026)")
print(SEP)

# ── Bloque 1: desglose de movimientos ───────────────────────────────────────
print(f"\n[1] DESGLOSE DE MOVIMIENTOS POR TIPO")
print(sep2)
print(f"  {'Tipo':<28} {'Febrero':>10}  {'Marzo':>10}  {'Total':>10}  Clasificacion")
print(f"  {'-'*28} {'-'*10}  {'-'*10}  {'-'*10}  {'-'*20}")
print(f"  {'Gastos operativos (Prestamo)':<28} ${op_feb:>9,.2f}  ${op_mar:>9,.2f}  ${op_total:>9,.2f}  Gasto real")
print(f"  {'Inversion en inventario':<28} ${inv_feb:>9,.2f}  ${inv_mar:>9,.2f}  ${inv_total:>9,.2f}  Capital (no gasto)")
print(f"  {'Retiro de socios':<28} ${ret_feb:>9,.2f}  ${ret_mar:>9,.2f}  ${ret_total:>9,.2f}  Distribucion utilidades")
print(f"  {'-'*28} {'-'*10}  {'-'*10}  {'-'*10}")
tot_feb = op_feb + inv_feb + ret_feb
tot_mar = op_mar + inv_mar + ret_mar
print(f"  {'TOTAL MOVIMIENTOS':<28} ${tot_feb:>9,.2f}  ${tot_mar:>9,.2f}  ${tot_feb+tot_mar:>9,.2f}")

# ── Bloque 2: comparativa corregida ─────────────────────────────────────────
print(f"\n[2] COMPARATIVA GANANCIA BRUTA vs GASTOS OPERATIVOS")
print(sep2)
print(f"  {'Concepto':<32} {'Febrero':>10}  {'Marzo':>10}  {'Total':>10}")
print(f"  {'-'*32} {'-'*10}  {'-'*10}  {'-'*10}")
print(f"  {'Ingresos totales':<32} ${vta_feb:>9,.2f}  ${vta_mar:>9,.2f}  ${vta_total:>9,.2f}")
print(f"  {'Ganancia bruta':<32} ${gan_feb:>9,.2f}  ${gan_mar:>9,.2f}  ${gan_total:>9,.2f}")
print(f"  {'(-) Gastos operativos (Prestamo)':<32} ${op_feb:>9,.2f}  ${op_mar:>9,.2f}  ${op_total:>9,.2f}")
print(f"  {'GANANCIA NETA REAL':<32} ${neta_feb:>9,.2f}  ${neta_mar:>9,.2f}  ${neta_total:>9,.2f}")
print(f"  {'Margen neto':<32}  {neta_feb/vta_feb*100:>9.1f}%   {neta_mar/vta_mar*100:>9.1f}%   {neta_total/vta_total*100:>9.1f}%")

# ── Bloque 3: reinversion y retiro ──────────────────────────────────────────
print(f"\n[3] REINVERSION Y DISTRIBUCION DE UTILIDADES")
print(sep2)
print(f"  {'Concepto':<32} {'Febrero':>10}  {'Marzo':>10}  {'Total':>10}")
print(f"  {'-'*32} {'-'*10}  {'-'*10}  {'-'*10}")
print(f"  {'Reinversion en inventario':<32} ${inv_feb:>9,.2f}  ${inv_mar:>9,.2f}  ${inv_total:>9,.2f}")
print(f"  {'Retiro de socios':<32} ${ret_feb:>9,.2f}  ${ret_mar:>9,.2f}  ${ret_total:>9,.2f}")
print(f"  {'Ganancia neta disponible':<32} ${neta_feb:>9,.2f}  ${neta_mar:>9,.2f}  ${neta_total:>9,.2f}")
print()

# Analisis de sostenibilidad del retiro
cobertura_feb = neta_feb / ret_feb * 100 if ret_feb else 0
cobertura_mar = neta_mar / ret_mar * 100 if ret_mar else 0
print(f"  Cobertura retiro con ganancia neta:")
print(f"    Febrero: ganancia neta cubre {cobertura_feb:.0f}% del retiro")
print(f"    Marzo  : ganancia neta cubre {cobertura_mar:.0f}% del retiro")
print(f"    (La diferencia se cubre con ingresos totales de ventas)")

print(f"\n{SEP}\n")

# ════════════════════════════════════════════════════════════════════════════
# GRAFICA — 3 paneles
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(13, 8),
                         gridspec_kw={"hspace": 0.5, "wspace": 0.35})
fig.suptitle("Analisis Financiero Corregido — FIT LAND  (Feb–Mar 2026)",
             fontsize=13, fontweight="bold")

ax_pie  = axes[0, 0]   # pastel: clasificacion de movimientos
ax_bar  = axes[0, 1]   # barras: feb vs mar por tipo
ax_neta = axes[1, 0]   # ganancia bruta / neta
ax_cob  = axes[1, 1]   # reinversion y retiro

fmt = plt.FuncFormatter(lambda x, _: f"${x:,.0f}")

# ── Pastel: clasificacion de movimientos totales ─────────────────────────
etiq  = ["Gastos\noperativos\n(Prestamo)", "Reinversion\nen inventario", "Retiro\nde socios"]
vals  = [op_total, inv_total, ret_total]
cols  = [ROJO, AZUL, NARANJA]
exp   = [0.06, 0.03, 0.03]

wedges, _, autotexts = ax_pie.pie(
    vals, autopct="%1.1f%%", startangle=130, explode=exp,
    colors=cols, pctdistance=0.72,
    wedgeprops=dict(linewidth=1.2, edgecolor="white"),
)
for at in autotexts:
    at.set_fontsize(9); at.set_fontweight("bold"); at.set_color("white")

leyenda = [mpatches.Patch(color=cols[i], label=f"{etiq[i].replace(chr(10),' ')}  ${vals[i]:,.0f}")
           for i in range(3)]
ax_pie.legend(handles=leyenda, loc="lower center", bbox_to_anchor=(0.5, -0.28),
              fontsize=8, framealpha=0.5)
ax_pie.set_title("Clasificacion de\nmovimientos totales", fontsize=10, fontweight="bold")

# ── Barras: desglose por mes ─────────────────────────────────────────────
categorias = ["Gastos operativos\n(Prestamo)", "Reinversion\ninventario", "Retiro\nsocios"]
vals_feb_b = [op_feb, inv_feb, ret_feb]
vals_mar_b = [op_mar, inv_mar, ret_mar]
x      = np.arange(len(categorias))
width  = 0.32
cols_b = [ROJO, AZUL, NARANJA]

b_f = ax_bar.bar(x - width/2, vals_feb_b, width, label="Febrero",
                 color=[c + "BB" for c in cols_b], edgecolor="white")
b_m = ax_bar.bar(x + width/2, vals_mar_b, width, label="Marzo",
                 color=cols_b, edgecolor="white")

for b in list(b_f) + list(b_m):
    h = b.get_height()
    if h > 0:
        ax_bar.text(b.get_x() + b.get_width()/2, h + 50,
                    f"${h:,.0f}", ha="center", fontsize=8.5)

ax_bar.set_xticks(x); ax_bar.set_xticklabels(categorias, fontsize=9.5)
ax_bar.yaxis.set_major_formatter(fmt)
ax_bar.set_ylim(0, max(vals_feb_b + vals_mar_b) * 1.22)
ax_bar.set_title("Movimientos por tipo y mes", fontsize=10, fontweight="bold")
ax_bar.legend(fontsize=9, framealpha=0.5)

# ── Panel ganancia: ingresos / ganancia bruta / gastos op / ganancia neta ─
conceptos_n = ["Ingresos\ntotales", "Ganancia\nbruta", "Gastos op.\n(Prestamo)", "Ganancia\nneta real"]
vals_feb_n  = [vta_feb,  gan_feb,  op_feb,  neta_feb]
vals_mar_n  = [vta_mar,  gan_mar,  op_mar,  neta_mar]
cols_n      = [AZUL, VERDE, ROJO, MORADO]

x3 = np.arange(len(conceptos_n)); w3 = 0.32
b_f3 = ax_neta.bar(x3 - w3/2, vals_feb_n, w3, label="Febrero",
                   color=[c + "AA" for c in cols_n], edgecolor="white")
b_m3 = ax_neta.bar(x3 + w3/2, vals_mar_n, w3, label="Marzo",
                   color=cols_n, edgecolor="white")

for b, v in zip(list(b_f3) + list(b_m3), vals_feb_n + vals_mar_n):
    if v > 0:
        ax_neta.text(b.get_x() + b.get_width()/2, v + 80,
                     f"${v:,.0f}", ha="center", fontsize=7.5)

ax_neta.set_xticks(x3); ax_neta.set_xticklabels(conceptos_n, fontsize=9)
ax_neta.yaxis.set_major_formatter(fmt)
ax_neta.set_ylim(0, max(vals_feb_n + vals_mar_n) * 1.2)
ax_neta.set_title(f"Ganancia neta real  (margen {neta_total/vta_total*100:.1f}%)",
                  fontsize=10, fontweight="bold")
ax_neta.legend(fontsize=8.5, framealpha=0.5)

# ── Panel retiro y reinversion ───────────────────────────────────────────
conceptos_c = ["Reinversion\ninventario", "Retiro\nsocios", "Ganancia\nneta disponible"]
vals_feb_c  = [inv_feb, ret_feb, neta_feb]
vals_mar_c  = [inv_mar, ret_mar, neta_mar]
cols_c      = [AZUL, NARANJA, MORADO]

x4 = np.arange(len(conceptos_c)); w4 = 0.32
b_f4 = ax_cob.bar(x4 - w4/2, vals_feb_c, w4, label="Febrero",
                  color=[c + "AA" for c in cols_c], edgecolor="white")
b_m4 = ax_cob.bar(x4 + w4/2, vals_mar_c, w4, label="Marzo",
                  color=cols_c, edgecolor="white")

for b, v in zip(list(b_f4) + list(b_m4), vals_feb_c + vals_mar_c):
    if v > 0:
        ax_cob.text(b.get_x() + b.get_width()/2, v + 80,
                    f"${v:,.0f}", ha="center", fontsize=7.5)

ax_cob.set_xticks(x4); ax_cob.set_xticklabels(conceptos_c, fontsize=9)
ax_cob.yaxis.set_major_formatter(fmt)
ax_cob.set_ylim(0, max(vals_feb_c + vals_mar_c) * 1.2)
ax_cob.set_title("Reinversion y distribucion de utilidades", fontsize=10, fontweight="bold")
ax_cob.legend(fontsize=8.5, framealpha=0.5)

out = os.path.join(DATA_DIR, "6_analisis_gastos.png")
fig.savefig(out, bbox_inches="tight", dpi=90)
plt.close(fig)
print(f"[OK] Grafica guardada: {out}")
