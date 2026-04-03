import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os

# ── Configuracion general ────────────────────────────────────────────────────
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
    "figure.dpi": 150,
})

VERDE    = "#2ECC71"
AZUL     = "#3498DB"
NARANJA  = "#E67E22"
ROJO     = "#E74C3C"
PALETA   = ["#3498DB", "#2ECC71", "#E67E22", "#E74C3C",
            "#9B59B6", "#1ABC9C", "#F39C12", "#95A5A6", "#D35400"]

# ── Carga de datos ───────────────────────────────────────────────────────────
df = pd.read_csv(os.path.join(DATA_DIR, "ventas_limpio.csv"))
df["Fecha"] = pd.to_datetime(df["Fecha"])
df["Dia_semana"] = df["Fecha"].dt.day_name()   # en ingles, se traduce abajo
df["Mes_num"]    = df["Fecha"].dt.month

DIAS_ES = {
    "Monday": "Lunes", "Tuesday": "Martes", "Wednesday": "Miercoles",
    "Thursday": "Jueves", "Friday": "Viernes", "Saturday": "Sabado",
    "Sunday": "Domingo",
}
ORDEN_DIAS = ["Lunes", "Martes", "Miercoles", "Jueves",
              "Viernes", "Sabado", "Domingo"]

df["Dia_semana_es"] = df["Dia_semana"].map(DIAS_ES)

fmt_peso = mticker.FuncFormatter(lambda x, _: f"${x:,.0f}")

# ════════════════════════════════════════════════════════════════════════════
# 1. Tendencia de ventas diarias
# ════════════════════════════════════════════════════════════════════════════
ventas_dia = (
    df.groupby("Fecha")["Total venta"]
    .sum()
    .reset_index()
    .sort_values("Fecha")
)

fig, ax = plt.subplots(figsize=(12, 5))

ax.fill_between(ventas_dia["Fecha"], ventas_dia["Total venta"],
                alpha=0.18, color=AZUL)
ax.plot(ventas_dia["Fecha"], ventas_dia["Total venta"],
        color=AZUL, linewidth=2, marker="o", markersize=4, zorder=3)

# Media movil 7 dias
mm7 = ventas_dia.set_index("Fecha")["Total venta"].rolling(7, min_periods=1).mean()
ax.plot(mm7.index, mm7.values, color=NARANJA, linewidth=2,
        linestyle="--", label="Media movil 7 dias", zorder=4)

# Separador feb / mar
sep = pd.Timestamp("2026-03-01")
ax.axvline(sep, color="gray", linestyle=":", linewidth=1.2, alpha=0.7)
ax.text(sep, ax.get_ylim()[1] * 0.97, " Marzo", fontsize=8,
        color="gray", va="top")

ax.yaxis.set_major_formatter(fmt_peso)
ax.set_title("Tendencia de Ventas Diarias — FIT LAND", fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Fecha")
ax.set_ylabel("Total de venta")
ax.legend(framealpha=0.5)
fig.tight_layout()

out1 = os.path.join(DATA_DIR, "1_tendencia_ventas_diarias.png")
fig.savefig(out1, bbox_inches="tight")
plt.close(fig)
print(f"[OK] {out1}")

# ════════════════════════════════════════════════════════════════════════════
# 2. Ganancia por categoria
# ════════════════════════════════════════════════════════════════════════════
ganancia_cat = (
    df.groupby("Categor\u00eda")["Ganancia"]
    .sum()
    .sort_values(ascending=True)
    .reset_index()
)
ganancia_cat.columns = ["Categoria", "Ganancia"]

fig, ax = plt.subplots(figsize=(9, 5))

bars = ax.barh(
    ganancia_cat["Categoria"],
    ganancia_cat["Ganancia"],
    color=PALETA[:len(ganancia_cat)],
    edgecolor="white",
    height=0.6,
)

# Etiquetas al final de cada barra
for bar in bars:
    w = bar.get_width()
    ax.text(w + 50, bar.get_y() + bar.get_height() / 2,
            f"${w:,.0f}", va="center", fontsize=9)

ax.xaxis.set_major_formatter(fmt_peso)
ax.set_title("Ganancia por Categoria de Producto", fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Ganancia total")
ax.set_ylabel("")
ax.set_xlim(0, ganancia_cat["Ganancia"].max() * 1.22)
fig.tight_layout()

out2 = os.path.join(DATA_DIR, "2_ganancia_por_categoria.png")
fig.savefig(out2, bbox_inches="tight")
plt.close(fig)
print(f"[OK] {out2}")

# ════════════════════════════════════════════════════════════════════════════
# 3. Ventas por dia de la semana
# ════════════════════════════════════════════════════════════════════════════
ventas_dia_sem = (
    df.groupby("Dia_semana_es")["Total venta"]
    .sum()
    .reindex(ORDEN_DIAS)
    .fillna(0)
    .reset_index()
)
ventas_dia_sem.columns = ["Dia", "Total venta"]

max_dia = ventas_dia_sem["Total venta"].max()
colores_bar = [ROJO if v == max_dia else AZUL for v in ventas_dia_sem["Total venta"]]

fig, ax = plt.subplots(figsize=(9, 5))

bars = ax.bar(
    ventas_dia_sem["Dia"],
    ventas_dia_sem["Total venta"],
    color=colores_bar,
    edgecolor="white",
    width=0.6,
)

for bar in bars:
    h = bar.get_height()
    if h > 0:
        ax.text(bar.get_x() + bar.get_width() / 2, h + 80,
                f"${h:,.0f}", ha="center", fontsize=8)

ax.yaxis.set_major_formatter(fmt_peso)
ax.set_title("Ventas por Dia de la Semana", fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Dia de la semana")
ax.set_ylabel("Total de venta")
ax.set_ylim(0, max_dia * 1.18)

# Leyenda manual
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=ROJO, label="Dia con mas ventas"),
                   Patch(color=AZUL, label="Otros dias")],
          framealpha=0.5)
fig.tight_layout()

out3 = os.path.join(DATA_DIR, "3_ventas_por_dia_semana.png")
fig.savefig(out3, bbox_inches="tight")
plt.close(fig)
print(f"[OK] {out3}")

# ════════════════════════════════════════════════════════════════════════════
# 4. Comparativa Febrero vs Marzo
# ════════════════════════════════════════════════════════════════════════════
comp = (
    df.groupby("Mes_num")[["Total venta", "Ganancia", "Total Costo"]]
    .sum()
    .rename(index={2: "Febrero", 3: "Marzo"})
)

x = range(len(comp.columns))
width = 0.32
meses = comp.index.tolist()
colores_meses = [AZUL, VERDE]

fig, ax = plt.subplots(figsize=(9, 5))

for i, (mes, color) in enumerate(zip(meses, colores_meses)):
    offset = (i - 0.5) * width
    rects = ax.bar(
        [xi + offset for xi in x],
        comp.loc[mes],
        width=width,
        label=mes,
        color=color,
        edgecolor="white",
    )
    for rect in rects:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, h + 80,
                f"${h:,.0f}", ha="center", fontsize=8, rotation=0)

etiquetas = ["Ingresos totales", "Ganancia neta", "Costo total"]
ax.set_xticks(list(x))
ax.set_xticklabels(etiquetas)
ax.yaxis.set_major_formatter(fmt_peso)
ax.set_title("Comparativa de Ingresos: Febrero vs Marzo", fontsize=14, fontweight="bold", pad=12)
ax.set_ylabel("Monto")
ax.legend(framealpha=0.5)
fig.tight_layout()

out4 = os.path.join(DATA_DIR, "4_comparativa_febrero_marzo.png")
fig.savefig(out4, bbox_inches="tight")
plt.close(fig)
print(f"[OK] {out4}")

print("\nTodas las graficas guardadas correctamente.")
