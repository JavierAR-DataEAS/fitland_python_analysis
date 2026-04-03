import pandas as pd
import numpy as np
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from sklearn.linear_model import LinearRegression
import unicodedata, os

# py/ contiene los scripts; CSV/ contiene los datos; raiz guarda las imagenes
PY_DIR   = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(PY_DIR)
CSV_DIR  = os.path.join(BASE_DIR, "CSV")

def csv(name): return os.path.join(CSV_DIR, name)

def limpiar_texto(s):
    """Elimina caracteres no ASCII problemáticos para matplotlib."""
    if not isinstance(s, str):
        return str(s)
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii").strip()

# ── Paleta y estilo global ───────────────────────────────────────────────────
AZUL    = "#2C6FBF"
VERDE   = "#27AE60"
NARANJA = "#E67E22"
ROJO    = "#E74C3C"
MORADO  = "#8E44AD"
GRIS    = "#7F8C8D"
FONDO   = "#F7F9FC"
PALETA  = [AZUL, VERDE, NARANJA, ROJO, MORADO, "#1ABC9C", "#F39C12"]

plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.25,
    "grid.linestyle":     "--",
    "axes.facecolor":     FONDO,
    "figure.facecolor":   "white",
    "figure.dpi":         100,
})

fmt_peso = mticker.FuncFormatter(lambda x, _: f"${x:,.0f}")

# ════════════════════════════════════════════════════════════════════════════
# CARGA DE DATOS
# ════════════════════════════════════════════════════════════════════════════
ventas = pd.read_csv(csv("ventas_limpio.csv"))
ventas["Fecha"] = pd.to_datetime(ventas["Fecha"])

g_feb = pd.read_csv(csv("gastros_febrero.csv"))
g_mar = pd.read_csv(csv("gastos_marzo.csv"))
g_feb["Mes"] = "Febrero"; g_mar["Mes"] = "Marzo"
gastos = pd.concat([g_feb, g_mar], ignore_index=True)
gastos["Monto"] = (gastos["Monto"].astype(str)
                   .str.replace(r"[\$,]", "", regex=True).str.strip().astype(float))
gastos["Tipo"] = gastos["Tipo"].str.strip()

# ── Series diarias ───────────────────────────────────────────────────────────
vd = (ventas.groupby("Fecha")["Total venta"].sum()
      .reindex(pd.date_range(ventas["Fecha"].min(), ventas["Fecha"].max(), freq="D"), fill_value=0)
      .reset_index().rename(columns={"index": "Fecha", "Total venta": "Ventas"}))
vd["t"] = np.arange(len(vd))

# Regresion lineal
modelo = LinearRegression().fit(vd[["t"]], vd["Ventas"])
vd["RL"] = modelo.predict(vd[["t"]])

# Promedio movil 7d
vd["PM7"] = vd["Ventas"].rolling(7, min_periods=1).mean()

# Futuro abril (30 dias)
t_fut  = np.arange(vd["t"].max()+1, vd["t"].max()+31)
f_fut  = pd.date_range(vd["Fecha"].max() + pd.Timedelta(days=1), periods=30, freq="D")
p_rl   = np.maximum(modelo.predict(t_fut.reshape(-1,1)), 0)
# PM7 extendida
serie_ext = pd.concat([vd[["Fecha","Ventas"]],
                        pd.DataFrame({"Fecha": f_fut, "Ventas": p_rl})]).reset_index(drop=True)
serie_ext["PM7_ext"] = serie_ext["Ventas"].rolling(7, min_periods=1).mean()
p_pm7 = serie_ext.iloc[-30:]["PM7_ext"].values

df_fut = pd.DataFrame({"Fecha": f_fut, "Pred_RL": p_rl, "PM7": p_pm7})
total_rl  = p_rl.sum()
total_pm7 = p_pm7.sum()

# ── Top 5 productos mas rentables ────────────────────────────────────────────
top5 = (ventas.groupby("ID")
        .agg(Ganancia=("Ganancia","sum"), Cantidad=("Cantidad","sum"))
        .nlargest(5, "Ganancia").reset_index())

# Combinar ambos inventarios para lookup completo (case-insensitive)
inv_feb = pd.read_csv(csv("inventario_febrero.csv"))
inv_mar = pd.read_csv(csv("inventario_marzo.csv"))
inv = pd.concat([inv_feb, inv_mar], ignore_index=True)
inv = inv[inv["ID"] != "--"][["ID","Producto","Talla","Color"]].copy()
inv["ID_key"] = inv["ID"].str.lower().str.strip()
inv = inv.drop_duplicates("ID_key")

top5["ID_key"] = top5["ID"].str.lower().str.strip()
top5 = top5.merge(inv[["ID_key","Producto","Talla","Color"]], on="ID_key", how="left").drop(columns="ID_key")

# Limpiar y normalizar nombres para matplotlib
top5["Producto"] = top5["Producto"].fillna("Sin nombre").apply(limpiar_texto)
top5["Talla"]    = top5["Talla"].fillna("?").astype(str).apply(limpiar_texto)
top5["Color"]    = top5["Color"].fillna("?").apply(limpiar_texto)
top5["Etiqueta"] = top5["Producto"] + "\n" + top5["Talla"] + " - " + top5["Color"]

# Imprimir en consola para verificacion
print("=== Top 5 productos mas rentables ===")
for _, r in top5.iterrows():
    print(f"  {r['ID']:<20} | {r['Etiqueta'].replace(chr(10),' | '):<35} | Ganancia: ${r['Ganancia']:,.0f} | {int(r['Cantidad'])} uds")

# ── Gastos por tipo ───────────────────────────────────────────────────────────
por_tipo = gastos.groupby("Tipo")["Monto"].sum().sort_values(ascending=False)

# ════════════════════════════════════════════════════════════════════════════
# FIGURA DASHBOARD
# ════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 11))
fig.patch.set_facecolor("white")

gs = gridspec.GridSpec(
    2, 2,
    figure=fig,
    left=0.07, right=0.97,
    top=0.88,  bottom=0.08,
    hspace=0.52, wspace=0.32,
)

# Encabezado principal
fig.text(0.5, 0.95, "FIT LAND — Dashboard de Ventas",
         ha="center", va="center", fontsize=18, fontweight="bold", color="#2C3E50")
fig.text(0.5, 0.915,
         f"Periodo: Feb–Mar 2026   |   Ingresos: ${ventas['Total venta'].sum():,.0f}"
         f"   |   Ganancia bruta: ${ventas['Ganancia'].sum():,.0f}"
         f"   |   Margen: {ventas['Ganancia'].sum()/ventas['Total venta'].sum()*100:.1f}%",
         ha="center", va="center", fontsize=9.5, color=GRIS)

# ════════════════════════════════════════════════════════════════════════════
# PANEL A — Ventas diarias con tendencia  (arriba izquierda)
# ════════════════════════════════════════════════════════════════════════════
ax_a = fig.add_subplot(gs[0, 0])

ax_a.fill_between(vd["Fecha"], vd["Ventas"], alpha=0.12, color=AZUL)
ax_a.plot(vd["Fecha"], vd["Ventas"],
          color=AZUL, linewidth=1.2, alpha=0.8,
          marker="o", markersize=2.5, label="Ventas reales", zorder=3)
ax_a.plot(vd["Fecha"], vd["RL"],
          color=NARANJA, linewidth=1.8, label="Tendencia (RL)", zorder=4)
ax_a.plot(vd["Fecha"], vd["PM7"],
          color=VERDE, linewidth=1.5, linestyle="--",
          label="Media movil 7d", zorder=4)

# Mejor dia
idx_max = vd["Ventas"].idxmax()
ax_a.annotate(f"Max\n${vd.loc[idx_max,'Ventas']:,.0f}",
              xy=(vd.loc[idx_max,"Fecha"], vd.loc[idx_max,"Ventas"]),
              xytext=(vd.loc[idx_max,"Fecha"] + pd.Timedelta(days=3),
                      vd.loc[idx_max,"Ventas"] * 0.85),
              fontsize=7, color=ROJO, arrowprops=dict(arrowstyle="->", color=ROJO, lw=0.8))

ax_a.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
ax_a.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=2))
plt.setp(ax_a.xaxis.get_majorticklabels(), rotation=25, ha="right", fontsize=7.5)
ax_a.yaxis.set_major_formatter(fmt_peso)
ax_a.tick_params(axis="y", labelsize=7.5)
ax_a.set_title("Ventas Diarias con Tendencia", fontsize=10.5, fontweight="bold", pad=8)
ax_a.legend(fontsize=7, framealpha=0.6, loc="upper right")

# ════════════════════════════════════════════════════════════════════════════
# PANEL B — Top 5 productos mas rentables  (arriba derecha)
# ════════════════════════════════════════════════════════════════════════════
ax_b = fig.add_subplot(gs[0, 1])

colores_top = [PALETA[i] for i in range(len(top5))]
bars = ax_b.barh(top5["Etiqueta"][::-1], top5["Ganancia"][::-1],
                 color=colores_top[::-1], edgecolor="white", height=0.55)

for bar in bars:
    w = bar.get_width()
    ax_b.text(w + 30, bar.get_y() + bar.get_height()/2,
              f"${w:,.0f}", va="center", fontsize=8, fontweight="bold")

ax_b.xaxis.set_major_formatter(fmt_peso)
ax_b.tick_params(axis="x", labelsize=7.5)
ax_b.tick_params(axis="y", labelsize=7.8)
ax_b.set_xlim(0, top5["Ganancia"].max() * 1.28)
ax_b.set_title("Top 5 Productos mas Rentables", fontsize=10.5, fontweight="bold", pad=8)

# Badge de unidades vendidas
for i, (_, row) in enumerate(top5[::-1].iterrows()):
    ax_b.text(top5["Ganancia"].max() * 1.26,
              i,
              f"{int(row['Cantidad'])} uds",
              va="center", ha="right", fontsize=7, color=GRIS,
              style="italic")

# ════════════════════════════════════════════════════════════════════════════
# PANEL C — Gastos por tipo  (abajo izquierda)
# ════════════════════════════════════════════════════════════════════════════
ax_c = fig.add_subplot(gs[1, 0])

colores_g = [ROJO, AZUL, NARANJA]
explode   = [0.04] * len(por_tipo)

wedges, _, autotexts = ax_c.pie(
    por_tipo.values,
    autopct="%1.1f%%",
    startangle=140,
    explode=explode,
    colors=colores_g[:len(por_tipo)],
    pctdistance=0.72,
    wedgeprops=dict(linewidth=1.2, edgecolor="white"),
)
for at in autotexts:
    at.set_fontsize(9)
    at.set_fontweight("bold")
    at.set_color("white")

leyenda = [mpatches.Patch(color=colores_g[i],
                          label=f"{tipo}  ${val:,.0f}")
           for i, (tipo, val) in enumerate(por_tipo.items())]
ax_c.legend(handles=leyenda, loc="lower center",
            bbox_to_anchor=(0.5, -0.18), fontsize=8.5, framealpha=0.5, ncol=1)

# Total en el centro
total_gastos = por_tipo.sum()
ax_c.text(0, 0.05, f"${total_gastos:,.0f}", ha="center", va="center",
          fontsize=11, fontweight="bold", color="#2C3E50")
ax_c.text(0, -0.18, "total gastos", ha="center", va="center",
          fontsize=7.5, color=GRIS)

ax_c.set_title("Distribucion de Gastos por Tipo", fontsize=10.5, fontweight="bold", pad=8)

# ════════════════════════════════════════════════════════════════════════════
# PANEL D — Prediccion abril  (abajo derecha)
# ════════════════════════════════════════════════════════════════════════════
ax_d = fig.add_subplot(gs[1, 1])

# Ultimas 2 semanas de datos reales como contexto
contexto = vd.tail(14)
ax_d.fill_between(contexto["Fecha"], contexto["Ventas"], alpha=0.12, color=AZUL)
ax_d.plot(contexto["Fecha"], contexto["Ventas"],
          color=AZUL, linewidth=1.4, marker="o", markersize=3,
          label="Historico reciente", zorder=3)

# Zona de proyeccion
ax_d.fill_between(df_fut["Fecha"], df_fut["Pred_RL"], alpha=0.13, color=NARANJA)
ax_d.plot(df_fut["Fecha"], df_fut["Pred_RL"],
          color=NARANJA, linewidth=2, linestyle="--",
          label=f"Reg. Lineal  ${total_rl:,.0f}", zorder=5)
ax_d.plot(df_fut["Fecha"], df_fut["PM7"],
          color=VERDE, linewidth=2, linestyle="--",
          label=f"Prom. Movil  ${total_pm7:,.0f}", zorder=5)

# Linea divisoria
sep = vd["Fecha"].max() + pd.Timedelta(hours=12)
ax_d.axvline(sep, color=GRIS, linewidth=1, linestyle=":", alpha=0.8)
ax_d.text(sep + pd.Timedelta(days=0.5),
          max(df_fut["Pred_RL"].max(), df_fut["PM7"].max()) * 0.97,
          "ABRIL", fontsize=8, color=NARANJA, fontweight="bold", va="top")

# Caja con estimados
estimado_txt = (f"Estimado abril\n"
                f"RL:  ${total_rl:,.0f}\n"
                f"PM7: ${total_pm7:,.0f}")
ax_d.text(0.97, 0.96, estimado_txt,
          transform=ax_d.transAxes, fontsize=8,
          va="top", ha="right",
          bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.85, ec=NARANJA, lw=1))

ax_d.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
ax_d.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
plt.setp(ax_d.xaxis.get_majorticklabels(), rotation=25, ha="right", fontsize=7.5)
ax_d.yaxis.set_major_formatter(fmt_peso)
ax_d.tick_params(axis="y", labelsize=7.5)
ax_d.set_title("Proyeccion de Ventas — Abril 2026", fontsize=10.5, fontweight="bold", pad=8)
ax_d.legend(fontsize=7.5, framealpha=0.6, loc="lower right")

# ── Guardar ──────────────────────────────────────────────────────────────────
out = os.path.join(BASE_DIR, "dashboard_fitland.png")
fig.savefig(out, bbox_inches="tight", dpi=100)
plt.close(fig)
print(f"[OK] Dashboard guardado: {out}")
