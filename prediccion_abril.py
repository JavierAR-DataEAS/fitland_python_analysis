import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Estilo global ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "grid.linestyle":     "--",
    "figure.dpi":         150,
})

AZUL    = "#2C6FBF"
VERDE   = "#27AE60"
NARANJA = "#E67E22"
ROJO    = "#E74C3C"
GRIS    = "#95A5A6"

# ════════════════════════════════════════════════════════════════════════════
# 1. Carga y agrupacion diaria
# ════════════════════════════════════════════════════════════════════════════
df = pd.read_csv(os.path.join(DATA_DIR, "ventas_limpio.csv"))
df["Fecha"] = pd.to_datetime(df["Fecha"])

ventas_dia = (
    df.groupby("Fecha")["Total venta"]
    .sum()
    .reset_index()
    .rename(columns={"Total venta": "Ventas"})
    .sort_values("Fecha")
)

# Rellenar dias sin ventas con 0 para series completa
rango_completo = pd.date_range(ventas_dia["Fecha"].min(), ventas_dia["Fecha"].max(), freq="D")
ventas_dia = (
    ventas_dia.set_index("Fecha")
    .reindex(rango_completo, fill_value=0)
    .rename_axis("Fecha")
    .reset_index()
)

n = len(ventas_dia)
ventas_dia["t"] = np.arange(n)   # variable numerica para regresion

print("=" * 58)
print("  MODELO PREDICTIVO DE VENTAS — FIT LAND")
print("=" * 58)
print(f"\n  Dias con datos reales : {n}")
print(f"  Rango                 : {ventas_dia['Fecha'].min().date()} a "
      f"{ventas_dia['Fecha'].max().date()}")
print(f"  Venta diaria promedio : ${ventas_dia['Ventas'].mean():,.2f}")
print(f"  Maximo en un dia      : ${ventas_dia['Ventas'].max():,.2f}  "
      f"({ventas_dia.loc[ventas_dia['Ventas'].idxmax(), 'Fecha'].date()})")

# ════════════════════════════════════════════════════════════════════════════
# 2. Regresion lineal
# ════════════════════════════════════════════════════════════════════════════
X = ventas_dia[["t"]]
y = ventas_dia["Ventas"]

modelo = LinearRegression()
modelo.fit(X, y)

ventas_dia["Pred_RL"] = modelo.predict(X)

# Dias futuros: abril completo (30 dias)
ultimo_t     = ventas_dia["t"].max()
ultima_fecha = ventas_dia["Fecha"].max()

futuro_t      = np.arange(ultimo_t + 1, ultimo_t + 31)
fechas_futuro = pd.date_range(ultima_fecha + pd.Timedelta(days=1), periods=30, freq="D")
pred_futuro   = modelo.predict(futuro_t.reshape(-1, 1))
pred_futuro   = np.maximum(pred_futuro, 0)   # no permitir negativos

df_futuro = pd.DataFrame({"Fecha": fechas_futuro, "t": futuro_t, "Pred_RL": pred_futuro})

# ════════════════════════════════════════════════════════════════════════════
# 3. Promedio movil 7 dias (sobre datos reales, extendido con el ultimo valor)
# ════════════════════════════════════════════════════════════════════════════
ventana = 7
ventas_dia["PM7"] = ventas_dia["Ventas"].rolling(ventana, min_periods=1).mean()

# Para el futuro: continuar la PM7 con las predicciones de RL
serie_extendida = pd.concat([
    ventas_dia[["Fecha", "Ventas"]],
    df_futuro[["Fecha"]].assign(Ventas=df_futuro["Pred_RL"])
]).reset_index(drop=True)

serie_extendida["PM7_ext"] = serie_extendida["Ventas"].rolling(ventana, min_periods=1).mean()
pm7_futuro = serie_extendida.iloc[-30:]["PM7_ext"].values

df_futuro["PM7"] = pm7_futuro

# ════════════════════════════════════════════════════════════════════════════
# 4. Metricas de error (sobre datos reales)
# ════════════════════════════════════════════════════════════════════════════
mae  = mean_absolute_error(y, ventas_dia["Pred_RL"])
rmse = np.sqrt(mean_squared_error(y, ventas_dia["Pred_RL"]))
r2   = modelo.score(X, y)

print(f"\n--- Metricas del modelo (regresion lineal) ---")
print(f"  MAE  (Error absoluto medio)  : ${mae:,.2f}")
print(f"  RMSE (Raiz error cuadratico) : ${rmse:,.2f}")
print(f"  R2   (Coef. determinacion)   : {r2:.4f}")
print(f"  Pendiente diaria             : ${modelo.coef_[0]:+.2f} / dia")

# ════════════════════════════════════════════════════════════════════════════
# 5. Estimado total abril
# ════════════════════════════════════════════════════════════════════════════
total_abril_rl  = df_futuro["Pred_RL"].sum()
total_abril_pm7 = df_futuro["PM7"].sum()
prom_dia_abril  = df_futuro["Pred_RL"].mean()

print(f"\n{'=' * 58}")
print(f"  Estimado de ventas para abril (Reg. Lineal) : ${total_abril_rl:>10,.2f}")
print(f"  Estimado de ventas para abril (Prom. Movil) : ${total_abril_pm7:>10,.2f}")
print(f"  Promedio diario proyectado                  : ${prom_dia_abril:>10,.2f}")
print(f"{'=' * 58}")

# ════════════════════════════════════════════════════════════════════════════
# 6. Grafica
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 1, figsize=(13, 9),
                         gridspec_kw={"height_ratios": [3, 1]})
ax, ax_err = axes

# -- Panel superior: series principales --
# Datos reales
ax.fill_between(ventas_dia["Fecha"], ventas_dia["Ventas"],
                alpha=0.12, color=AZUL)
ax.plot(ventas_dia["Fecha"], ventas_dia["Ventas"],
        color=AZUL, linewidth=1.4, alpha=0.85,
        marker="o", markersize=3, label="Ventas reales", zorder=3)

# Regresion lineal — historico
ax.plot(ventas_dia["Fecha"], ventas_dia["Pred_RL"],
        color=NARANJA, linewidth=2, linestyle="-",
        label="Regresion lineal (hist.)", zorder=4)

# PM7 — historico
ax.plot(ventas_dia["Fecha"], ventas_dia["PM7"],
        color=VERDE, linewidth=1.8, linestyle="-",
        label="Promedio movil 7 dias (hist.)", zorder=4)

# Predicciones abril — con banda de confianza aproximada
ax.fill_between(df_futuro["Fecha"],
                np.maximum(df_futuro["Pred_RL"] - rmse, 0),
                df_futuro["Pred_RL"] + rmse,
                alpha=0.13, color=NARANJA, label=f"Banda +/- RMSE (${rmse:,.0f})")

ax.plot(df_futuro["Fecha"], df_futuro["Pred_RL"],
        color=NARANJA, linewidth=2.2, linestyle="--",
        label=f"Proyeccion RL abril (${total_abril_rl:,.0f})", zorder=5)

ax.plot(df_futuro["Fecha"], df_futuro["PM7"],
        color=VERDE, linewidth=2, linestyle="--",
        label=f"Proyeccion PM7 abril (${total_abril_pm7:,.0f})", zorder=5)

# Linea separadora historico / futuro
sep = ultima_fecha + pd.Timedelta(days=0.5)
ax.axvline(sep, color=ROJO, linewidth=1.4, linestyle=":", alpha=0.8)
ax.text(sep + pd.Timedelta(days=0.8),
        ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else ventas_dia["Ventas"].max() * 0.95,
        "  Abril (proyeccion)", color=ROJO, fontsize=9, va="top")

ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

ax.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f"${x:,.0f}")
)
ax.set_title("Prediccion de Ventas para Abril 2026 — FIT LAND",
             fontsize=14, fontweight="bold", pad=14)
ax.set_ylabel("Total de venta diario")
ax.legend(loc="upper left", fontsize=8, framealpha=0.6, ncol=2)

# Anotacion con el estimado
ax.annotate(
    f"Estimado abril\n${total_abril_rl:,.0f} (RL)\n${total_abril_pm7:,.0f} (PM7)",
    xy=(df_futuro["Fecha"].iloc[14], df_futuro["Pred_RL"].iloc[14]),
    xytext=(df_futuro["Fecha"].iloc[14] - pd.Timedelta(days=10),
            df_futuro["Pred_RL"].iloc[14] + rmse * 1.5),
    fontsize=8.5,
    arrowprops=dict(arrowstyle="->", color="gray"),
    bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.8),
)

# -- Panel inferior: residuos del modelo --
residuos = ventas_dia["Ventas"] - ventas_dia["Pred_RL"]
colores_res = [ROJO if r < 0 else VERDE for r in residuos]
ax_err.bar(ventas_dia["Fecha"], residuos, color=colores_res, alpha=0.65, width=0.8)
ax_err.axhline(0, color="gray", linewidth=1)
ax_err.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
ax_err.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
plt.setp(ax_err.xaxis.get_majorticklabels(), rotation=30, ha="right")
ax_err.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax_err.set_ylabel("Residuo")
ax_err.set_title(f"Residuos del modelo  |  MAE=${mae:,.0f}   RMSE=${rmse:,.0f}   R²={r2:.3f}",
                 fontsize=10)

fig.tight_layout(pad=2.5)
out = os.path.join(DATA_DIR, "5_prediccion_ventas_abril.png")
fig.savefig(out, bbox_inches="tight")
plt.close(fig)
print(f"\n[OK] Grafica guardada en: {out}")
