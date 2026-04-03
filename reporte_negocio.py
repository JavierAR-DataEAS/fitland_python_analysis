import pandas as pd
import os
from datetime import datetime

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Carga de datos ───────────────────────────────────────────────────────────
ventas = pd.read_csv(os.path.join(DATA_DIR, "ventas_limpio.csv"))
ventas["Fecha"] = pd.to_datetime(ventas["Fecha"])
ventas.columns = ventas.columns.str.strip()

inv_feb = pd.read_csv(os.path.join(DATA_DIR, "inventario_febrero.csv"))
inv_mar = pd.read_csv(os.path.join(DATA_DIR, "inventario_marzo.csv"))

def limpiar_inv(df):
    df = df[df["ID"] != "--"].copy()
    df.columns = df.columns.str.strip()
    for col in ["Costo", "Precio"]:
        df[col] = df[col].astype(str).str.replace(r"[\$,]", "", regex=True).str.strip().astype(float)
    df["Stock inicial"] = pd.to_numeric(df["Stock inicial"], errors="coerce").fillna(0)
    df["Stock actual"]  = pd.to_numeric(df["Stock actual"],  errors="coerce").fillna(0)
    return df

inv_feb = limpiar_inv(inv_feb)
inv_mar = limpiar_inv(inv_mar)

# Periodos
ventas_feb = ventas[ventas["Mes"] == "2026-02"]
ventas_mar = ventas[ventas["Mes"] == "2026-03"]

ultimo_mes = "marzo 2026"
mes_anterior = "febrero 2026"

# ════════════════════════════════════════════════════════════════════════════
# 1. TENDENCIA AL ALZA: productos que vendieron mas en marzo vs febrero
# ════════════════════════════════════════════════════════════════════════════
def ventas_por_cat(df):
    return df.groupby("Categor\u00eda")["Cantidad"].sum()

cat_feb = ventas_por_cat(ventas_feb).rename("feb")
cat_mar = ventas_por_cat(ventas_mar).rename("mar")
tend_cat = pd.concat([cat_feb, cat_mar], axis=1).fillna(0)
tend_cat["variacion"] = tend_cat["mar"] - tend_cat["feb"]
tend_cat["pct_cambio"] = ((tend_cat["mar"] - tend_cat["feb"]) / (tend_cat["feb"] + 0.001) * 100).round(1)
tend_cat_alza = tend_cat[tend_cat["variacion"] > 0].sort_values("pct_cambio", ascending=False)

# Por producto individual
prod_feb = ventas_feb.groupby("ID")["Cantidad"].sum().rename("feb")
prod_mar = ventas_mar.groupby("ID")["Cantidad"].sum().rename("mar")
tend_prod = pd.concat([prod_feb, prod_mar], axis=1).fillna(0)
tend_prod["variacion"] = tend_prod["mar"] - tend_prod["feb"]
tend_prod = tend_prod[tend_prod["variacion"] > 0].sort_values("variacion", ascending=False)
# Enriquecer con nombre
tend_prod = tend_prod.merge(
    inv_mar[["ID", "Producto", "Categor\u00eda", "Talla", "Color"]].drop_duplicates("ID"),
    left_index=True, right_on="ID", how="left"
).set_index("ID")

# ════════════════════════════════════════════════════════════════════════════
# 2. SIN VENTAS EN MARZO: productos en inventario que no se vendieron
# ════════════════════════════════════════════════════════════════════════════
ids_vendidos_mar = set(ventas_mar["ID"].str.strip().unique())
sin_venta_mar = inv_mar[
    (~inv_mar["ID"].isin(ids_vendidos_mar)) & (inv_mar["Stock actual"] > 0)
].copy()
sin_venta_mar = sin_venta_mar.sort_values(["Categor\u00eda", "Producto"])

# ════════════════════════════════════════════════════════════════════════════
# 3. TALLAS MAS DEMANDADAS POR CATEGORIA
# ════════════════════════════════════════════════════════════════════════════
tallas = (
    ventas.groupby(["Categor\u00eda", "Talla"])["Cantidad"]
    .sum()
    .reset_index()
    .sort_values(["Categor\u00eda", "Cantidad"], ascending=[True, False])
)
# Top 3 tallas por categoria
top_tallas = (
    tallas.groupby("Categor\u00eda")
    .apply(lambda g: g.nlargest(3, "Cantidad"), include_groups=False)
    .reset_index(level=0)
)

# ════════════════════════════════════════════════════════════════════════════
# 4. REPOSICION URGENTE
#    Criterios: alto volumen de ventas total + stock actual bajo o agotado
# ════════════════════════════════════════════════════════════════════════════
ventas_total = ventas.groupby("ID")["Cantidad"].sum().rename("unidades_vendidas")
reposicion = inv_mar.merge(ventas_total, on="ID", how="left")
reposicion["unidades_vendidas"] = reposicion["unidades_vendidas"].fillna(0)
reposicion["rotacion"] = reposicion["unidades_vendidas"] / (reposicion["Stock inicial"] + 0.001)

# Solo los que tienen stock bajo (<=1) y buenas ventas
urgentes = reposicion[reposicion["Stock actual"] <= 1].copy()
urgentes = urgentes.sort_values("unidades_vendidas", ascending=False).head(10)
top3_urgentes = urgentes.head(3)

# ════════════════════════════════════════════════════════════════════════════
# 5. MARGEN DE GANANCIA POR CATEGORIA
# ════════════════════════════════════════════════════════════════════════════
margen = (
    ventas.groupby("Categor\u00eda")
    .agg(
        Ingresos=("Total venta", "sum"),
        Costos=("Total Costo", "sum"),
        Ganancia=("Ganancia", "sum"),
        Transacciones=("ID", "count"),
    )
    .reset_index()
)
margen["Margen_%"] = (margen["Ganancia"] / margen["Ingresos"] * 100).round(1)
margen = margen.sort_values("Margen_%", ascending=False)

# Margen promedio general
margen_general = (ventas["Ganancia"].sum() / ventas["Total venta"].sum() * 100)

# ════════════════════════════════════════════════════════════════════════════
# CONSTRUCCION DEL REPORTE
# ════════════════════════════════════════════════════════════════════════════
lineas = []

def titulo(t):
    lineas.append("")
    lineas.append("=" * 62)
    lineas.append(f"  {t}")
    lineas.append("=" * 62)

def sub(t):
    lineas.append(f"\n  >> {t}")
    lineas.append("  " + "-" * 50)

def li(txt):
    lineas.append(f"     - {txt}")

def parr(txt):
    lineas.append(f"  {txt}")

ahora = datetime.now().strftime("%d/%m/%Y %H:%M")
lineas.append("=" * 62)
lineas.append("  REPORTE DE ANALISIS DE NEGOCIO — FIT LAND")
lineas.append(f"  Generado el {ahora}")
lineas.append(f"  Periodo analizado: {mes_anterior}  a  {ultimo_mes}")
lineas.append("=" * 62)

# ── 1. Tendencia al alza ─────────────────────────────────────────────────
titulo("1. PRODUCTOS / CATEGORIAS CON TENDENCIA AL ALZA")

sub("Categorias con mayor crecimiento (Feb -> Mar)")
for cat, row in tend_cat_alza.iterrows():
    signo = "+" if row["variacion"] > 0 else ""
    li(f"{cat:<18}  Feb: {int(row['feb']):>3} uds   Mar: {int(row['mar']):>3} uds   "
       f"({signo}{row['pct_cambio']}%)")

if tend_cat_alza.empty:
    parr("No se detectaron categorias con crecimiento entre periodos.")

sub("Top 5 productos individuales con mas crecimiento")
for idx, (iid, row) in enumerate(tend_prod.head(5).iterrows()):
    nombre = f"{row.get('Producto','?')} | {row.get('Talla','?')} | {row.get('Color','?')}"
    li(f"{nombre:<38} +{int(row['variacion'])} uds  (Mar: {int(row['mar'])})")

if tend_prod.empty:
    parr("No hay productos que hayan subido ventas entre periodos.")

parr("")
parr("CONCLUSION: Enfoca reabastecimiento en las categorias/productos")
parr("            con tendencia al alza antes de que se agoten.")

# ── 2. Sin ventas en marzo ───────────────────────────────────────────────
titulo("2. PRODUCTOS SIN VENTAS EN MARZO (con stock disponible)")

parr(f"Se encontraron {len(sin_venta_mar)} productos con stock > 0 que no")
parr("se vendieron en marzo. Pueden requerir promocion o descuento.")
parr("")

if not sin_venta_mar.empty:
    cat_actual = None
    for _, row in sin_venta_mar.iterrows():
        cat = row["Categor\u00eda"].strip() if pd.notna(row["Categor\u00eda"]) else "Sin categoria"
        if cat != cat_actual:
            lineas.append(f"\n  [{cat}]")
            cat_actual = cat
        li(f"{row['ID']:<22} {row['Producto']:<12} Talla {row['Talla']:<6} "
           f"Color: {str(row['Color']):<14} Stock: {int(row['Stock actual'])}")
else:
    parr("Todos los productos con stock tuvieron al menos una venta en marzo.")

# ── 3. Tallas mas demandadas ─────────────────────────────────────────────
titulo("3. TALLAS MAS DEMANDADAS POR CATEGORIA")

for cat, grupo in top_tallas.groupby("Categor\u00eda"):
    lineas.append(f"\n  [{cat.strip()}]")
    for _, row in grupo.iterrows():
        bar_len = int(row["Cantidad"] / top_tallas["Cantidad"].max() * 20)
        bar = "#" * bar_len
        li(f"Talla {str(row['Talla']):<8}  {int(row['Cantidad']):>3} uds  [{bar}]")

parr("")
parr("RECOMENDACION: Al hacer nuevos pedidos, prioriza las tallas")
parr("de mayor demanda en cada categoria.")

# ── 4. Reposicion urgente ────────────────────────────────────────────────
titulo("4. PRODUCTOS A REPONER URGENTE (Top 3)")

parr("Criterio: alto volumen de ventas historico + stock actual <= 1")
parr("")

posicion = 1
for _, row in top3_urgentes.iterrows():
    nombre = f"{row['Producto'].strip()} | Talla {row['Talla']} | {row['Color'].strip()}"
    rotacion_txt = f"{row['rotacion']:.1f}x" if row["Stock inicial"] > 0 else "N/A"
    lineas.append(f"  #{posicion}  {row['ID']}")
    li(f"Descripcion : {nombre}")
    li(f"Ventas hist.: {int(row['unidades_vendidas'])} unidades vendidas en total")
    li(f"Stock actual: {int(row['Stock actual'])} unidad(es)")
    li(f"Costo reponer 1 pza: ${row['Costo']:.2f}   Precio venta: ${row['Precio']:.2f}")
    lineas.append("")
    posicion += 1

parr("ACCION: Coloca pedido esta semana para los 3 productos anteriores.")
parr("         Prioriza #1 y #2 si el presupuesto es limitado.")

# ── 5. Margen de ganancia por categoria ─────────────────────────────────
titulo("5. MARGEN DE GANANCIA POR CATEGORIA")

parr(f"  Margen general del negocio: {margen_general:.1f}%")
parr("")
parr(f"  {'Categoria':<18} {'Ingresos':>10} {'Costos':>9} {'Ganancia':>10} {'Margen':>8} {'Ventas':>7}")
parr("  " + "-" * 66)
for _, row in margen.iterrows():
    parr(
        f"  {row['Categor\u00eda'].strip():<18} "
        f"${row['Ingresos']:>9,.0f} "
        f"${row['Costos']:>8,.0f} "
        f"${row['Ganancia']:>9,.0f} "
        f"  {row['Margen_%']:>6.1f}% "
        f"  {row['Transacciones']:>5}"
    )

parr("")
top_margen = margen.iloc[0]
bot_margen = margen.iloc[-1]
parr(f"  Categoria mas rentable : {top_margen['Categor\u00eda'].strip()} ({top_margen['Margen_%']}%)")
parr(f"  Categoria menos rentable: {bot_margen['Categor\u00eda'].strip()} ({bot_margen['Margen_%']}%)")
parr("")
parr("RECOMENDACION: Revisa si puedes ajustar precios en categorias con")
parr("margen bajo, o reducir costos de adquisicion negociando con proveedor.")

# ── Resumen ejecutivo ────────────────────────────────────────────────────
titulo("RESUMEN EJECUTIVO — ACCIONES PRIORITARIAS")

total_ventas   = ventas["Total venta"].sum()
ganancia_total = ventas["Ganancia"].sum()
crecimiento    = ((ventas_mar["Total venta"].sum() - ventas_feb["Total venta"].sum())
                  / ventas_feb["Total venta"].sum() * 100)

parr(f"  Periodo: {mes_anterior} a {ultimo_mes}")
parr(f"  Ingresos totales : ${total_ventas:,.2f}")
parr(f"  Ganancia total   : ${ganancia_total:,.2f}  (margen {margen_general:.1f}%)")
parr(f"  Crecimiento MoM  : +{crecimiento:.1f}% en ingresos")
parr("")
lineas.append("  ACCIONES INMEDIATAS:")
li(f"Reponer {top3_urgentes.iloc[0]['ID']} y {top3_urgentes.iloc[1]['ID']} (stock critico + alta rotacion)")
li(f"Aplicar descuento/promocion a los {len(sin_venta_mar)} productos sin venta en marzo")
cat_alza_top = tend_cat_alza.index[0] if not tend_cat_alza.empty else "N/A"
li(f"Aumentar inventario en categoria '{cat_alza_top}' (mayor crecimiento)")
li(f"Revisar precios en '{bot_margen['Categor\u00eda'].strip()}' para mejorar margen ({bot_margen['Margen_%']}%)")
lineas.append("")
lineas.append("=" * 62)
lineas.append("  Reporte generado automaticamente con Python / pandas")
lineas.append("=" * 62)

# ── Guardar reporte ──────────────────────────────────────────────────────
reporte_txt = "\n".join(lineas)
output_path = os.path.join(DATA_DIR, "reporte_fitland.txt")
with open(output_path, "w", encoding="utf-8") as f:
    f.write(reporte_txt)

print(reporte_txt)
print(f"\n[OK] Reporte guardado en: {output_path}")
