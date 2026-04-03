import pandas as pd
import os

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# ── 1. Cargar y combinar archivos de ventas ──────────────────────────────────
ventas_feb = pd.read_csv(os.path.join(DATA_DIR, "ventas_febrero.csv"))
ventas_mar = pd.read_csv(os.path.join(DATA_DIR, "ventas_marzo.csv"))

ventas = pd.concat([ventas_feb, ventas_mar], ignore_index=True)

# Eliminar columnas vacías extra (las que vienen sin nombre)
ventas = ventas.loc[:, ~ventas.columns.str.startswith("Unnamed")]
ventas.columns = ventas.columns.str.strip()

# Eliminar filas sin Fecha ni ID (filas vacías de relleno)
ventas.dropna(subset=["Fecha", "ID"], inplace=True)

# ── 2. Limpiar y convertir tipos ─────────────────────────────────────────────
# Fecha → datetime
ventas["Fecha"] = pd.to_datetime(ventas["Fecha"], format="%m/%d/%Y")

# Columnas monetarias: quitar $ y comas, convertir a float
money_cols = ["Precio de venta", "Costo unitario", "Total venta", "Total Costo", "Ganancia"]
for col in money_cols:
    ventas[col] = (
        ventas[col]
        .astype(str)
        .str.replace(r"[\$,]", "", regex=True)
        .str.strip()
        .astype(float)
    )

ventas["Cantidad"] = pd.to_numeric(ventas["Cantidad"], errors="coerce")

# Agregar columna Mes para facilitar agrupaciones
ventas["Mes"] = ventas["Fecha"].dt.to_period("M")

# ── 3. Enriquecer con datos de inventario (nombre del producto) ──────────────
inv_feb = pd.read_csv(os.path.join(DATA_DIR, "inventario_febrero.csv"))
inv_mar = pd.read_csv(os.path.join(DATA_DIR, "inventario_marzo.csv"))

inventario = pd.concat([inv_feb, inv_mar], ignore_index=True)
inventario = inventario[inventario["ID"] != "--"].drop_duplicates(subset="ID")
inventario.columns = inventario.columns.str.strip()

ventas = ventas.merge(
    inventario[["ID", "Producto", "Categoría", "Talla", "Color"]],
    on="ID",
    how="left",
)

# ── 4. Resumen general ───────────────────────────────────────────────────────
total_ventas   = ventas["Total venta"].sum()
ganancia_total = ventas["Ganancia"].sum()
ticket_promedio = ventas["Total venta"].mean()
num_transacciones = len(ventas)

# Producto más vendido (por cantidad)
mas_vendido = (
    ventas.groupby("ID")["Cantidad"]
    .sum()
    .sort_values(ascending=False)
)
top_id = mas_vendido.index[0]
top_cantidad = mas_vendido.iloc[0]

# Enriquecer el top con nombre de producto
top_info = inventario[inventario["ID"] == top_id][["ID", "Producto", "Talla", "Color"]]
top_desc = f"{top_id}"
if not top_info.empty:
    r = top_info.iloc[0]
    top_desc = f"{r['Producto']} | Talla {r['Talla']} | {r['Color']} ({top_id})"

# Resumen por mes
resumen_mes = ventas.groupby("Mes").agg(
    Transacciones=("ID", "count"),
    Total_Ventas=("Total venta", "sum"),
    Ganancia=("Ganancia", "sum"),
    Ticket_Promedio=("Total venta", "mean"),
).reset_index()

# Resumen por categoría
resumen_categoria = ventas.groupby("Categoría").agg(
    Unidades=("Cantidad", "sum"),
    Total_Ventas=("Total venta", "sum"),
    Ganancia=("Ganancia", "sum"),
).sort_values("Ganancia", ascending=False).reset_index()

# ── 5. Imprimir resultados ───────────────────────────────────────────────────
sep = "-" * 55

print(f"\n{'FIT LAND - Resumen de Ventas':^55}")
print(sep)
print(f"  Periodo analizado : {ventas['Fecha'].min().date()} a {ventas['Fecha'].max().date()}")
print(f"  Transacciones     : {num_transacciones:>8,}")
print(f"  Total ventas      : ${total_ventas:>12,.2f}")
print(f"  Ganancia total    : ${ganancia_total:>12,.2f}")
print(f"  Ticket promedio   : ${ticket_promedio:>12,.2f}")
print(f"  Producto + vendido: {top_desc}")
print(f"                      {top_cantidad:.0f} unidades")

print(f"\n{'Por mes':^55}")
print(sep)
print(resumen_mes.to_string(index=False))

print(f"\n{'Por categoría':^55}")
print(sep)
print(resumen_categoria.to_string(index=False))

# ── 6. Guardar datos limpios ─────────────────────────────────────────────────
output_path = os.path.join(DATA_DIR, "ventas_limpio.csv")
ventas.to_csv(output_path, index=False, encoding="utf-8-sig")
print(f"\n[OK] Datos limpios guardados en: {output_path}")
print(f"  {len(ventas)} filas  ×  {len(ventas.columns)} columnas")
