# librerias
import streamlit as st
import pandas as pd
import statsmodels.api as sm

# 1. Configuracion del título de la pestaña del navegador
st.set_page_config(page_title="Calculadora Estadística", page_icon="📈")

st.title("Proyecto Final - Estadística Inferencial 📊")

# 2. Barra lateral (sidebar) para el menú
st.sidebar.title("Navegación 🧭")
opcion = st.sidebar.radio(
    "Selecciona la parte del proyecto:",
    ("Inicio", "Parte 1: ANOVA", "Parte 2: Regresión Múltiple")
)

# 3. Le decimos a la app qué mostrar según la opción elegida
if opcion == "Inicio":
    st.write("¡Bienvenido! Usa el menú de la izquierda para navegar.")
    
elif opcion == "Parte 1: ANOVA":
    st.subheader("Evaluación de Arquitecturas de Red 🌐")
    st.write("Aquí se cargan los datos de SwiftVen, SwiftFast y SwiftPay.")
    
elif opcion == "Parte 2: Regresión Múltiple":
    st.subheader("Evaluación del Uso de CPU 💻")
    st.write("Datos recolectados del tráfico transaccional:")
    
    # 1. Upload de datos (simulado aquí con un diccionario)
    datos_cpu = {
        "y (Uso CPU)": [9.8, 12.6, 11.9, 13.1, 13.3, 13.5, 10.1, 13.1, 10.7, 11.0, 13.0, 11.6, 12.0, 11.4, 12.2, 12.8, 12.4, 13.2, 10.6, 7.9],
        "x1 (Peticiones/s)": [3.3, 4.4, 3.9, 5.9, 4.6, 5.2, 4.0, 4.7, 4.5, 3.7, 4.6, 4.7, 3.9, 4.6, 5.1, 5.0, 4.8, 5.3, 3.9, 3.4],
        "x2 (Tamaño Trama)": [2.8, 4.9, 5.3, 2.6, 5.1, 3.2, 4.0, 4.5, 4.1, 3.6, 4.6, 3.5, 4.6, 4.0, 3.6, 4.4, 4.4, 3.5, 3.8, 3.8],
        "x3 (Latencia Bóveda)": [3.1, 3.5, 4.8, 3.1, 5.0, 3.3, 3.3, 3.5, 3.7, 3.3, 3.6, 3.5, 3.6, 3.4, 3.3, 3.6, 3.4, 3.6, 3.4, 3.4],
        "x4 (Memoria Microserv.)": [4.1, 3.9, 4.7, 3.6, 4.1, 4.3, 4.0, 3.8, 3.6, 3.6, 3.6, 3.7, 4.1, 3.6, 4.0, 3.7, 3.6, 3.7, 4.0, 3.4]
    }
    
    # 2. Conversion de los datos en una tabla interactiva con "pandas"
    df_cpu = pd.DataFrame(datos_cpu)
    st.dataframe(df_cpu, use_container_width=True)

    # --- PREGUNTA 1: ESTIMADORES ---
    st.write("---")
    st.subheader("1. Estimadores del Modelo (Mínimos Cuadrados)")
    
    X = df_cpu[['x1 (Peticiones/s)', 'x2 (Tamaño Trama)', 'x3 (Latencia Bóveda)', 'x4 (Memoria Microserv.)']]
    y = df_cpu['y (Uso CPU)']
    X = sm.add_constant(X)
    modelo = sm.OLS(y, X).fit()
    
    # Extraemos los betas (coeficientes) y los mostramos en un cuadro azul claro
    st.info(f"""
    **Ecuación del Modelo:**
    * **$\\beta_0$ (Constante):** {modelo.params.iloc[0]:.4f}
    * **$\\beta_1$ (Peticiones/s):** {modelo.params.iloc[1]:.4f}
    * **$\\beta_2$ (Tamaño Trama):** {modelo.params.iloc[2]:.4f}
    * **$\\beta_3$ (Latencia Bóveda):** {modelo.params.iloc[3]:.4f}
    * **$\\beta_4$ (Memoria Microserv.):** {modelo.params.iloc[4]:.4f}
    """)

    # Escondemos el gran resumen estadístico en un botón desplegable
    with st.expander("Ver tabla estadística completa (Statsmodels)"):
        st.text(modelo.summary())

    st.write("---")
    st.subheader("2. Predicción del Uso de CPU")
    st.write("Escenario de prueba: $x_1=5.1$, $x_2=4.7$, $x_3=4.8$, $x_4=4.0$")
    
    # Se crea una lista con los valores del profesor. 
    # El primer '1' es obligatorio para representar la constante matemática (el Beta 0)
    valores_prediccion = [1, 5.1, 4.7, 4.8, 4.0]
    
    # Le pedimos al modelo que calcule el resultado
    resultado_cpu = modelo.predict(valores_prediccion)
    
    # Muestra el resultado en una cajita verde muy visual
    st.success(f"El Uso de CPU estimado por el modelo es: **{resultado_cpu[0]:.5f}%**")

    # --- PREGUNTA 3: ANOVA (BONDAD DE AJUSTE) ---
    st.write("---")
    st.subheader("3. Bondad de Ajuste (ANOVA)")
    
    # Extraemos los valores exactos escondidos en el modelo
    f_stat = modelo.fvalue
    p_valor = modelo.f_pvalue
    
    # Mostramos los números limpios
    st.write(f"**Estadístico F:** {f_stat:.2f}")
    st.write(f"**Valor-p:** {p_valor:.6f}")
    
    # Le enseñamos a la app a tomar la decisión automáticamente
    if p_valor < 0.05:
        st.success("✅ **Conclusión:** Como el Valor-p es menor a **0.05**, se rechaza la Hipótesis Nula. **¡El modelo tiene una bondad de ajuste significativa!** (Las variables sí ayudan a predecir el Uso de CPU).")
    else:
        st.error("❌ **Conclusión:** Como el Valor-p es mayor a **0.05**, se acepta la Hipótesis Nula. El modelo no es significativo.")

# --- PREGUNTA 4: COEFICIENTE DE DETERMINACIÓN ---
    st.write("---")
    st.subheader("4. Coeficiente de Determinación ($R^2$)")
    
    # Extraemos el R-cuadrado directamente del modelo
    r_cuadrado = modelo.rsquared
    
    # Mostramos el valor numérico
    st.write(f"**Valor de $R^2$:** {r_cuadrado:.4f}")
    
    # Cajita azul con la interpretación automática
    st.info(f"💡 **Interpretación:** El modelo es capaz de explicar aproximadamente el **{r_cuadrado * 100:.2f}%** de la variabilidad en el Uso de CPU. Esto significa que las variaciones en el tráfico (peticiones, tamaño, latencia y memoria) son responsables de casi el 80% del comportamiento del procesador de los servidores.")

# --- PREGUNTA 5: INTERVALOS DE CONFIANZA ---
    st.write("---")
    st.subheader("5. Intervalos de Confianza (95%) para los Estimadores")
    
    # Extraemos los intervalos del modelo
    intervalos = modelo.conf_int(alpha=0.05)
    
    # Les ponemos nombres bonitos a las columnas
    intervalos.columns = ['Límite Inferior (2.5%)', 'Límite Superior (97.5%)']
    
    # Mostramos la tabla limpia en la app
    st.dataframe(intervalos, use_container_width=True)
    
    # Agregamos la interpretación visual
    st.info("💡 **Interpretación:** Estos rangos indican que estamos un 95% seguros de que el verdadero valor (el multiplicador exacto) de cada parámetro se encuentra dentro de estos límites. Si un intervalo incluye el cero (pasa de negativo a positivo), significa que esa variable podría no tener un impacto real.")

# --- PREGUNTA 6: SIGNIFICANCIA INDIVIDUAL ---
    st.write("---")
    st.subheader("6. Significancia Individual (Prueba t)")
    st.write("Nivel de significancia: **5% (0.05)**")
    
    # Extraemos los p-valores individuales (saltando el Beta 0 que es la constante)
    p_valores_ind = modelo.pvalues[1:] 
    nombres_vars = p_valores_ind.index
    
    # Evaluamos cada variable con un ciclo automático
    for nombre, p_val in zip(nombres_vars, p_valores_ind):
        if p_val < 0.05:
            st.success(f"✅ **{nombre}:** Valor-p = {p_val:.3f} (Aprobada / Significativa)")
        else:
            st.warning(f"❌ **{nombre}:** Valor-p = {p_val:.3f} (Reprobada / No significativa)")

# --- PREGUNTA 7: RECOMENDACIÓN ---
    st.write("---")
    st.subheader("7. Recomendación Final")
    
    st.info("💡 **Modelo Propuesto:** Se recomienda implementar un **modelo reducido** que utilice exclusivamente $x_1$ (Peticiones) y $x_2$ (Tamaño de Trama).")
    
    st.write("Las pruebas estadísticas demuestran que las variables de Latencia y Memoria no superan el nivel de significancia del 5%, por lo que descartarlas optimizará la ecuación sin perder capacidad predictiva real.")