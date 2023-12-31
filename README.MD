# openCore_stats

Estadisticas del proyecto `openCore`. Taller de sistemas. UNAP 2023.

## Tabla de contenidos

- [openCore\_stats](#opencore_stats)
  - [Tabla de contenidos](#tabla-de-contenidos)
  - [Instalación](#instalación)
  - [Uso](#uso)
  - [Documentación Streamlit](#documentación-streamlit)

## Instalación

1. Clona el repositorio.
    ```shell
    git clone https://github.com/agutierrezmorag/opencore_stats.git
    ```

2. Navega al directorio del proyecto.
    ```shell
    cd opencore_stats
    ```

3. Crea un entorno virtual.
    ```shell
    python3 -m venv env
    ```

4. Activa el entorno virtual.
    - En Windows:
        ```shell
        .\env\Scripts\activate
        ```
    - En Unix o MacOS:
        ```shell
        source env/bin/activate
        ```

5. Instala las dependencias.
    ```shell
    pip install -r requirements.txt
    ```

6. Mueve el archivo secrets.toml (disponible en los canales de comunicacion) a la carpeta `.streamlit`.

## Uso

1. Ejecuta la aplicación Streamlit.
    ```shell
    streamlit run app.py
    ```

2. Abre tu navegador web y ve a `http://localhost:8501`.

## Documentación Streamlit

Streamlit es una biblioteca de Python de código abierto que facilita la creación de aplicaciones web interactivas para proyectos relacionados con el data science.

Para más detalles y una guía completa sobre cómo usar Streamlit, consultar la [documentación oficial de la API de Streamlit](https://docs.streamlit.io/en/stable/api.html).