"""Revisión y reparación del entorno en Windows (choque de librerías de C++ con LightGBM).

Problema que resuelve: en Windows, scikit-learn 1.4.2 trae su propia copia de la librería
de C++ de Microsoft (`sklearn/.libs/msvcp140.dll`, versión 14.32) y la carga al importarse.
LightGBM 4.7.0 está compilado con una versión más nueva (14.44) y necesita una
`msvcp140.dll` 14.40 o superior. Como Windows reutiliza la copia que ya está cargada,
LightGBM termina usando la vieja y falla al entrenar con
`OSError: exception: access violation reading 0x0000000000000000`.

La reparación reemplaza la copia vieja de scikit-learn por la versión del sistema
(Microsoft garantiza que las versiones nuevas funcionan con programas compilados con las
viejas), guardando un respaldo. No cambia ninguna versión de librería ni ningún resultado:
solo la librería de soporte de C++ que usan scikit-learn y LightGBM.

Uso (Windows, con el entorno .venv activado):

    python -m churn.reparar_entorno              # diagnóstico
    python -m churn.reparar_entorno --reparar    # reparación (cerrar antes los kernels de Jupyter)

Este módulo solo usa la librería estándar y no importa scikit-learn ni LightGBM, para no
cargar las DLL que revisa.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

VERSION_MINIMA = (14, 40)  # msvcp140.dll a partir de la cual funciona LightGBM 4.7 (VS 2022 17.10)
DLLS_REEMPLAZABLES = ("msvcp140.dll", "vcomp140.dll")


def version_dll(ruta: Path) -> tuple[int, int, int, int] | None:
    """Versión de archivo de una DLL, leída de su estructura VS_FIXEDFILEINFO."""
    try:
        contenido = Path(ruta).read_bytes()
    except OSError:
        return None
    posicion = contenido.find(b"\xbd\x04\xef\xfe")  # firma 0xFEEF04BD
    if posicion < 0:
        return None
    ms = int.from_bytes(contenido[posicion + 8:posicion + 12], "little")
    ls = int.from_bytes(contenido[posicion + 12:posicion + 16], "little")
    return ms >> 16, ms & 0xFFFF, ls >> 16, ls & 0xFFFF


def version_enlazador(ruta: Path) -> tuple[int, int] | None:
    """Versión del compilador (enlazador) con que se construyó una DLL, desde su encabezado PE."""
    try:
        contenido = Path(ruta).read_bytes()
        inicio_pe = int.from_bytes(contenido[0x3C:0x40], "little")
        if contenido[inicio_pe:inicio_pe + 4] != b"PE\x00\x00":
            return None
        opcional = inicio_pe + 24
        return contenido[opcional + 2], contenido[opcional + 3]
    except (OSError, IndexError):
        return None


def _carpeta_paquete(nombre: str) -> Path | None:
    """Carpeta de un paquete instalado, sin importarlo."""
    especificacion = importlib.util.find_spec(nombre)
    if especificacion is None or not especificacion.submodule_search_locations:
        return None
    return Path(list(especificacion.submodule_search_locations)[0])


def rutas(site_packages_sklearn: Path | None = None, carpeta_lightgbm: Path | None = None,
          system32: Path | None = None) -> dict[str, Path | None]:
    """Rutas de las DLL involucradas (se pueden pasar a mano para probar)."""
    sklearn_dir = site_packages_sklearn or _carpeta_paquete("sklearn")
    lightgbm_dir = carpeta_lightgbm or _carpeta_paquete("lightgbm")
    system32 = system32 or Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"
    dll_lightgbm = None
    if lightgbm_dir is not None:  # mismas ubicaciones que busca LightGBM
        for candidata in (lightgbm_dir.parent / "lib_lightgbm.dll", lightgbm_dir / "bin" / "lib_lightgbm.dll",
                          lightgbm_dir / "lib" / "lib_lightgbm.dll"):
            if candidata.is_file():
                dll_lightgbm = candidata
                break
    salida = {"lightgbm": dll_lightgbm}
    for dll in DLLS_REEMPLAZABLES:
        salida[f"sklearn_{dll}"] = sklearn_dir / ".libs" / dll if sklearn_dir else None
        salida[f"sistema_{dll}"] = system32 / dll
    return salida


def diagnosticar(rutas_dll: dict | None = None) -> dict:
    """Versiones encontradas y si hay choque entre la copia de scikit-learn y LightGBM."""
    rutas_dll = rutas_dll or rutas()
    informe = {"lightgbm_compilado_con": version_enlazador(rutas_dll["lightgbm"]) if rutas_dll["lightgbm"] else None}
    for clave, ruta in rutas_dll.items():
        if clave != "lightgbm":
            informe[clave] = version_dll(ruta) if ruta is not None and Path(ruta).is_file() else None
    copia_sklearn = informe.get("sklearn_msvcp140.dll")
    compilado = informe["lightgbm_compilado_con"]
    informe["hay_choque"] = bool(compilado and copia_sklearn and compilado >= VERSION_MINIMA
                                 and copia_sklearn[:2] < VERSION_MINIMA)
    sistema = informe.get("sistema_msvcp140.dll")
    informe["sistema_actualizado"] = bool(sistema and sistema[:2] >= VERSION_MINIMA)
    return informe


def reparar(rutas_dll: dict | None = None) -> list[str]:
    """Reemplaza las DLL viejas de scikit-learn por las del sistema, con respaldo `.original`.
    Devuelve la lista de acciones. Lanza RuntimeError si el sistema no tiene una versión nueva."""
    rutas_dll = rutas_dll or rutas()
    informe = diagnosticar(rutas_dll)
    if not informe["hay_choque"]:
        return ["No hay choque: no se cambió nada."]
    if not informe["sistema_actualizado"]:
        raise RuntimeError(
            "La msvcp140.dll de Windows también es vieja. Instale primero el 'Microsoft Visual C++ "
            "Redistributable 2015-2022 (x64)' más reciente: https://aka.ms/vs/17/release/vc_redist.x64.exe "
            "y vuelva a ejecutar este comando.")
    acciones = []
    for dll in DLLS_REEMPLAZABLES:
        destino, origen = rutas_dll[f"sklearn_{dll}"], rutas_dll[f"sistema_{dll}"]
        v_destino, v_origen = version_dll(destino), version_dll(origen)
        if v_destino is None or v_origen is None or v_destino >= v_origen:
            continue
        respaldo = Path(str(destino) + ".original")
        try:
            if not respaldo.exists():
                shutil.copy2(destino, respaldo)
            shutil.copy2(origen, destino)
        except PermissionError as error:
            raise RuntimeError("Windows no dejó reemplazar el archivo porque está en uso. Cierre VS Code "
                               "(o reinicie los kernels de Jupyter) y ejecute el comando desde una terminal "
                               "nueva.") from error
        acciones.append(f"{dll}: {'.'.join(map(str, v_destino))} -> {'.'.join(map(str, v_origen))} "
                        f"(respaldo en {respaldo.name})")
    return acciones


def probar_lightgbm() -> bool:
    """Entrena un LightGBM pequeño en un proceso nuevo, importando antes scikit-learn (el orden
    que produce el error). True si funciona."""
    codigo = ("import sklearn, numpy as np\nfrom lightgbm import LGBMClassifier\n"
              "X = np.random.rand(300, 4); y = (X[:, 0] > 0.5).astype(int)\n"
              "LGBMClassifier(n_jobs=1, verbose=-1).fit(X, y); print('ok')")
    resultado = subprocess.run([sys.executable, "-c", codigo], capture_output=True, text=True)
    return resultado.returncode == 0 and "ok" in resultado.stdout


def advertir_si_hay_choque() -> None:
    """Aviso corto al importar `churn` en Windows si el choque sigue presente."""
    if sys.platform != "win32":
        return
    try:
        if diagnosticar()["hay_choque"]:
            import warnings
            warnings.warn("LightGBM fallará en este Windows por una msvcp140.dll vieja dentro de scikit-learn. "
                          "Ejecute en la terminal: python -m churn.reparar_entorno --reparar", stacklevel=3)
    except Exception:  # el aviso nunca debe impedir importar el proyecto
        pass


def _texto(version) -> str:
    return ".".join(map(str, version)) if version else "no encontrada"


def main(argumentos: list[str] | None = None) -> int:
    lector = argparse.ArgumentParser(prog="python -m churn.reparar_entorno", description=__doc__.splitlines()[0])
    lector.add_argument("--reparar", action="store_true", help="Reemplaza las DLL viejas de scikit-learn")
    args = lector.parse_args(argumentos)
    if sys.platform != "win32":
        print("Este diagnóstico solo aplica en Windows; en este sistema no hace falta.")
        return 0
    informe = diagnosticar()
    print(f"LightGBM compilado con: {_texto(informe['lightgbm_compilado_con'])}")
    for dll in DLLS_REEMPLAZABLES:
        print(f"{dll}: scikit-learn {_texto(informe[f'sklearn_{dll}'])} | Windows {_texto(informe[f'sistema_{dll}'])}")
    print(f"Choque detectado: {'sí' if informe['hay_choque'] else 'no'}")
    if args.reparar:
        try:
            for accion in reparar():
                print(f"   {accion}")
        except RuntimeError as error:
            print(f"   No se pudo reparar: {error}")
            return 1
    funciona = probar_lightgbm()
    print(f"Prueba de LightGBM después de importar scikit-learn: {'OK' if funciona else 'FALLA'}")
    if not funciona and not args.reparar:
        print("Ejecute: python -m churn.reparar_entorno --reparar")
    return 0 if funciona else 1

