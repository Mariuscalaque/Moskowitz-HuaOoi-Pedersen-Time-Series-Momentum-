#!/usr/bin/env python3
"""
regenerate_all.py — Régénération COMPLÈTE en une passe : cœur de la réplication
                    (pipeline) PUIS les trois extensions copule (A, B, C).

Pourquoi ce script ?
  `regenerate.py` ne régénérait que le cœur (tables/figures/perf du papier).
  Les extensions A/B/C, elles, devaient être lancées à la main, séparément, et
  APRÈS le pipeline (car A et B consomment outputs/tables/diversified_tsmom_series.csv
  produit par le pipeline). Ce script enchaîne tout dans le bon ordre, depuis la
  racine du projet, pour que TOUS les outputs (cœur + extensions) proviennent du
  même run et du même code — une seule source de vérité.

Ordre d'exécution :
  1) (option) table rase : suppression de outputs/ ;
  2) pipeline principal -> outputs/tables, outputs/figures, performance_summary,
     diversified_tsmom_series ;
  3) extensions A, B, C (sous-processus, cwd = racine projet) -> ext_A_*, ext_B_*,
     ext_C_*, fig8..fig12 dans outputs/tables et outputs/figures.

Usage :
    python regenerate_all.py                       # 1985-2009, avec externes, tout
    python regenerate_all.py --end 2025-12-31       # échantillon étendu
    python regenerate_all.py --no-external          # sans téléchargements
    python regenerate_all.py --skip-extensions      # cœur seulement (= ancien regenerate)
    python regenerate_all.py --skip-core            # extensions seulement (cœur déjà à jour)
    python regenerate_all.py --refilter             # ext B sans look-ahead (refiltrage/fenêtre)
    python regenerate_all.py --ext-b-window 48      # ext B : fenêtre roulante = 48 mois
    python regenerate_all.py --no-clean             # ne pas effacer outputs/ d'abord
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from src.config import OUTPUT_DIR, TABLES_DIR, PAPER_START, PAPER_END
from src.pipeline import run

_ROOT = Path(__file__).resolve().parent

EXTENSIONS = [
    ("Extension A — copule TSMOM-Marche (plein echantillon)", "run_extension_A.py"),
    ("Extension B — rolling copula (stabilite temporelle)",   "run_extension_B.py"),
    ("Extension C — dependance de queue par classe d'actifs", "run_extension_C.py"),
]


def _run_extension(label: str, script: str, env: dict) -> bool:
    path = _ROOT / script
    if not path.exists():
        print(f"[skip] {script} introuvable a la racine du projet.")
        return False
    print("\n" + "=" * 64)
    print(f"[ext] {label}\n      ({script})")
    print("=" * 64)
    r = subprocess.run([sys.executable, str(path)], cwd=str(_ROOT), env=env)
    ok = (r.returncode == 0)
    print(f"[ext] {script} -> " + ("OK" if ok else f"ECHEC (code {r.returncode})"))
    return ok


def main():
    ap = argparse.ArgumentParser(
        description="Regenere le coeur de la replication PUIS les extensions A/B/C.")
    ap.add_argument("--start", default=PAPER_START)
    ap.add_argument("--end", default=PAPER_END)
    ap.add_argument("--no-external", action="store_true",
                    help="pas de telechargement (caches/coeur seulement)")
    ap.add_argument("--no-clean", action="store_true",
                    help="ne pas supprimer outputs/ avant de regenerer")
    ap.add_argument("--skip-extensions", action="store_true",
                    help="ne regenerer QUE le coeur (equivalent de l'ancien regenerate.py)")
    ap.add_argument("--skip-core", action="store_true",
                    help="ne regenerer QUE les extensions (le coeur doit deja etre a jour)")
    ap.add_argument("--refilter", action="store_true",
                    help="ext B : refiltrage AR-GARCH-t par fenetre (zero look-ahead, plus lent)")
    ap.add_argument("--ext-b-window", type=int, default=60,
                    help="ext B : taille de la fenetre roulante en mois (defaut 60)")
    args = ap.parse_args()

    # 1) Table rase (sauf si on ne touche pas au coeur, ou --no-clean)
    if not args.no_clean and not args.skip_core and OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
        print(f"[clean] {OUTPUT_DIR} supprime.")

    # 2) Coeur de la replication -> une seule source de verite
    if not args.skip_core:
        run(start=args.start, end=args.end, use_external=not args.no_external)
        print("\n[done] Coeur de la replication regenere.")

    # 3) Extensions
    if not args.skip_extensions:
        series = TABLES_DIR / "diversified_tsmom_series.csv"
        if not series.exists():
            print(f"\n[warn] {series} absent : les extensions A et B vont echouer.")
            print("       Lance d'abord le coeur (sans --skip-core), ou retire --skip-core.")
        env = os.environ.copy()
        env["EXT_B_WINDOW"] = str(args.ext_b_window)
        env["EXT_B_REFILTER"] = "1" if args.refilter else "0"
        results = {script: _run_extension(label, script, env)
                   for label, script in EXTENSIONS}
        print("\n[done] Extensions :",
              ", ".join(f"{s.replace('run_extension_','').replace('.py','')}="
                        + ("OK" if ok else "KO")
                        for s, ok in results.items()))

    print("\n[ALL DONE] Coeur + extensions : tous les outputs proviennent du meme run.")


if __name__ == "__main__":
    main()
