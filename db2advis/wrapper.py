#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""db2advis index-selection skill wrapper (long-lived HTTP server).

Wraps the DB2 Advisor heuristic index-selection algorithm (Valentin et al.,
ICDE 2000) from the XMUDM/Index_EAB repository (a fork of the hyrise
index_selection_evaluation framework) behind the uniform index-selection HTTP
protocol: GET /health, POST /recommend, GET /state, POST /shutdown.

The recommendation path is non-invasive: candidate evaluation uses HypoPG
hypothetical indexes (cost-only EXPLAIN) on --eval-dsn (defaults to --dsn); no
real indexes are created on the main DSN unless --apply is set. db2advis is a
deterministic, cost-based heuristic -- there is no online training, no model
weights, and no actual query execution (EXPLAIN without ANALYZE). State is
limited to cumulative request counters and the last recommendation, persisted
on graceful shutdown and rehydrated on restart.
"""
import sys
import os
import json
import re
import time
import threading
import traceback
import types
import importlib
import importlib.abc
import importlib.util
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

# ---------------------------------------------------------------------------
# Stub unneeded source subtrees BEFORE importing source modules.
#
# cost_evaluation.py unconditionally imports the learned benefit-estimation
# modules (tree_cost_infer / lib_infer / former_infer) which pull in
# torch/sklearn; candidate_generation.py imports psqlparse (a C extension that
# only builds on Python 3.7/3.8) and the dqn Encoding/ParserForIndex helpers.
# None of these are used by the default what-if permutation path db2advis
# exercises. A meta-path finder stubs the whole benefit_estimation and
# dqn_selection subtrees (empty packages + no-op loader functions) and a
# top-level psqlparse stub. This keeps the image light (no torch, no psqlparse
# build) and leaves the source tree untouched (git diff empty).
# ---------------------------------------------------------------------------
_BLOCKED_PREFIXES = (
    "index_advisor_selector.index_benefit_estimation",
    "index_advisor_selector.index_selection.dqn_selection",
)
_STUB_NAMES = (
    "load_model_tree", "get_tree_est_res",
    "load_model_lib", "get_lib_est_res",
    "load_model_former", "get_former_est_res",
)


class _StubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path, target=None):
        for prefix in _BLOCKED_PREFIXES:
            if fullname == prefix or fullname.startswith(prefix + "."):
                return importlib.util.spec_from_loader(fullname, self, is_package=True)
        return None

    def create_module(self, spec):
        m = types.ModuleType(spec.name)
        m.__path__ = []  # mark as a package so deeper imports also resolve to stubs
        m.__package__ = spec.name
        return m

    def exec_module(self, module):
        for name in _STUB_NAMES:
            setattr(module, name, lambda *a, **kw: None)


def _install_stubs():
    sys.meta_path.insert(0, _StubFinder())
    if "psqlparse" not in sys.modules:
        _ps = types.ModuleType("psqlparse")
        _ps.parse_dict = lambda q: []
        sys.modules["psqlparse"] = _ps


_install_stubs()

REPO = os.environ.get("DB2ADVIS_REPO", "/app/Index_EAB")
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import psycopg2  # noqa: E402
from index_advisor_selector.index_selection.heu_selection.heu_utils.postgres_dbms import (  # noqa: E402
    PostgresDatabaseConnector,
)
from index_advisor_selector.index_selection.heu_selection.heu_utils import heu_com  # noqa: E402
from index_advisor_selector.index_selection.heu_selection.heu_utils.workload import (  # noqa: E402
    Workload, Table, Column, Query,
)
from index_advisor_selector.index_selection.heu_selection.heu_utils.cost_evaluation import (  # noqa: E402
    CostEvaluation,
)
from index_advisor_selector.index_selection.heu_selection.heu_algos.db2advis_algorithm import (  # noqa: E402
    DB2AdvisAlgorithm,
)

# SelectionAlgorithm.__init__ calls database_connector.drop_indexes() which
# drops every real non-PK index on the connected database. The skill must
# NEVER mutate the user's main DSN, so neuter it to a no-op (runtime patch,
# no source edit).
PostgresDatabaseConnector.drop_indexes = lambda self: None

DEFAULT_STATE_DIR = os.path.join(REPO, "skill_state")
DEFAULT_MAX_INDEX_WIDTH = 2
DEFAULT_BUDGET_MB = 500.0
DEFAULT_MAX_INDEXES = 15
DEFAULT_TRY_VARIATIONS_SECONDS = 0
DEFAULT_TRY_VARIATIONS_MAX_REMOVALS = 4
_LARGE_BUDGET_MB = 1_000_000.0  # effectively unbounded disk when no budget given

_LITERAL_RE = re.compile(r"'[^']*'")
_LINE_COMMENT_RE = re.compile(r"--[^\n]*")
_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)


def _word_re(name):
    return re.compile(r"(?<![a-z0-9_])" + re.escape(name) + r"(?![a-z0-9_])")


def _strip_sql(text):
    text = _BLOCK_COMMENT_RE.sub(" ", text)
    text = _LINE_COMMENT_RE.sub(" ", text)
    text = _LITERAL_RE.sub(" ", text)
    return text.lower()


def _split_statements(content):
    content = content.strip()
    if not content:
        return []
    out = []
    for part in content.split(";"):
        p = part.strip()
        if p:
            out.append(p)
    return out


_DIRECTIVE_RE = re.compile(r"^:[A-Za-z]$")


def _clean_stmt(text):
    """Strip SQL line comments and dbgen-style directives (:x/:o) so the
    remaining text begins with the actual SQL verb. Returns the cleaned
    statement (comments/directives removed)."""
    kept = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s or s.startswith("--") or _DIRECTIVE_RE.match(s):
            continue
        kept.append(ln)
    return "\n".join(kept).strip()


def _is_read_query(text):
    t = _clean_stmt(text).lstrip().lower()
    return t.startswith("select") or t.startswith("with")


def _parse_dsn(dsn):
    return psycopg2.extensions.parse_dsn(dsn)


def _b_to_mb(b):
    # Match the source's b_to_mb convention (1e6, not 1024**2).
    return (b or 0) / 1_000_000.0


class DB2AdvisServer:
    def __init__(self, state_dir=DEFAULT_STATE_DIR):
        self._lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._connectors = {}        # dsn -> PostgresDatabaseConnector
        self._schema_cache = {}      # dsn -> (tables, columns, existing)
        self._table_size_cache = {}  # dsn -> {table_name: bytes}
        self._col_types_cache = {}    # dsn -> {(table,col): pg_type}
        self._max_bytes_cache = {}    # (dsn,table,col) -> max octet_length
        self.state_dir = state_dir
        self._counters = {
            "total_requests": 0,
            "total_queries_analyzed": 0,
            "last_workload_size": 0,
            "last_num_indexes": 0,
            "last_optimization_time": 0.0,
            "last_estimated_impact": 0.0,
        }
        self._last_recommendation = []
        self._last_config = {}
        self._shutdown = False
        self._load_state()

    # -- persistence --------------------------------------------------------
    def _state_files(self):
        return (
            os.path.join(self.state_dir, "counters.json"),
            os.path.join(self.state_dir, "last_recommendation.json"),
        )

    def _load_state(self):
        try:
            cf, rf = self._state_files()
            if os.path.isfile(cf):
                with open(cf) as fh:
                    saved = json.load(fh)
                for k, v in saved.items():
                    if k in self._counters:
                        self._counters[k] = v
            if os.path.isfile(rf):
                with open(rf) as fh:
                    self._last_recommendation = json.load(fh)
        except Exception:
            pass  # fresh start on any corruption

    def _persist_state(self):
        try:
            os.makedirs(self.state_dir, exist_ok=True)
            cf, rf = self._state_files()
            tmp_cf = cf + ".tmp"
            with open(tmp_cf, "w") as fh:
                json.dump(self._counters, fh, indent=2)
            os.replace(tmp_cf, cf)
            tmp_rf = rf + ".tmp"
            with open(tmp_rf, "w") as fh:
                json.dump(self._last_recommendation, fh, indent=2)
            os.replace(tmp_rf, rf)
        except Exception:
            pass

    # -- connections & schema ---------------------------------------------
    def _get_connector(self, dsn):
        conn = self._connectors.get(dsn)
        if conn is not None:
            try:
                cur = conn._cursor
                cur.execute("select 1")
                cur.fetchone()
                return conn
            except Exception:
                try:
                    conn.close()
                except Exception:
                    pass
                self._connectors.pop(dsn, None)
                self._schema_cache.pop(dsn, None)
                self._table_size_cache.pop(dsn, None)
        d = _parse_dsn(dsn)
        cfg = {"postgresql": d}
        conn = PostgresDatabaseConnector(
            cfg, autocommit=True,
            host=d.get("host"), port=str(d.get("port")),
            db_name=d.get("dbname"), user=d.get("user"),
            password=d.get("password"),
        )
        try:
            conn._cursor.execute("CREATE EXTENSION IF NOT EXISTS hypopg")
        except Exception:
            pass  # surface later if a what-if call actually fails
        conn.drop_hypo_indexes()
        self._connectors[dsn] = conn
        return conn

    def _introspect(self, dsn):
        cached = self._schema_cache.get(dsn)
        if cached is not None:
            return cached
        conn = self._get_connector(dsn)
        tables, columns = heu_com.get_columns_from_db(conn)
        existing = self._existing_indexes(conn)
        sizes = self._table_sizes(conn, tables)
        col_types = self._col_types(conn)
        self._schema_cache[dsn] = (tables, columns, existing)
        self._table_size_cache[dsn] = sizes
        self._col_types_cache[dsn] = col_types
        return tables, columns, existing

    def _existing_indexes(self, conn):
        """Return (set of (table, tuple(cols)), set of index_names)."""
        cols_set, names = set(), set()
        try:
            conn._cursor.execute(
                "select schemaname, tablename, indexname, indexdef "
                "from pg_indexes where schemaname = 'public'"
            )
            rows = conn._cursor.fetchall()
        except Exception:
            return cols_set, names
        for _schema, table, name, indexdef in rows:
            names.add(name.lower())
            m = re.search(r"\(([^)]*)\)\s*$", indexdef or "")
            if m:
                raw = m.group(1)
                colparts = []
                for tok in raw.split(","):
                    tok = tok.strip()
                    mm = re.match(r"([A-Za-z_][A-Za-z0-9_]*)", tok)
                    if mm:
                        colparts.append(mm.group(1).lower())
                if colparts:
                    cols_set.add((table.lower(), tuple(colparts)))
        return cols_set, names

    def _table_sizes(self, conn, tables):
        sizes = {}
        for t in tables:
            try:
                conn._cursor.execute(
                    "select pg_total_relation_size(%s)", (t.name,)
                )
                sizes[t.name] = conn._cursor.fetchone()[0]
            except Exception:
                sizes[t.name] = 0
        return sizes

    def _col_types(self, conn):
        types = {}
        try:
            conn._cursor.execute(
                "select table_name, column_name, data_type, character_maximum_length "
                "from information_schema.columns where table_schema = 'public'"
            )
            for tbl, col, dt, cml in conn._cursor.fetchall():
                types[(tbl.lower(), col.lower())] = (dt, cml)
        except Exception:
            pass
        return types

    _BTREE_KEY_LIMIT = 2700  # PG btree index-row safety threshold (~8191 B block)

    def _max_bytes(self, dsn, table, col):
        """Cached max octet_length of a column, for buildability probing."""
        key = (dsn, table, col)
        if key in self._max_bytes_cache:
            return self._max_bytes_cache[key]
        conn = self._get_connector(dsn)
        val = 0
        try:
            conn._cursor.execute("set statement_timeout = 60000")
            conn._cursor.execute(
                'select max(octet_length("%s")) from "%s"' % (col, table)
            )
            row = conn._cursor.fetchone()
            val = (row[0] or 0) if row else 0
        except Exception:
            # Timeout / error -> assume large (conservative skip) so we never
            # emit a DDL that will fail at CREATE INDEX time.
            val = self._BTREE_KEY_LIMIT + 1
            try:
                conn.rollback()
            except Exception:
                pass
        finally:
            try:
                conn._cursor.execute("set statement_timeout = 0")
            except Exception:
                pass
        self._max_bytes_cache[key] = val
        return val

    def _index_buildable(self, dsn, table, cols, col_types):
        """Return (buildable, reason). Skip indexes whose key columns include an
        unbounded text/bytea/json column with a value exceeding the btree
        index-row limit (CREATE INDEX would otherwise fail at apply time)."""
        for c in cols:
            dtype, cml = col_types.get((table, c), (None, None))
            if dtype in ("text", "bytea", "json", "xml") or (
                dtype == "character varying" and cml is None
            ):
                if self._max_bytes(dsn, table, c) > self._BTREE_KEY_LIMIT:
                    return False, (
                        f"index on {table}({','.join(cols)}) has unbounded "
                        f"{dtype} column '{c}' exceeding btree key-size limit; skipped"
                    )
        return True, None

    # -- workload construction --------------------------------------------
    def _populate_columns(self, text, columns):
        stripped = _strip_sql(text)
        matched = []
        for col in columns:
            tn = col.table.name
            if _word_re(tn).search(stripped) and _word_re(col.name).search(stripped):
                matched.append(col)
        return matched

    def _build_workload(self, workload_field, columns):
        """Return (Workload, num_analyzed, notes)."""
        entries = self._normalize_workload(workload_field)
        queries = []
        notes = []
        analyzed = 0
        for idx, (sql, weight) in enumerate(entries, start=1):
            for stmt in _split_statements(sql):
                stmt = _clean_stmt(stmt)
                if not stmt:
                    continue
                if not _is_read_query(stmt):
                    notes.append(f"Q{idx}: non-SELECT excluded from read-cost")
                    continue
                cols = self._populate_columns(stmt, columns)
                if not cols:
                    notes.append(f"Q{idx}: no schema columns matched (absent table or unparseable)")
                    continue
                q = Query(idx, stmt, columns=cols, frequency=max(1, int(weight or 1)))
                queries.append(q)
                analyzed += 1
        if not queries:
            return None, 0, notes
        return Workload(queries), analyzed, notes

    def _normalize_workload(self, workload_field):
        """Return list of (sql_text, weight)."""
        if workload_field is None:
            return []
        if isinstance(workload_field, list):
            out = []
            for item in workload_field:
                if isinstance(item, dict):
                    out.append((item.get("sql", ""), item.get("weight", item.get("frequency", 1))))
                elif isinstance(item, str):
                    out.append((item, 1))
            return out
        if isinstance(workload_field, str):
            s = workload_field.strip()
            if not s:
                return []
            if s.startswith("["):
                try:
                    parsed = json.loads(s)
                    return self._normalize_workload(parsed)
                except Exception:
                    pass
            if os.path.isfile(s):
                try:
                    with open(s) as fh:
                        return [(fh.read(), 1)]
                except Exception:
                    pass
            return [(s, 1)]
        return []

    # -- recommendation ----------------------------------------------------
    def recommend(self, req):
        t_start = time.time()
        degrade_notes = []
        dsn = req.get("dsn")
        if not dsn:
            return self._degrade("missing --dsn", [], {}, 0)
        workload_field = req.get("workload")
        eval_dsn = req.get("eval_dsn") or dsn
        budget = req.get("budget") or {}
        cfg = self._parse_config(req.get("config"))
        apply_flag = bool(req.get("apply"))

        with self._lock:
            self._last_config = cfg
            try:
                eval_conn = self._get_connector(eval_dsn)
            except Exception as e:
                return self._degrade(f"cannot connect to eval-dsn: {e}", [], cfg, 0,
                                     dsn=dsn, eval_dsn=eval_dsn)
            try:
                _, columns, _ = self._introspect(dsn)
                # Re-read the existing-index set on every request: it is a
                # single cheap catalog SELECT, and a cached snapshot can go
                # stale when indexes are created/dropped on the DSN between
                # requests (e.g. a previous --apply) -- a stale snapshot made
                # later requests skip their own recommendations as "already
                # exists".
                existing = self._existing_indexes(self._get_connector(dsn))
            except Exception as e:
                return self._degrade(f"schema introspection failed: {e}", [], cfg, 0,
                                     dsn=dsn, eval_dsn=eval_dsn)

            workload, num_analyzed, wnotes = self._build_workload(workload_field, columns)
            degrade_notes.extend(wnotes)
            if workload is None:
                return self._degrade("no parseable SELECT query with indexable columns",
                                      [], cfg, 0, notes=degrade_notes, dsn=dsn, eval_dsn=eval_dsn)

            params, cap_count, write_pct = self._build_params(cfg, budget)
            try:
                eval_conn.drop_hypo_indexes()
                algo = DB2AdvisAlgorithm(
                    eval_conn, params, process=False,
                    cand_gen=None, is_utilized=None, sel_oracle=cfg.get("sel_oracle"),
                )
                indexes = algo.calculate_best_indexes(workload)
                eval_conn.drop_hypo_indexes()
            except Exception as e:
                traceback.print_exc()
                eval_conn.drop_hypo_indexes()
                return self._degrade(f"algorithm failure: {e}", [], cfg, num_analyzed,
                                     notes=degrade_notes, dsn=dsn, eval_dsn=eval_dsn)

            table_sizes = self._table_size_cache.get(dsn, {})
            col_types = self._col_types_cache.get(dsn, {})
            rec_list = self._to_ddl(indexes, existing, table_sizes, col_types, dsn,
                                    cap_count, write_pct, degrade_notes)

            impact = self._estimate_impact(eval_conn, workload, indexes)
            dt = time.time() - t_start
            consumed_storage = sum(r.get("storage_mb") or 0 for r in rec_list)
            consumed_write = max([r.get("write_overhead_estimate") or 0 for r in rec_list] + [0.0])
            budget_used = {
                "storage_mb": budget.get("storage_mb"),
                "max_indexes": budget.get("max_indexes"),
                "max_write_overhead_pct": budget.get("max_write_overhead_pct"),
                "consumed_storage_mb": round(consumed_storage, 4),
                "consumed_indexes": len(rec_list),
                "consumed_write_overhead_pct": round(consumed_write, 4),
            }
            if not rec_list and not any("storage" in n or "budget" in n for n in degrade_notes):
                degrade_notes.append("no beneficial index within budget")

            response = {
                "recommended_indexes": rec_list,
                "metadata": {
                    "strategy_type": "cost-based",
                    "optimization_time": round(dt, 4),
                    "estimated_impact": round(impact, 4),
                    "num_queries_analyzed": num_analyzed,
                    "budget_used": budget_used,
                    "eval_mode": "standby" if eval_dsn != dsn else "hypothetical",
                    "mode": "inference-only",
                    "config": {k: cfg[k] for k in sorted(cfg)},
                    "degrade_note": "; ".join(degrade_notes) if degrade_notes else None,
                },
            }
            if apply_flag:
                response["apply_results"] = self._apply_indexes(rec_list, dsn)

            with self._state_lock:
                self._counters["total_requests"] += 1
                self._counters["total_queries_analyzed"] += num_analyzed
                self._counters["last_workload_size"] = num_analyzed
                self._counters["last_num_indexes"] = len(rec_list)
                self._counters["last_optimization_time"] = round(dt, 4)
                self._counters["last_estimated_impact"] = round(impact, 4)
                self._last_recommendation = rec_list
            return response

    # -- config & params ---------------------------------------------------
    def _parse_config(self, config_field):
        if not config_field:
            return {}
        if isinstance(config_field, dict):
            return dict(config_field)
        if isinstance(config_field, str):
            s = config_field.strip()
            if not s:
                return {}
            try:
                parsed = json.loads(s)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass
            return {"state_dir": s}
        return {}

    def _build_params(self, cfg, budget):
        max_index_width = int(cfg.get("max_index_width", DEFAULT_MAX_INDEX_WIDTH))
        budget_mb = float(cfg.get("budget_MB", DEFAULT_BUDGET_MB))
        max_indexes = int(cfg.get("max_indexes", DEFAULT_MAX_INDEXES))
        try_var = float(cfg.get("try_variations_seconds", DEFAULT_TRY_VARIATIONS_SECONDS))
        try_rem = int(cfg.get("try_variations_max_removals", DEFAULT_TRY_VARIATIONS_MAX_REMOVALS))

        storage_mb = budget.get("storage_mb")
        max_indexes_b = budget.get("max_indexes")
        write_pct = budget.get("max_write_overhead_pct")

        cap_count = None
        if storage_mb is not None:
            constraint = "storage"
            disk_mb = float(storage_mb)
            cap_count = int(max_indexes_b) if max_indexes_b is not None else max_indexes
        elif max_indexes_b is not None:
            constraint = "number"
            disk_mb = budget_mb
            max_indexes = int(max_indexes_b)
            cap_count = None
        else:
            constraint = "storage"
            disk_mb = _LARGE_BUDGET_MB
            cap_count = max_indexes

        params = {
            "max_index_width": max_index_width,
            "budget_MB": disk_mb,
            "max_indexes": max_indexes,
            "constraint": constraint,
            "try_variations_seconds": try_var,
            "try_variations_max_removals": try_rem,
        }
        return params, cap_count, write_pct

    # -- index -> DDL ------------------------------------------------------
    def _to_ddl(self, indexes, existing, table_sizes, col_types, dsn,
                cap_count, write_pct, notes):
        existing_cols, existing_names = existing
        used_names = set()
        rec = []
        for ind in indexes:
            table = ind.table().name
            cols = [c.name for c in ind.columns]
            key = (table, tuple(cols))
            if key in existing_cols:
                notes.append(f"index on {table}({','.join(cols)}) already exists; skipped")
                continue
            buildable, breason = self._index_buildable(dsn, table, cols, col_types)
            if not buildable:
                notes.append(breason)
                continue
            base = ind.index_idx()
            name = base
            n = 2
            while name.lower() in existing_names or name.lower() in used_names:
                name = f"{base}_{n}"
                n += 1
            used_names.add(name.lower())
            storage_mb = _b_to_mb(ind.estimated_size or 0)
            tsize = table_sizes.get(table, 0)
            ratio = (storage_mb / (_b_to_mb(tsize) or 1.0)) if tsize else 0.0
            write_overhead = min(100.0, 1.0 + ratio * 5.0)
            rec.append({
                "ddl": f"CREATE INDEX {name} ON {table} ({','.join(cols)})",
                "table": table,
                "columns": cols,
                "index_name": name,
                "storage_mb": round(storage_mb, 4),
                "write_overhead_estimate": round(write_overhead, 4),
            })
            if cap_count is not None and len(rec) >= cap_count:
                break
        if write_pct is not None:
            kept = []
            for r in rec:
                if (r["write_overhead_estimate"] or 0) <= float(write_pct):
                    kept.append(r)
                else:
                    notes.append(
                        f"{r['index_name']} write_overhead {r['write_overhead_estimate']}% "
                        f"exceeds budget {write_pct}%; dropped")
            rec = kept
        return rec

    def _estimate_impact(self, eval_conn, workload, indexes):
        try:
            ce = CostEvaluation(eval_conn)
            initial = ce.calculate_cost(workload, [])
            final = ce.calculate_cost(workload, list(indexes))
            ce.complete_cost_estimation()
            eval_conn.drop_hypo_indexes()
            if initial and initial > 0:
                return (initial - final) / initial * 100.0
            return 0.0
        except Exception:
            try:
                eval_conn.drop_hypo_indexes()
            except Exception:
                pass
            return 0.0

    def _apply_indexes(self, rec_list, dsn):
        results = []
        try:
            conn = self._get_connector(dsn)
        except Exception as e:
            return [{"index_name": r["index_name"], "status": "error",
                     "error": f"connect failed: {e}"} for r in rec_list]
        for r in rec_list:
            try:
                conn._cursor.execute("DROP INDEX IF EXISTS %s" % r["index_name"])
                conn._cursor.execute(r["ddl"])
                results.append({"index_name": r["index_name"], "status": "ok", "error": None})
            except Exception as e:
                conn.rollback()
                results.append({"index_name": r["index_name"], "status": "error",
                                 "error": str(e)})
        return results

    # -- degrade & state ---------------------------------------------------
    def _degrade(self, reason, rec_list, cfg, num_analyzed, notes=None,
                 dsn=None, eval_dsn=None):
        notes = notes or []
        notes.append(reason)
        return {
            "recommended_indexes": rec_list,
            "metadata": {
                "strategy_type": "cost-based",
                "optimization_time": 0.0,
                "estimated_impact": 0.0,
                "num_queries_analyzed": num_analyzed,
                "budget_used": {"storage_mb": None, "max_indexes": None,
                                "max_write_overhead_pct": None,
                                "consumed_storage_mb": 0.0, "consumed_indexes": 0,
                                "consumed_write_overhead_pct": 0.0},
                "eval_mode": "standby" if (eval_dsn and dsn and eval_dsn != dsn) else "hypothetical",
                "mode": "inference-only",
                "config": {k: cfg[k] for k in sorted(cfg)} if cfg else {},
                "degrade_note": "; ".join(notes),
            },
        }

    def state(self):
        with self._state_lock:
            counters = dict(self._counters)
            last_rec = list(self._last_recommendation)
            cfg = dict(self._last_config)
        return {
            "idle": True,  # no background work for a deterministic heuristic
            "model_loaded": True,
            "training_in_progress": False,
            "counters": counters,
            "last_recommendation": last_rec,
            "active_config": cfg,
            "state_dir": self.state_dir,
            "strategy_type": "cost-based",
        }

    def shutdown(self):
        self._shutdown = True
        with self._state_lock:
            self._persist_state()
        for dsn, conn in list(self._connectors.items()):
            try:
                conn.drop_hypo_indexes()
            except Exception:
                pass
            try:
                conn.close()
            except Exception:
                pass
        self._connectors.clear()


# ---------------------------------------------------------------------------
# HTTP layer
# ---------------------------------------------------------------------------
_SERVER = {}


def _make_handler(server_obj):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            pass

        def _send(self, code, obj):
            body = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.startswith("/health"):
                self._send(200, {"status": "ok", "strategy": "db2advis"})
                return
            if self.path.startswith("/state"):
                self._send(200, server_obj.state())
                return
            self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path.startswith("/recommend"):
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b""
                try:
                    req = json.loads(raw.decode("utf-8") or "{}")
                except Exception as e:
                    self._send(200, server_obj._degrade(f"invalid JSON body: {e}", [], {}, 0))
                    return
                try:
                    resp = server_obj.recommend(req)
                except Exception as e:
                    traceback.print_exc()
                    resp = server_obj._degrade(f"internal error: {e}", [], {}, 0)
                self._send(200, resp)
                return
            if self.path.startswith("/shutdown"):
                self._send(200, {"status": "shutting down"})
                threading.Thread(target=server_obj.shutdown, daemon=True).start()
                threading.Thread(
                    target=lambda: (time.sleep(0.5), _SERVER["httpd"].shutdown()),
                    daemon=True).start()
                return
            self._send(404, {"error": "not found"})

    return Handler


def main():
    port = int(os.environ.get("PORT", "8765"))
    state_dir = os.environ.get("DB2ADVIS_STATE_DIR", DEFAULT_STATE_DIR)
    server_obj = DB2AdvisServer(state_dir=state_dir)
    httpd = ThreadingHTTPServer(("0.0.0.0", port), _make_handler(server_obj))
    httpd.daemon_threads = True
    _SERVER["httpd"] = httpd
    print(f"db2advis wrapper listening on {port}", flush=True)
    try:
        httpd.serve_forever()
    finally:
        server_obj.shutdown()


if __name__ == "__main__":
    main()
