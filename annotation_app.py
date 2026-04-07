"""
Annotation App — Shiny for Python
- Server-side CSV path input (no upload widget)
- Autosave: every annotation change writes immediately to the dataframe AND disk
- Manual "Save snapshot" writes a timestamped copy to avoid overwriting the original
"""

import re
import base64
import io
from datetime import datetime
from pathlib import Path
from typing import Optional, List

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shiny import App, reactive, render, ui

# ── helpers ───────────────────────────────────────────────────────────────────

def img_to_b64(path: str) -> Optional[str]:
    p = Path(path)
    if p.exists():
        ext = p.suffix.lower().lstrip(".")
        mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(ext, "image/png")
        data = base64.b64encode(p.read_bytes()).decode()
        return f"data:{mime};base64,{data}"
    return None


def build_labelbee_url(video: str, frame, tagid=None, video_data_col: Optional[str] = None) -> str:
    base = "http://136.145.54.85/webapp-test/labelbee/gui"
    if video_data_col and str(video_data_col).strip():
        vd = str(video_data_col).strip()
    else:
        vd = re.sub(r"/mp4/", "/tracks/", str(video)) + ".tracks_tagged.csv"
    frame_val = int(frame) if pd.notna(frame) else 0
    url = f"{base}#video_id=/datasets/gurabo4/mp4/col10/{video}&frame={frame_val}&tracks_tagged_data=/datasets/gurabo4/tracks/col10/{vd}&frame={frame}"
    if (tagid is not None):
        url+=f"&tagid={tagid}"
    return url


def timestamped_path(original: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return original.parent / f"{original.stem}_{ts}{original.suffix}"


def write_df(df: pd.DataFrame, path: Path):
    df.to_csv(path, index=False)

def remove_suffix(input_string, suffix):
    if suffix and input_string.endswith(suffix):
        return input_string[:-len(suffix)]
    return input_string

from functools import lru_cache
import threading

class TrajectoryRenderer:
    def __init__(self):
        self._fdf = None

        @lru_cache(maxsize=16)
        def _render_cached(event_id):
            if self._fdf is None:
                return None
            return self._make_trajectory_plot_b64(event_id)

        self._render_cached = _render_cached

    def set_fdf(self, fdf):
        self._fdf = fdf
        self._render_cached.cache_clear()

    def render(self, event_id):
        return self._render_cached(event_id)
    
    def _make_trajectory_plot_b64(self, event_id) -> Optional[str]:
        """Render the trajectory matplotlib figure for a given group_id and return as base64 PNG."""
        #edf = fdf[fdf["group_id"] == event_id]

        gdf = self._fdf

        edf = gdf.get_group(event_id)
        #if edf.empty:
        if (edf is None):
            return None

        print(f"Plotting {event_id}")

        track_tagid = edf.iloc[0]["track_tagid"]
        video = remove_suffix( str(edf.iloc[0]["video"]), ".cfr.mp4" )
        first_frame = edf["frame"].min()
        last_frame = edf["frame"].max()

        fig, ax = plt.subplots(2, 1, figsize=(6, 8))
        fig.patch.set_facecolor("#0d1117")
        for a in ax:
            a.set_facecolor("#161b22")
            a.tick_params(colors="#c9d1d9")
            a.xaxis.label.set_color("#c9d1d9")
            a.yaxis.label.set_color("#c9d1d9")
            a.title.set_color("#58a6ff")
            for spine in a.spines.values():
                spine.set_edgecolor("#30363d")

        # ── top panel: frame vs cy ────────────────────────────────────────────────
        #plt.sca(ax[0])
        cax = ax[0]
        cax.hlines([0, 1440], first_frame, last_frame, colors="#484f58", linestyles="--")
        for track_id, tdf in edf.groupby("track_id"):
            cax.plot(tdf["frame"], tdf["cy"], ".", label=f"trk {track_id}", markersize=4)
        cax.set_xlim(first_frame, last_frame)
        cax.set_ylim(-200, 1640)
        for _, row in edf[~edf["tag_id"].isna()].iterrows():
            tag_id = row["tag_id"]
            col = "#3fb950" if tag_id == track_tagid else "#f85149"
            cax.text(row["frame"], row["cy"] + 40, str(int(tag_id)),
                    fontsize=8, horizontalalignment="center", color=col)
        cax.legend(fontsize=7, facecolor="#21262d", edgecolor="#30363d", labelcolor="#c9d1d9")
        cax.set_title(f"event={event_id}  tag={track_tagid}  video={video}", fontsize=9)
        cax.set_xlabel("frame")
        cax.set_ylabel("cy")
        cax.set_yticks([0, 200, 400, 600, 800, 1000, 1200, 1440])
        cax.text(first_frame, -100, "outside",  ha="left", va="center", color="#c9d1d9", fontweight="bold", fontsize=8)
        cax.text(first_frame,  1540, "inside", ha="left", va="center", color="#c9d1d9", fontweight="bold", fontsize=8)

        # ── bottom panel: cx vs cy ────────────────────────────────────────────────
        #plt.sca(ax[1])
        cax = ax[1]
        cax.hlines([0, 1440], 0, 2560, colors="#484f58", linestyles="--")
        for track_id, tdf in edf.groupby("track_id"):
            plt.plot(tdf["cx"], tdf["cy"], ".", label=f"trk {track_id}", markersize=4)
        cax.set_xlim(0, 2560)
        cax.set_ylim(-200, 1640)
        for _, row in edf[~edf["tag_id"].isna()].iterrows():
            tag_id = row["tag_id"]
            col = "#3fb950" if tag_id == track_tagid else "#f85149"
            cax.text(row["cx"], row["cy"] + 40, str(int(tag_id)),
                    fontsize=8, horizontalalignment="center", color=col)
        cax.legend(fontsize=7, facecolor="#21262d", edgecolor="#30363d", labelcolor="#c9d1d9")
        cax.set_title(f"event={event_id}  tag={track_tagid}  video={video}", fontsize=9)
        cax.set_xlabel("cx")
        cax.set_ylabel("cy")
        cax.set_yticks([0, 200, 400, 600, 800, 1000, 1200, 1440])
        cax.text(0, -100,  "outside",  ha="left", va="center", color="#c9d1d9", fontweight="bold", fontsize=8)
        cax.text(0,  1540, "inside", ha="left", va="center", color="#c9d1d9", fontweight="bold", fontsize=8)

        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode()
        return f"data:image/png;base64,{b64}"




# ── UI ────────────────────────────────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; }
body {
    background: #0d1117; color: #c9d1d9;
    font-family: 'IBM Plex Mono', monospace; font-size: 13px;
    margin: 0; padding: 0;
}
.card            { background: #161b22; border: 1px solid #30363d; border-radius: 8px; margin-bottom: .75rem; }
.card-header     { background: #21262d; border-bottom: 1px solid #30363d;
                   color: #58a6ff; font-weight: 700; letter-spacing:.06em;
                   padding: .5rem .9rem; border-radius: 8px 8px 0 0; font-size:.85rem; }
.card-body { max-height: none !important; overflow: visible !important; }
.ann-label       { color: #8b949e; font-size:.75rem; text-transform:uppercase;
                   letter-spacing:.08em; margin: .6rem 0 .2rem; }
.ann-btn-group   { display: flex; flex-wrap: wrap; gap: 5px; margin-bottom: .3rem; }
.ann-btn         { padding: 3px 11px; font-size:.78rem; font-family:inherit;
                   border: 1px solid #30363d; border-radius: 5px;
                   background: #21262d; color: #c9d1d9; cursor: pointer;
                   transition: background .12s, border-color .12s; }
.ann-btn:hover   { background: #2d333b; border-color: #58a6ff; }
.ann-btn.active  { background: #1f6feb !important; border-color: #58a6ff !important;
                   color: #fff !important; font-weight: 600; }
textarea.form-control, input.form-control {
    background: #0d1117 !important; color: #c9d1d9 !important;
    border: 1px solid #30363d !important; border-radius: 5px !important;
    font-family: inherit !important; font-size: .8rem !important;
}
textarea.form-control:focus, input.form-control:focus {
    border-color: #58a6ff !important; box-shadow: none !important;
}
.nav-btn         { padding: 5px 14px; font-size:.8rem; font-family:inherit;
                   border-radius: 5px; border: 1px solid; cursor: pointer;
                   transition: opacity .12s; }
.nav-btn:hover   { opacity:.82; }
.btn-prev        { background:#bb8009; border-color:#d29922; color:#0d1117; }
.btn-next        { background:#bb8009; border-color:#d29922; color:#0d1117; }
.btn-snap        { background:#238636; border-color:#2ea043; color:#fff; }
.btn-load        { background:#1f6feb; border-color:#388bfd; color:#fff; }
.img-frame       { background:#010409; border: 1px solid #21262d; border-radius:6px;
                   min-height:300px; display:flex; align-items:center;
                   justify-content:center; overflow:hidden; }
.img-frame img   { width:100%; height:auto; object-fit: contain; }
.status-ok       { color:#3fb950; }
.status-err      { color:#f85149; }
.status-info     { color:#58a6ff; }
.status-bar      { background:#161b22; border:1px solid #30363d; border-radius:6px;
                   padding:.35rem .75rem; font-size:.75rem; margin-top:.4rem; }
.autosave-pulse  { display:inline-block; width:8px; height:8px; border-radius:50%;
                   background:#3fb950; margin-right:5px;
                   animation: pulse 1s ease-out 1; }
@keyframes pulse {
    0%   { box-shadow: 0 0 0 0 rgba(63,185,80,.7); }
    70%  { box-shadow: 0 0 0 8px rgba(63,185,80,0); }
    100% { box-shadow: 0 0 0 0 rgba(63,185,80,0); }
}
.path-row        { display:flex; gap:8px; align-items:flex-end; flex-wrap:wrap; }
.path-row .form-group { flex:1; min-width:180px; margin:0; }
.shiny-input-container > label { color: #8b949e !important; font-size:.75rem !important; }
.ann-scroll      { max-height: 82vh; overflow-y: auto; padding: .6rem .8rem; }
.lb-btn          { display:block; width:100%; margin-top:.6rem; padding:5px;
                   background:#0d419d; border:1px solid #1f6feb; border-radius:5px;
                   color:#58a6ff; text-align:center; font-size:.78rem;
                   text-decoration:none; }
.lb-btn:hover    { background:#1f6feb; color:#fff; }
.lb-url          { word-break:break-all; color:#484f58; font-size:.68rem;
                   margin-top:.3rem; line-height:1.3; }
.ft-input        { background:#0d1117; color:#e6c97a; border:1px solid #30363d;
                   border-radius:5px; font-family:inherit; font-size:.75rem;
                   padding:2px 7px; width:80px; height:26px; outline:none;
                   margin-left:4px; }
.ft-input:focus  { border-color:#58a6ff; }
.ann-row         { margin-bottom:.1rem; }
.fdf-tag         { color:#3fb950; font-size:.72rem; margin-left:6px; }
.fdf-none        { color:#484f58; font-size:.72rem; margin-left:6px; }
"""


def ann_buttons(field: str, choices: List[str], current: str = ""):
    """Render annotation toggle buttons + inline free-text showing current value."""
    input_id = f"ft_{field}"
    return ui.div(
        ui.div(
            *[
                ui.tags.button(
                    c,
                    id=f"ann_{field}_{c}",
                    class_=f"ann-btn{'  active' if c == current else ''}",
                    onclick=(
                        f"Shiny.setInputValue('ann_click',"
                        f"{{field:'{field}',value:'{c}',ts:Date.now()}},"
                        f"{{priority:'event'}});"
                    ),
                )
                for c in choices
            ],
            ui.tags.input(
                id=input_id,
                type="text",
                value=current,
                placeholder="—",
                class_="ft-input",
                oninput=(
                    f"Shiny.setInputValue('ann_freetext',"
                    f"{{field:'{field}',value:this.value,ts:Date.now()}},"
                    f"{{priority:'event'}});"
                ),
            ),
            class_="ann-btn-group",
            id=f"anngrp_{field}",
        ),
        class_="ann-row",
    )


app_ui = ui.page_fluid(
    ui.tags.head(ui.tags.style(CSS)),
    ui.div(
        # ── TOP BAR ──────────────────────────────────────────────────────────
        ui.card(
            ui.card_header("🔬 Bee events annotations"),
            ui.div(
                ui.div(
                    # row 1: main CSV
                    ui.div(
                        ui.div(
                            ui.input_text(
                                "csv_path",
                                "CSV path on server",
                                placeholder="5seconds_remi.csv",
                                value="5seconds_remi.csv",
                                width="100%",
                            ),
                            class_="form-group",
                        ),
                        ui.div(
                            ui.tags.button("Load", id="btn_load", class_="nav-btn btn-load"),
                            style="padding-bottom:2px;",
                        ),
                        ui.div(
                            ui.tags.button("💾 Save snapshot", id="btn_snap", class_="nav-btn btn-snap"),
                            style="padding-bottom:2px;",
                        ),
                        class_="path-row",
                    ),
                    # row 2: tracks_tagged_ann CSV (optional)
                    ui.div(
                        ui.div(
                            ui.input_text(
                                "fdf_path",
                                "tracks_tagged_ann CSV (optional — enables trajectory plot)",
                                placeholder="tracks_tagged_for_ann.csv",
                                value="5seconds_tracks_tagged_for_ann.csv",
                                width="100%",
                            ),
                            class_="form-group",
                        ),
                        ui.div(
                            ui.tags.button("Load fdf", id="btn_load_fdf", class_="nav-btn btn-load"),
                            style="padding-bottom:2px;",
                        ),
                        class_="path-row",
                        style="margin-top:.45rem;",
                    ),
                ),
                ui.output_ui("status_ui"),
                style="padding:.6rem .9rem .7rem;",
            ),
            ui.div(
                ui.output_ui("group_selector_ui"),
                style="display:flex; gap:8px; margin-top:.5rem;",
            ),
        ),
        # wire native HTML buttons → Shiny inputs
        ui.tags.script("""
            ['btn_load','btn_prev','btn_next','btn_snap','btn_load_fdf'].forEach(function(id){
                document.addEventListener('click', function(e){
                    if(e.target && e.target.id === id){
                        var cur = Shiny.shinyapp.$inputValues[id] || 0;
                        Shiny.setInputValue(id, cur + 1, {priority:'event'});
                    }
                });
            });

            var EVENT_KEYS = {'1':'D', '2':'U', '3':'DU', '4':'UD'};
            document.addEventListener('keydown', function(e){
                var tag = document.activeElement ? document.activeElement.tagName : '';
                if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
                if (e.metaKey || e.ctrlKey || e.altKey) return;
                if (EVENT_KEYS[e.key]) {
                    e.preventDefault();
                    var btn = document.getElementById('ann_event_' + EVENT_KEYS[e.key]);
                    if (btn) btn.click();
                }
                if (e.key === 'ArrowLeft')  { e.preventDefault(); document.getElementById('btn_prev').click(); }
                if (e.key === 'ArrowRight') { e.preventDefault(); document.getElementById('btn_next').click(); }
            });
        """),
        # ── MAIN AREA ────────────────────────────────────────────────────────
        ui.layout_columns(
            ui.card(
                ui.card_header("Image Preview"),
                ui.div(ui.output_ui("img_display"), class_="img-frame"),
                ui.output_ui("labelbee_ui"),
                style="padding:.6rem;",
            ),
            ui.card(
                ui.card_header("Annotations", ui.output_ui("group_id_text")),
                ui.div(
                    ui.tags.button("◀ Prev", id="btn_prev", class_="nav-btn btn-prev"),
                    ui.tags.button("Next ▶", id="btn_next", class_="nav-btn btn-next"),
                    
                ),
                ui.output_ui("ann_panel"),
                ui.div(
                    ui.div("comment", class_="ann-label"),
                    ui.input_text_area("comment", None, placeholder="Enter comment…",
                                       rows=2, width="100%"),
                    ui.div("to_review", class_="ann-label"),
                    ui.input_text_area("to_review", None, placeholder="Review notes…",
                                       rows=2, width="100%"),
                    style="padding: 0 .8rem .8rem;",
                ),
            ),
            col_widths=[6, 6],
        ),
        style="max-width:1400px; margin:auto; padding:.75rem;",
    ),
)




# ── Server ────────────────────────────────────────────────────────────────────

import functools
import time
def debounce(delay_secs):
    def wrapper(f):
        when = reactive.Value(None)
        trigger = reactive.Value(0)

        @reactive.Calc
        def cached():
            """
            Just in case f isn't a reactive calc already, wrap it in one. This ensures
            that f() won't execute any more than it needs to.
            """
            return f()

        @reactive.Effect(priority=102)
        def primer():
            """
            Whenever cached() is invalidated, set a new deadline for when to let
            downstream know--unless cached() invalidates again
            """
            try:
                cached()
            except Exception:
                ...
            finally:
                when.set(time.time() + delay_secs)

        @reactive.Effect(priority=101)
        def timer():
            """
            Watches changes to the deadline and triggers downstream if it's expired; if
            not, use invalidate_later to wait the necessary time and then try again.
            """
            deadline = when()
            if deadline is None:
                return
            time_left = deadline - time.time()
            if time_left <= 0:
                # The timer expired
                with reactive.isolate():
                    when.set(None)
                    trigger.set(trigger() + 1)
            else:
                reactive.invalidate_later(time_left)

        @reactive.Calc
        @reactive.event(trigger, ignore_none=False)
        @functools.wraps(f)
        def debounced():
            return cached()

        return debounced

    return wrapper

def server(input, output, session):

    df_store     = reactive.Value(None)
    fdf_store    = reactive.Value(None)   # tracks_tagged_ann dataframe (optional)
    csv_fpath    = reactive.Value(None)
    group_ids    = reactive.Value([])
    cur_idx      = reactive.Value(0)
    ann_state    = reactive.Value({
        "event": "", "ambiguous": "", "complex": "",
        "error": "", "labelbee_valid": "",
        "comment": "", "to_review": "",
    })
    status_msg   = reactive.Value(("info", "Enter a CSV path and click Load."))
    loading_flag = reactive.Value(False)
    
    trajectory_renderer = TrajectoryRenderer()
    prerender_cancel = threading.Event()

    def start_prerender(gids, start_idx):
        nonlocal prerender_cancel
        prerender_cancel.set()          # cancel previous
        prerender_cancel = threading.Event()
        cancel = prerender_cancel

        def _prerender(gids, start_idx, cancel, n=5):
            end_idx = min(start_idx+n,len(gids))
            start_idx = max(0,start_idx)
            for gid in gids[start_idx:end_idx]:
                if cancel.is_set():
                    return
                #make_trajectory_plot_b64(gid)
                trajectory_renderer.render(gid)

        threading.Thread(
            target=_prerender,
            args=(gids, start_idx, prerender_cancel),
            daemon=True
        ).start()

    # ── helpers ───────────────────────────────────────────────────────────────

    def current_gid():
        gids = group_ids.get()
        idx  = cur_idx.get()
        return gids[idx] if gids and idx < len(gids) else None

    def autosave():
        return
        df   = df_store.get()
        path = csv_fpath.get()
        if df is None or path is None:
            return
        try:
            write_df(df, path)
        except Exception as e:
            status_msg.set(("err", f"Autosave failed: {e}"))

    def apply_ann_to_df(field_map: dict):
        df  = df_store.get()
        gid = current_gid()
        if df is None or gid is None:
            return
        mask = df["group_id"] == gid
        col_map = {
            "event":          "event",
            "ambiguous":      "ambiguousflag",
            "complex":        "complex",
            "error":          "error",
            "labelbee_valid": "labelbee_valid",
            "comment":        "comment",
            "to_review":      "to_review",
        }
        for field, val in field_map.items():
            col = col_map.get(field, field)
            if col in df.columns:
                df.loc[mask, col] = val
        df_store.set(df.copy())
        autosave()

    def load_group_into_widgets(gid):
        loading_flag.set(True)
        df = df_store.get()
        if df is None:
            loading_flag.set(False)
            return
        rows = df[df["group_id"] == gid]
        if rows.empty:
            loading_flag.set(False)
            return
        row = rows.iloc[0]

        def _get(col):
            v = row.get(col, "")
            if pd.isna(v) or str(v).strip() in ("", "nan", "NA"):
                return ""
            return str(v)

        state = {
            "event":          _get("event"),
            "ambiguous":      _get("ambiguousflag"),
            "complex":        _get("complex"),
            "error":          _get("error"),
            "labelbee_valid": _get("labelbee_valid"),
            "comment":        _get("comment"),
            "to_review":      _get("to_review"),
        }
        with reactive.isolate():
            idx = cur_idx.get()
        print(f"UPDATE STATE, idx={idx} comment=",state['comment'])
        ann_state.set(state)
        print(f"UPDATE COMMENT FIELD, comment=",state['comment'])
        with reactive.isolate():
            print(f"input.comment()=",input.comment())
        ui.update_text_area("comment",   value=state["comment"],   session=session)
        ui.update_text_area("to_review", value=state["to_review"], session=session)
        loading_flag.set(False)
        with reactive.isolate():
            print(f"input.comment()=",input.comment())
        with reactive.isolate():
            idx = cur_idx.get()
        print(f"DONE UPDATING idx={idx}")

    # ── load main CSV ─────────────────────────────────────────────────────────

    @reactive.Effect
    @reactive.event(input.btn_load)
    def _load():
        raw = (input.csv_path() or "").strip()
        if not raw:
            status_msg.set(("err", "Please enter a file path."))
            return
        p = Path(raw)
        if not p.exists():
            status_msg.set(("err", f"File not found: {p}"))
            return
        try:
            df = pd.read_csv(p)
        except Exception as e:
            status_msg.set(("err", f"Could not read CSV: {e}"))
            return

        df.columns = [c.strip() for c in df.columns]
        for col, default in [
            ("event", ""), ("ambiguousflag", ""),
            ("complex", ""), ("error", ""),
            ("comment", ""), ("to_review", ""),
            ("labelbee_valid", ""),
        ]:
            if col not in df.columns:
                df[col] = default

        df_store.set(df)
        csv_fpath.set(p)
        gids = sorted(df["group_id"].dropna().unique().tolist())
        group_ids.set(gids)
        cur_idx.set(0)
        fdf_note = "  |  fdf loaded ✓" if fdf_store.get() is not None else ""
        status_msg.set(("ok", f"Loaded {p.name} — {len(df)} rows, {len(gids)} groups. Autosave ON → {p}{fdf_note}"))

        #if gids:
        #    print(f"LOAD idx={0}")
        #    load_group_into_widgets(gids[0])
        #    _sync_slider(gids[0])

    # ── load tracks_tagged_ann CSV (fdf) ─────────────────────────────────────

    @reactive.Effect
    @reactive.event(input.btn_load_fdf)
    def _load_fdf():
        raw = (input.fdf_path() or "").strip()
        if not raw:
            status_msg.set(("err", "Please enter a path for the tracks_tagged_ann CSV."))
            return
        p = Path(raw)
        if not p.exists():
            status_msg.set(("err", f"fdf file not found: {p}"))
            return
        try:
            fdf = pd.read_csv(p)
        except Exception as e:
            status_msg.set(("err", f"Could not read fdf CSV: {e}"))
            return
        fdf.columns = [c.strip() for c in fdf.columns]

        gdf = fdf.groupby('group_id')

        print( gdf.get_group(8) )

        fdf_store.set(gdf)
        n_groups = fdf["group_id"].nunique() if "group_id" in fdf.columns else "?"
        status_msg.set(("ok", f"fdf loaded: {p.name} — {len(fdf)} rows, {n_groups} groups. Trajectory plot active."))
        trajectory_renderer.set_fdf(gdf)

    # ── group selector ────────────────────────────────────────────────────────

    @output
    @render.ui
    def group_selector_ui():
        gids = group_ids.get()
        if not gids:
            return ui.span("(load a CSV first)", style="color:#484f58; font-size:.8rem;")
        idx = cur_idx.get()
        fdf  = fdf_store.get()
        fdf_badge = (
            ui.span("● trajectory plot active", class_="fdf-tag")
            if fdf is not None
            else ui.span("○ no fdf — showing image", class_="fdf-none")
        )
        try:
            vals = [int(g) for g in gids]
            cur  = int(gids[idx]) if idx < len(gids) else vals[0]
            return ui.div(
                ui.input_slider(
                    "grp_slider", f"group_id  [{idx+1} / {len(gids)}]",
                    min=min(vals), max=max(vals), value=cur, step=1, width="100%",
                ),
                fdf_badge,
                style="width:100%;",
            )
        except (TypeError, ValueError):
            opts = {str(g): str(g) for g in gids}
            cur  = str(gids[idx]) if idx < len(gids) else list(opts)[0]
            return ui.div(
                ui.input_select(
                    "grp_slider", f"group_id  [{idx+1} / {len(gids)}]",
                    choices=opts, selected=cur, width="100%",
                ),
                fdf_badge,
            )


    @debounce(0.3)
    @reactive.Effect
    async def _slider_changed():
        gids = group_ids.get()
        if not gids:
            return
        try:
            v = input.grp_slider()
        except Exception:
            return
        if v is None:
            return
        try:
            new_idx = [str(g) for g in gids].index(str(int(float(v))))
        except ValueError:
            return
        if new_idx != cur_idx.get():
            print(f"SLIDER CHANGED idx={new_idx}")
            cur_idx.set(new_idx)
            load_group_into_widgets(gids[new_idx])
            #start_prerender(gids, new_idx - 2)
            gid = gids[new_idx]
            status_msg.set(("info", f"Group {gid}  ({new_idx+1}/{len(gids)})"))

    def _sync_slider(val):
        #print(f"SYNC_SLIDER ABORT")
        #return
        try:
            ui.update_slider("grp_slider", value=int(val), session=session)
        except Exception:
            try:
                ui.update_select("grp_slider", selected=str(val), session=session)
            except Exception:
                pass

    # ── prev / next ───────────────────────────────────────────────────────────

    @reactive.Effect
    @reactive.event(input.btn_prev)
    def _prev():
        gids = group_ids.get()
        if not gids:
            return
        new_idx = max(0, cur_idx.get() - 1)
        print(f"PREV SETTING idx={new_idx}")
        #with reactive.isolate():
        _sync_slider(gids[new_idx])
        #cur_idx.set(new_idx)
        #load_group_into_widgets(gids[new_idx])
        #start_prerender(gids, new_idx - 5)
        status_msg.set(("info", f"Group {gids[new_idx]}  ({new_idx+1}/{len(gids)})"))

    @reactive.Effect
    @reactive.event(input.btn_next)
    def _next():
        gids = group_ids.get()
        if not gids:
            return
        new_idx = min(len(gids) - 1, cur_idx.get() + 1)
        print(f"NEXT SETTING idx={new_idx}")
        #with reactive.isolate():
        _sync_slider(gids[new_idx])
        #cur_idx.set(new_idx)
        #load_group_into_widgets(gids[new_idx])
        start_prerender(gids, new_idx + 1)
        status_msg.set(("info", f"Group {gids[new_idx]}  ({new_idx+1}/{len(gids)})"))

    # ── annotation button clicks → autosave ───────────────────────────────────

    @reactive.Effect
    @reactive.event(input.ann_click)
    def _ann_btn():
        if loading_flag.get():
            print(f"_ann_btn: loading. ABORT")
            return
        click = input.ann_click()
        if not click:
            return
        field = click.get("field")
        value = click.get("value")
        if not field or value is None:
            return
        s = dict(ann_state.get())
        new_val = "" if s.get(field) == value else value
        s[field] = new_val
        ann_state.set(s)  # Should auto update bbutton display
        apply_ann_to_df({field: new_val})
        disp = new_val if new_val else "(cleared)"
        status_msg.set(("ok", f"✓ {field} = {disp}  (group {current_gid()}, autosaved)"))

    @reactive.Effect
    @reactive.event(input.ann_freetext)
    def _ann_freetext():
        if loading_flag.get():
            print(f"_ann_freetext: loading. ABORT")
            return
        ev = input.ann_freetext()
        if not ev:
            return
        field = ev.get("field")
        value = ev.get("value", "")
        if not field:
            return
        with reactive.isolate():
            s = dict(ann_state.get())
            if value == s.get(field):
                return
            s[field] = value
            ann_state.set(s)
        apply_ann_to_df({field: value})
        disp = value if value else "(cleared)"
        status_msg.set(("ok", f"✓ {field} = {disp}  (group {current_gid()}, autosaved)"))

    @reactive.Effect
    def _comment_changed():
        print(f"_comment_changed")
        if loading_flag.get():
            print(f"_comment_changed: loading. ABORT")
            return
        val = input.comment()
        print(f"_comment_changed val=",val)
        with reactive.isolate():
            idx = cur_idx.get()
            print(f"_comment_changed: idx={idx}")

            s   = dict(ann_state.get())
            print(f"_comment_changed s.get('comment')=",s.get("comment"))
            if val == s.get("comment"):
                print(f"_comment_changed: no change. ABORT")
                return
            s["comment"] = val
            ann_state.set(s)
            print(f"_comment_changed idx={idx} applying",val)
        apply_ann_to_df({"comment": val})

    @reactive.Effect
    def _to_review_changed():
        if loading_flag.get():
            print(f"_to_review_changed: loading. ABORT")
            return
        val = input.to_review()
        with reactive.isolate():
            s   = dict(ann_state.get())
            if val == s.get("to_review"):
                return
            s["to_review"] = val
            ann_state.set(s)
        apply_ann_to_df({"to_review": val})

    # ── snapshot (timestamped copy) ───────────────────────────────────────────

    @reactive.Effect
    @reactive.event(input.btn_snap)
    def _snapshot():
        df   = df_store.get()
        path = csv_fpath.get()
        if df is None or path is None:
            status_msg.set(("err", "No data loaded."))
            return
        out = timestamped_path(path)
        try:
            write_df(df, out)
            status_msg.set(("ok", f"💾 Snapshot → {out}"))
        except Exception as e:
            status_msg.set(("err", f"Save failed: {e}"))

    # ── annotation panel ──────────────────────────────────────────────────────

    @output
    @render.ui
    def ann_panel():
        s = ann_state.get()
        # build JS to sync the free-text inputs after re-render
        # (value= in html is initial-value only; script corrects live value)
        fields = ["event", "ambiguous", "complex", "error", "labelbee_valid"]
        sync_js = "".join(
            f"(function(){{var el=document.getElementById('ft_{f}');"
            f"if(el)el.value={repr(s.get(f,''))};}})();"
            for f in fields
        )
        return ui.div(
            ui.div("event", class_="ann-label"),
            ann_buttons("event", ["D", "U", "DU", "UD"], s.get("event", "")),

            ui.div("ambiguous", class_="ann-label"),
            ann_buttons("ambiguous", ["True", "False"], s.get("ambiguous", "")),

            ui.div("complex", class_="ann-label"),
            ann_buttons("complex", ["True", "False"], s.get("complex", "")),

            ui.div("error", class_="ann-label"),
            ann_buttons("error", ["True", "False"], s.get("error", "")),

            ui.div("labelbee_valid", class_="ann-label"),
            ann_buttons("labelbee_valid", ["True", "False"], s.get("labelbee_valid", "")),

            ui.tags.script(sync_js),
            class_="ann-scroll",
        )

    # ── image / trajectory display ────────────────────────────────────────────

    @output
    @render.ui
    def img_display():
        with reactive.isolate():
            df   = df_store.get() # Avoid redrw when updating the annotations. SHould actually separate annotations from basic df
        gids = group_ids.get()
        idx  = cur_idx.get()
        gdf  = fdf_store.get()

        if df is None or not gids or idx >= len(gids):
            return ui.p("No image.", style="color:#484f58; padding:1rem;")

        gid  = gids[idx]
        rows = df[df["group_id"] == gid]
        if rows.empty:
            return ui.p("No rows for this group.", style="color:#484f58; padding:1rem;")

        # ── trajectory plot (fdf loaded) ──────────────────────────────────
        if gdf is not None:
            print(f"Display  {gid}")
            b64 = trajectory_renderer.render(gid) #make_trajectory_plot_b64(gid)
            if b64:
                return ui.tags.img(src=b64, style="max-width:100%;")
            return ui.p(
                f"No trajectory data for group {gid} in fdf.",
                style="color:#f85149; padding:1rem;",
            )

        # ── fallback: static image ────────────────────────────────────────
        row   = rows.iloc[0]
        fname = f"5second/group_{row['group_id']:04}_track_{row['track_id']:05}_5s.png"
        b64   = img_to_b64(str(fname))
        if b64:
            return ui.tags.img(src=b64, style="max-width:100%; max-height:500px;")
        return ui.div(
            ui.p(str(fname), style="color:#58a6ff; word-break:break-all; padding:.5rem;"),
            ui.p("(file not found on disk)", style="color:#f85149; font-size:.75rem; padding:0 .5rem;"),
        )

    # ── labelbee link ─────────────────────────────────────────────────────────

    @output
    @render.ui
    def group_id_text():
        gids = group_ids.get()
        if (gids is None):
            return ui.div("GID -", class_="ann-label")
        idx  = cur_idx.get()
        if (idx is None):
            return ui.p("GID --", class_="ann-label")
        if (idx >= len(gids)):
            return ui.p("GID ---", class_="ann-label")
        gid = gids[idx]
        return ui.div("GID "+str(gid), class_="ann-label"),

    @output
    @render.ui
    def labelbee_ui():
        df   = df_store.get()
        gids = group_ids.get()
        idx  = cur_idx.get()
        gdf  = fdf_store.get()

        if df is None or not gids or idx >= len(gids):
            return ui.div()
        rows = df[df["group_id"] == gids[idx]]
        if rows.empty:
            return ui.div()
        row   = rows.iloc[0]
        video = row.get("video", None)
        if (gdf is not None):
            gid = gids[idx]
            print(f'labelbee gid={gid}')
            edf = gdf.get_group(gid)
            frame = edf.iloc[0]['frame'] # First frame
            print(f"labelbee url: has gdf, frame={frame}")
        else:
            frame = row.get("frame", 0)
            print(f"labelbee url: no gdf, frame={frame}")
        tagid = row.get("tagid")
        print(f"tagid={tagid}")
        vdata = row.get("video_data", None)
        if not video or pd.isna(video):
            return ui.p("(no video column)", style="color:#484f58; font-size:.75rem; margin-top:.5rem;")
        url = build_labelbee_url(
            str(video), frame, tagid,
            str(vdata) if vdata and not pd.isna(vdata) else None,
        )
        return ui.div(
            ui.tags.a("Open in LabelBee", href=url, target="labelbee", class_="lb-btn"),
            ui.tags.a("(new tab)", href=url, target="_blank", class_="lb-btn"),
            ui.p(url, class_="lb-url"),
        )

    # ── status bar ────────────────────────────────────────────────────────────

    @output
    @render.ui
    def status_ui():
        kind, msg = status_msg.get()
        cls   = {"ok": "status-ok", "err": "status-err", "info": "status-info"}.get(kind, "status-info")
        pulse = ui.tags.span(class_="autosave-pulse") if kind == "ok" else ui.span()
        return ui.div(pulse, ui.span(msg, class_=cls), class_="status-bar")


app = App(app_ui, server)