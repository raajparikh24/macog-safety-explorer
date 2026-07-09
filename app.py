"""
MACOG Safety Explorer — Streamlit app
Run: streamlit run app.py
Expects a `data/` folder next to this file containing:
    - macog_base.zip  (from BuildBase notebook)   -- OR the 4 gpkgs already unzipped
    - fpa.py
"""
import os, sys, io, time, zipfile, importlib
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import linref as lr
import folium
from folium.features import GeoJsonTooltip
import streamlit as st
from streamlit_folium import st_folium

# ----------------------- Config -----------------------
st.set_page_config(page_title='MACOG Safety Explorer', layout='wide')

DATA_DIR = Path(__file__).parent / 'data'
BASE_DIR = DATA_DIR / 'base_files'

SEGMENT_LENGTHS = [0.10, 0.25, 0.50, 1.0]
HIN_MODES = ['is_ped_fsi','is_bike_fsi','is_buggy_fsi','is_motorcycle_fsi','is_vehicle_fsi']
MODE_LABELS = {'is_ped_fsi':'Pedestrian','is_bike_fsi':'Bicycle','is_buggy_fsi':'Horse & Buggy',
               'is_motorcycle_fsi':'Motorcycle','is_vehicle_fsi':'Vehicle'}
COUNTIES = ['Elkhart County','Kosciusko County','Marshall County','St Joseph County']
HRN_ANALYSES = ['fsi_all_elk-stj_inc','fsi_all_kos-mar_inc','fsi_all_reg_uninc',
                'fsi_vru_reg_inc','fsi_vru_reg_uninc']
TIER_ORDER = ['Critical','High','Medium','Low','Minimal']

FC_LABELS = {
    1:'1 - Interstate', 2:'2 - Other Freeway/Expressway',
    3:'3 - Other Principal Arterial', 4:'4 - Minor Arterial',
    5:'5 - Major Collector', 6:'6 - Minor Collector', 7:'7 - Local',
}
TIER_COLORS = {'Critical':'#8b0000','High':'#e34a33','Medium':'#fdbb84',
               'Low':'#67a9cf','Minimal':'#bdbdbd'}

HIN_THRESHOLDS = {
    'Elkhart County':   {'is_ped_fsi':0.363,'is_bike_fsi':0.316,'is_buggy_fsi':0.226,'is_motorcycle_fsi':0.342,'is_vehicle_fsi':0.810},
    'Kosciusko County': {'is_ped_fsi':0.294,'is_bike_fsi':0.810,'is_buggy_fsi':0.222,'is_motorcycle_fsi':0.312,'is_vehicle_fsi':0.322},
    'Marshall County':  {'is_ped_fsi':0.333,'is_bike_fsi':0.311,'is_buggy_fsi':0.240,'is_motorcycle_fsi':0.337,'is_vehicle_fsi':0.476},
    'St Joseph County': {'is_ped_fsi':0.401,'is_bike_fsi':0.324,'is_buggy_fsi':0.290,'is_motorcycle_fsi':0.391,'is_vehicle_fsi':1.019},
}

BASE_URLS = {
    '010': 'https://github.com/raajparikh24/macog-safety-explorer/releases/download/v1.0/base_seg010.gpkg',
    '025': 'https://github.com/raajparikh24/macog-safety-explorer/releases/download/v1.0/base_seg025.gpkg',
    '050': 'https://github.com/raajparikh24/macog-safety-explorer/releases/download/v1.0/base_seg050.gpkg',
    '100': 'https://github.com/raajparikh24/macog-safety-explorer/releases/download/v1.0/base_seg100.gpkg',
}

# ----------------------- One-time setup -----------------------
def _tag(length): return f'{int(round(length*100)):03d}'

def _prepare_data_dir():
    """Download base files from Release + patch/import fpa.py."""
    DATA_DIR.mkdir(exist_ok=True)
    BASE_DIR.mkdir(exist_ok=True)

    # Download any missing base files from the GitHub Release
    import urllib.request
    for tag, url in BASE_URLS.items():
        dest = BASE_DIR / f'base_seg{tag}.gpkg'
        if not dest.exists():
            with st.spinner(f'First-run download: base_seg{tag}.gpkg (~30 MB)…'):
                urllib.request.urlretrieve(url, dest)

    # Patch fpa.py (must be committed to repo at data/fpa.py)
    fpa_src_path = DATA_DIR / 'fpa.py'
    if not fpa_src_path.exists():
        st.error(f'Missing `{fpa_src_path}` — upload fpa.py to the repo under data/.')
        st.stop()
    fpa_work = DATA_DIR / '_fpa_patched.py'
    src = fpa_src_path.read_text()
    if 'from tooles.utils.utils import format_numbers_in_string' in src:
        src = src.replace(
            'from tooles.utils.utils import format_numbers_in_string',
            'format_numbers_in_string = lambda *a, **k: None')
    lines = src.split('\n')
    for i, line in enumerate(lines):
        if 'np.where(' in line and i+1 < len(lines) and '], np.nan)' in lines[i+1]:
            if '.astype(object)' not in lines[i+1]:
                lines[i+1] = lines[i+1].replace(
                    '], np.nan)', '].astype(object), np.nan).astype(object)')
    fpa_work.write_text('\n'.join(lines))
    sys.path.insert(0, str(DATA_DIR))
    if '_fpa_patched' in sys.modules: del sys.modules['_fpa_patched']
    import _fpa_patched  # noqa: import validates it loads

_prepare_data_dir()

# ----------------------- Cached loaders -----------------------
@st.cache_resource(max_entries=1, show_spinner='Loading base file…')
def load_base(length):
    """Load one segment length's gpkg + rebuild EventsCollections. Cached across reruns."""
    tag = _tag(length)
    path = BASE_DIR / f'base_seg{tag}.gpkg'
    segments = gpd.read_file(path, layer='segments')
    crashes_many = gpd.read_file(path, layer='crashes_many')
    hrn = gpd.read_file(path, layer='hrn_roadways')

    # Attach functional + city_code onto segments from HRN layer (roadways_ec has them)
    if 'functional' not in segments.columns:
        attr_lookup = (hrn[['county_name','route_id','chain','functional']]
                       .drop_duplicates(subset=['county_name','route_id','chain']))
        segments = segments.merge(attr_lookup, on=['county_name','route_id','chain'], how='left')

    segments_ec = lr.EventsCollection(
        segments, ['county_name','route_id','chain'],
        beg='begin', end='end', missing_data='drop')
    crashes_many_ec = lr.EventsCollection(
        crashes_many, ['county_name','route_id','chain'],
        beg='loc', missing_data='drop')
    return dict(segments=segments, segments_ec=segments_ec,
                crashes_many_ec=crashes_many_ec, hrn=hrn)

@st.cache_data(max_entries=4, show_spinner='Computing HIN…')
def compute_hin(length, blur_size, threshold_scale, year_range):
    """Live HIN. Keyed on (length, blur, thr_scale, year_range) so repeats are instant."""
    base = load_base(length)
    segments_ec = base['segments_ec']

    # Filter crashes by year window
    ymin, ymax = year_range
    cdf = base['crashes_many_ec'].df
    cdf = cdf[cdf['Collision Year'].between(ymin, ymax)]
    crashes_many_ec = lr.EventsCollection(
        cdf, ['county_name','route_id','chain'],
        beg='loc', missing_data='drop')

    distributed_sw = segments_ec.merge(crashes_many_ec).distribute(
        column=HIN_MODES, blur_size=int(blur_size), blur_style='linear')
    distributed_near = segments_ec.merge(crashes_many_ec).distribute(
        column=HIN_MODES, blur_size=0)

    df = segments_ec.df.copy()
    for m in HIN_MODES:
        df[m+'_sw']   = distributed_sw[m].round(3)
        df[m+'_near'] = distributed_near[m].round(3)
    for county, thresholds in HIN_THRESHOLDS.items():
        cmask = df['county_name'] == county
        for metric, thr in thresholds.items():
            df.loc[cmask, metric+'_thr'] = thr * threshold_scale
    for m in HIN_MODES:
        df[m+'_hin'] = (df[m+'_sw'] >= df[m+'_thr']).astype(int)
    return df

# ----------------------- Summaries + map -----------------------
def hin_summary(hin_df, counties_filter):
    df = hin_df[hin_df['county_name'].isin(counties_filter)]
    rows = []
    for m in HIN_MODES:
        for county in counties_filter:
            d = df[df['county_name']==county]
            total_mi = d['length'].sum()
            hin_mi   = (d[m+'_hin'] * d['length']).sum()
            fsi_count = d[m+'_near'].sum()
            hin_fsi_count = (d[m+'_hin'] * d[m+'_near']).sum()
            rows.append({
                'Mode': MODE_LABELS[m],
                'County': county.replace(' County',''),
                'HIN Miles': round(hin_mi, 1),
                'Total Miles': round(total_mi, 1),
                'HIN %': round(100*hin_mi/total_mi, 1) if total_mi else 0,
                'FSI Count':      int(fsi_count),
                'HIN FSI Count':  int(hin_fsi_count),
                'HIN FSI %': round(100*hin_fsi_count/fsi_count, 1) if fsi_count else 0,
            })
    return pd.DataFrame(rows)

def hrn_summary(hrn_df, analysis, counties_filter):
    df = hrn_df[hrn_df['county_name'].isin(counties_filter)]
    tcol = f'tier_{analysis}'
    return (df.groupby(['county_name', tcol])['length']
              .sum().unstack(tcol).reindex(columns=TIER_ORDER).round(1).fillna(0))

def build_map(view, base, hin_df, mode, hrn_analysis, counties_filter):
    if view == 'HIN':
        gdf = hin_df[hin_df['county_name'].isin(counties_filter)]
        col = f'{mode}_hin'
    else:
        gdf = base['hrn'][base['hrn']['county_name'].isin(counties_filter)]
        col = f'tier_{hrn_analysis}'

    if len(gdf) == 0:
        return folium.Map(location=[41.68, -86.25], zoom_start=10, tiles='cartodbpositron')

    gdf_wgs = gdf.to_crs('EPSG:4326')
    bounds = gdf_wgs.total_bounds
    center = [(bounds[1]+bounds[3])/2, (bounds[0]+bounds[2])/2]
    m = folium.Map(location=center, zoom_start=10, tiles='cartodbpositron')

    if view == 'HIN':
        base_gdf = gdf_wgs[gdf_wgs[col] == 0]
        hin_gdf  = gdf_wgs[gdf_wgs[col] == 1]
        if len(base_gdf):
            folium.GeoJson(base_gdf.__geo_interface__,
                style_function=lambda f: {'color':'#cccccc','weight':1,'opacity':0.5},
                name='All segments').add_to(m)
        if len(hin_gdf):
            folium.GeoJson(hin_gdf.__geo_interface__,
                style_function=lambda f: {'color':'#d62728','weight':3,'opacity':0.9},
                name=f'HIN — {MODE_LABELS[mode]}',
                tooltip=GeoJsonTooltip(
                    fields=['county_name','route_id',f'{mode}_sw',f'{mode}_thr',f'{mode}_near'],
                    aliases=['County','Route','SW count','Threshold','Nearest count'],
                    localize=True)).add_to(m)
    else:
        def _style(feat):
            tier = feat['properties'].get(col, 'Minimal')
            return {'color': TIER_COLORS.get(tier, '#bdbdbd'),
                    'weight': 3 if tier in ('Critical','High') else 2, 'opacity': 0.85}
        folium.GeoJson(gdf_wgs.__geo_interface__, style_function=_style,
            name=f'HRN — {hrn_analysis}',
            tooltip=GeoJsonTooltip(
                fields=['county_name','route_id',col,f'pred_{hrn_analysis}'],
                aliases=['County','Route','Tier','Predicted density'],
                localize=True)).add_to(m)

    m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
    folium.LayerControl().add_to(m)
    return m

# ----------------------- UI -----------------------
st.title('MACOG Safety Explorer')

with st.sidebar:
    st.header('Controls')
    length = st.selectbox('Segment length', SEGMENT_LENGTHS,
                          index=1, format_func=lambda x: f'{x:.2f} mi')
    view = st.radio('View', ['HIN','HRN'], horizontal=True)

    st.markdown('**HIN parameters**')
    blur = st.slider('Blur size', 0, 5, 2)
    thr_scale = st.slider('Threshold scale', 0.5, 2.0, 1.0, 0.05,
                          help='1.0 = consultant; <1 stricter; >1 looser')
    mode = st.selectbox('Mode', HIN_MODES, index=4, format_func=lambda m: MODE_LABELS[m])

    st.markdown('**HRN parameters**')
    hrn_analysis = st.selectbox('HRN analysis', HRN_ANALYSES)

    st.markdown('**Filters**')
    counties_filter = st.multiselect('Counties', COUNTIES, default=COUNTIES)
    if not counties_filter:
        counties_filter = COUNTIES

    fc_filter = st.multiselect(
        'Functional class', list(FC_LABELS.keys()),
        default=list(FC_LABELS.keys()),
        format_func=lambda x: FC_LABELS[x])
    if not fc_filter:
        fc_filter = list(FC_LABELS.keys())

    # Detect available years from the base file
    _tmp_years = load_base(length)['crashes_many_ec'].df['Collision Year'].dropna()
    ymin_avail, ymax_avail = int(_tmp_years.min()), int(_tmp_years.max())
    year_range = st.slider('Crash years (HIN only)',
                           ymin_avail, ymax_avail, (ymin_avail, ymax_avail))
    st.caption('HRN tiers are fixed at build time (2019–2023 model).')

# Compute
base = load_base(length)
hin_df = compute_hin(length, blur, thr_scale, year_range)

# Apply functional-class filter to both hin and hrn
hin_df = hin_df[hin_df['functional'].isin(fc_filter)]
hrn_view = base['hrn'][base['hrn']['functional'].isin(fc_filter)]
base = {**base, 'hrn': hrn_view}

# Layout: map on left, summary on right
col_map, col_summary = st.columns([3, 2])

with col_map:
    m = build_map(view, base, hin_df, mode, hrn_analysis, counties_filter)
    st_folium(m, width=None, height=650, returned_objects=[])

with col_summary:
    if view == 'HIN':
        st.subheader(f'HIN summary — blur={blur}, scale={thr_scale:.2f}')
        st.dataframe(hin_summary(hin_df, counties_filter),
                     use_container_width=True, hide_index=True)
    else:
        st.subheader(f'HRN summary — {hrn_analysis}')
        st.dataframe(hrn_summary(base['hrn'], hrn_analysis, counties_filter),
                     use_container_width=True)

# ----------------------- Export -----------------------
st.divider()
st.subheader('Export')

def make_exports():
    hin_export = hin_df[hin_df['county_name'].isin(counties_filter)]
    hrn_export = base['hrn'][base['hrn']['county_name'].isin(counties_filter)]

    # gpkg to bytes
    gpkg_buf = io.BytesIO()
    tmp_path = DATA_DIR / f'_export_tmp_{int(time.time())}.gpkg'
    if not isinstance(hin_export, gpd.GeoDataFrame):
        hin_export = gpd.GeoDataFrame(hin_export, geometry='geometry', crs=base['segments'].crs)
    hin_export.to_file(tmp_path, layer='hin_roadways', driver='GPKG')
    hrn_export.to_file(tmp_path, layer='hrn_roadways', driver='GPKG')
    gpkg_buf.write(tmp_path.read_bytes())
    tmp_path.unlink()
    gpkg_buf.seek(0)

    # xlsx
    xlsx_buf = io.BytesIO()
    with pd.ExcelWriter(xlsx_buf, engine='openpyxl') as w:
        hin_summary(hin_df, counties_filter).to_excel(w, sheet_name='HIN summary', index=False)
        for a in HRN_ANALYSES:
            hrn_summary(base['hrn'], a, counties_filter).to_excel(w, sheet_name=a[:31])
        pd.DataFrame([
            ['Segment length (mi)', length], ['Blur size', blur],
            ['HIN threshold scale', thr_scale],
            ['Counties', ', '.join(counties_filter)],
            ['Exported at', time.strftime('%Y-%m-%d %H:%M:%S')],
        ], columns=['Parameter','Value']).to_excel(w, sheet_name='Parameters', index=False)
    xlsx_buf.seek(0)
    return gpkg_buf.getvalue(), xlsx_buf.getvalue()

stamp = time.strftime('%Y%m%d_%H%M%S')
tag = _tag(length)
c1, c2 = st.columns(2)
if c1.button('Prepare exports'):
    gpkg_bytes, xlsx_bytes = make_exports()
    st.session_state['gpkg'] = gpkg_bytes
    st.session_state['xlsx'] = xlsx_bytes
if 'gpkg' in st.session_state:
    c1.download_button('Download GPKG', st.session_state['gpkg'],
                       file_name=f'macog_export_{tag}_{stamp}.gpkg',
                       mime='application/octet-stream')
    c2.download_button('Download XLSX', st.session_state['xlsx'],
                       file_name=f'macog_export_{tag}_{stamp}.xlsx',
                       mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
