"""
新能源汽车充电桩空间布局与使用效率分析系统
EV Charging Station Spatial Layout & Usage Efficiency Analysis System

增强版：支持数据上传、智能清洗、多格式下载
基于 Streamlit + Plotly 构建的交互式数据可视化大屏
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════
# 页面配置
# ═══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="新能源充电桩分析系统",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════
# 自定义 CSS 样式
# ═══════════════════════════════════════════════════════════
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    * { font-family: 'Inter', 'Microsoft YaHei', sans-serif; }

    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #0d1233 30%, #101845 60%, #0a0f2c 100%);
    }

    .main-header {
        background: linear-gradient(90deg, rgba(0, 210, 255, 0.08) 0%, rgba(58, 123, 213, 0.15) 50%, rgba(0, 210, 255, 0.08) 100%);
        border-bottom: 1px solid rgba(0, 210, 255, 0.2);
        padding: 28px 40px; margin-bottom: 24px; border-radius: 16px;
        display: flex; align-items: center; justify-content: space-between;
    }
    .main-header .title-group h1 {
        font-size: 2.0rem; font-weight: 700;
        background: linear-gradient(90deg, #00d2ff, #3a7bd5, #00d2ff);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text; margin: 0; letter-spacing: 0.5px;
    }
    .main-header .title-group p {
        color: rgba(255, 255, 255, 0.5); font-size: 0.85rem;
        margin: 4px 0 0 0; letter-spacing: 1.5px;
    }

    .metric-card {
        background: linear-gradient(135deg, rgba(20, 25, 60, 0.9) 0%, rgba(25, 30, 70, 0.85) 100%);
        border: 1px solid rgba(0, 210, 255, 0.12); border-radius: 16px;
        padding: 22px 24px; text-align: center; transition: all 0.3s ease;
        backdrop-filter: blur(10px);
    }
    .metric-card:hover {
        border-color: rgba(0, 210, 255, 0.4); transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(0, 150, 255, 0.12);
    }
    .metric-card .metric-icon { font-size: 2rem; margin-bottom: 6px; }
    .metric-card .metric-value {
        font-size: 2rem; font-weight: 700;
        background: linear-gradient(180deg, #ffffff, #a0d0ff);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .metric-card .metric-label {
        color: rgba(255, 255, 255, 0.5); font-size: 0.8rem;
        text-transform: uppercase; letter-spacing: 1px;
    }

    .section-title {
        font-size: 1.15rem; font-weight: 600; color: #e0f0ff;
        margin: 8px 0 16px 0; padding-left: 14px;
        border-left: 3px solid #00d2ff; letter-spacing: 0.5px;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(10, 14, 39, 0.98) 0%, rgba(13, 18, 50, 0.96) 100%);
        border-right: 1px solid rgba(0, 210, 255, 0.1);
    }
    [data-testid="stSidebar"] .stMarkdown h2, [data-testid="stSidebar"] .stMarkdown h3 {
        color: #00d2ff !important;
    }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stMultiSelect label {
        color: rgba(255, 255, 255, 0.8) !important; font-size: 0.85rem !important;
    }

    .filter-info {
        background: rgba(0, 210, 255, 0.06); border: 1px solid rgba(0, 210, 255, 0.15);
        border-radius: 10px; padding: 10px 18px;
        color: rgba(255, 255, 255, 0.6); font-size: 0.82rem; margin-bottom: 16px;
    }

    .data-source-badge {
        display: inline-block; background: rgba(0,210,255,0.12);
        border: 1px solid rgba(0,210,255,0.25); border-radius: 20px;
        padding: 4px 14px; font-size: 0.78rem; color: #00d2ff; margin-right: 8px;
    }

    .clean-card {
        background: rgba(15, 25, 55, 0.7); border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px; padding: 18px 20px; margin: 8px 0;
        transition: all 0.3s ease;
    }
    .clean-card:hover { border-color: rgba(0,210,255,0.3); }

    .quality-good { color: #2ecc71; font-weight: 700; }
    .quality-warn { color: #f39c12; font-weight: 700; }
    .quality-bad { color: #e74c3c; font-weight: 700; }

    .footer {
        text-align: center; color: rgba(255, 255, 255, 0.25);
        font-size: 0.75rem; padding: 20px;
        border-top: 1px solid rgba(255, 255, 255, 0.05); margin-top: 40px;
    }

    .stDownloadButton button {
        background: linear-gradient(135deg, rgba(0,210,255,0.15), rgba(58,123,213,0.15)) !important;
        border: 1px solid rgba(0,210,255,0.3) !important; color: #00d2ff !important;
        transition: all 0.3s ease !important;
    }
    .stDownloadButton button:hover {
        background: linear-gradient(135deg, rgba(0,210,255,0.25), rgba(58,123,213,0.25)) !important;
        border-color: rgba(0,210,255,0.5) !important;
    }

    hr { border-color: rgba(255,255,255,0.06); }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# 中英文列名映射表（用于图表标签中文化）
# ═══════════════════════════════════════════════════════════
CN_LABELS = {
    # 地理信息
    "Latitude": "纬度", "Longitude": "经度", "Address": "地址", "City": "城市",
    # 设备参数
    "Charger Type": "充电桩类型", "Charging Capacity (kW)": "充电容量 (kW)",
    "Connector Types": "连接器类型", "Parking Spots": "停车位数量",
    # 运营指标
    "Cost (USD/kWh)": "充电成本 (美元/kWh)", "Availability": "可用时间",
    "Usage Stats (avg users/day)": "日均使用人数",
    "Distance to City (km)": "距城市距离 (km)",
    # 评价反馈
    "Reviews (Rating)": "用户评分", "Maintenance Frequency": "维护频率",
    # 基础属性
    "Station ID": "站点编号", "Station Operator": "运营商",
    "Installation Year": "安装年份", "Renewable Energy Source": "可再生能源",
    # 衍生变量
    "Capacity Level": "容量等级", "Usage Level": "使用强度等级",
}
# 图表中常见缩写的翻译
CN_CHARGER = {"AC Level 1": "交流一级 (AC L1)", "AC Level 2": "交流二级 (AC L2)",
              "DC Fast Charger": "直流快充 (DC)"}
CN_RENEWABLE = {"Yes": "可再生能源", "No": "非可再生能源", "可再生": "可再生能源", "非可再生": "非可再生能源"}
CN_AVAIL = {"24/7": "全天候", "9:00-18:00": "白天时段", "6:00-22:00": "日间时段"}
CN_MAINT = {"Monthly": "月度", "Quarterly": "季度", "Annually": "年度"}

def cn_label(col):
    """将英文列名转为中文，未匹配则返回原名"""
    return CN_LABELS.get(col, col)

def cn_val(val):
    """将常见英文分类值转为中文"""
    return CN_CHARGER.get(val, CN_RENEWABLE.get(val, CN_AVAIL.get(val, CN_MAINT.get(val, val))))

def cn_df(df_in):
    """返回一个列名中文化后的DataFrame副本，用于图表显示"""
    df_cn = df_in.copy()
    df_cn.columns = [cn_label(c) for c in df_cn.columns]
    for col in df_cn.columns:
        if df_cn[col].dtype == 'object' and df_cn[col].nunique() < 30:
            try:
                df_cn[col] = df_cn[col].apply(cn_val)
            except:
                pass
    return df_cn

# ═══════════════════════════════════════════════════════════
# 通用函数
# ═══════════════════════════════════════════════════════════

def auto_infer_columns(df):
    """自动推断数据列类型，返回数值列/分类列/文本列的列表"""
    num_cols, cat_cols, text_cols, geo_cols = [], [], [], []
    for col in df.columns:
        col_lower = col.lower().strip()
        if any(k in col_lower for k in ['lat', 'latitude', '纬度']):
            geo_cols.append(col)
        elif any(k in col_lower for k in ['lng', 'lon', 'long', 'longitude', '经度']):
            geo_cols.append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            num_cols.append(col)
        elif df[col].nunique() < min(50, len(df) * 0.3):
            cat_cols.append(col)
        else:
            text_cols.append(col)
    return num_cols, cat_cols, text_cols, geo_cols

def detect_lat_lon(df):
    """自动检测经纬度列"""
    lat_col, lon_col = None, None
    for col in df.columns:
        col_lower = col.lower().strip()
        if not lat_col and any(k in col_lower for k in ['lat', 'latitude', '纬度']):
            lat_col = col
        if not lon_col and any(k in col_lower for k in ['lng', 'lon', 'long', 'longitude', '经度']):
            lon_col = col
    return lat_col, lon_col

# ═══════════════════════════════════════════════════════════
# 初始化 Session State
# ═══════════════════════════════════════════════════════════
for key, default in {
    "df": None, "raw_df": None, "file_loaded": False,
    "cleaned": False, "source_name": "", "clean_log": [],
    "data_source_type": ""
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ═══════════════════════════════════════════════════════════
# 侧边栏 - 数据源管理
# ═══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 20px 0 10px 0;">
        <div style="font-size: 2.5rem;">⚡</div>
        <div style="font-size: 1.1rem; font-weight: 700;
                    background: linear-gradient(90deg, #00d2ff, #3a7bd5);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                    margin: 4px 0;">充电桩分析平台</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── 数据加载区 ──
    st.markdown("### 📂 数据源管理")

    uploaded_file = st.file_uploader(
        "上传数据文件",
        type=["csv", "xlsx", "xls", "json"],
        help="支持 CSV / Excel / JSON 格式。不传则用默认数据集。",
        key="file_uploader"
    )

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                raw = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                raw = pd.read_excel(uploaded_file, engine='openpyxl')
            elif uploaded_file.name.endswith('.json'):
                raw = pd.read_json(uploaded_file)

            raw.columns = raw.columns.str.strip()
            st.session_state.raw_df = raw.copy()
            st.session_state.df = raw.copy()
            st.session_state.file_loaded = True
            st.session_state.source_name = uploaded_file.name
            st.session_state.data_source_type = "upload"
            st.session_state.cleaned = False
            st.session_state.clean_log = []

            st.success(f"✅ 已加载: {uploaded_file.name}")
            st.caption(f"行数: {len(raw):,} | 列数: {len(raw.columns)} | 大小: {uploaded_file.size/1024:.0f} KB")

        except Exception as e:
            st.error(f"❌ 加载失败: {e}")
            st.session_state.file_loaded = False
            st.session_state.df = None

    # 加载默认数据按钮
    if st.button("🔄 加载默认数据集", use_container_width=True):
        @st.cache_data(ttl=600)
        def load_default():
            df = pd.read_csv(os.path.join(os.path.dirname(__file__), "detailed_ev_charging_stations.csv"))
            df.columns = df.columns.str.strip()
            return df

        d = load_default()
        st.session_state.raw_df = d.copy()
        st.session_state.df = d.copy()
        st.session_state.file_loaded = True
        st.session_state.source_name = "detailed_ev_charging_stations.csv (默认)"
        st.session_state.data_source_type = "default"
        st.session_state.cleaned = False
        st.session_state.clean_log = []
        st.rerun()

    if st.session_state.file_loaded:
        st.markdown(f'<span class="data-source-badge">📁 {st.session_state.source_name}</span>', unsafe_allow_html=True)
        if st.session_state.cleaned:
            st.markdown('<span style="color:#2ecc71;font-size:0.78rem;">✅ 已清洗</span>', unsafe_allow_html=True)
    else:
        st.warning("⚠️ 请上传文件或加载默认数据")

    st.markdown("---")

    # ── 数据过滤（仅在加载后显示）──
    if st.session_state.file_loaded and st.session_state.df is not None:
        df = st.session_state.df
        num_cols, cat_cols, text_cols, geo_cols = auto_infer_columns(df)

        st.markdown("### 🔍 数据筛选器")

        # 动态选择器
        filter_active = False
        filters = {}

        if cat_cols:
            # 优先展示常见筛选字段
            priority_cats = [c for c in cat_cols if any(
                k in c.lower() for k in ['type', 'operator', 'city', '城市', '类型', '运营商', 'charger']
            )]
            other_cats = [c for c in cat_cols if c not in priority_cats]
            sorted_cats = priority_cats + other_cats[:3]  # 最多5个

            for col in sorted_cats[:5]:
                options = sorted(df[col].dropna().unique().tolist())
                if 2 <= len(options) <= 20:
                    selected = st.multiselect(
                        f"📌 {col}",
                        options=options, default=options,
                        key=f"filter_cat_{col}"
                    )
                    if len(selected) < len(options):
                        filter_active = True
                    filters[col] = selected

        if num_cols:
            # 容量/使用相关字段的范围过滤
            for col in num_cols[:4]:
                col_lower = col.lower()
                if any(k in col_lower for k in ['capacity', 'usage', 'cost', 'distance', 'rating', 'capacity', '使用', '成本']):
                    mn, mx = float(df[col].min()), float(df[col].max())
                    if mx > mn:
                        sel = st.slider(
                            f"📊 {col}",
                            min_value=mn, max_value=mx,
                            value=(mn, mx),
                            key=f"filter_num_{col}"
                        )
                        if sel != (mn, mx):
                            filter_active = True
                        filters[col] = sel

        if st.button("🔄 重置筛选条件", use_container_width=True):
            for key in list(st.session_state.keys()):
                if key.startswith("filter_"):
                    del st.session_state[key]
            st.rerun()

        st.markdown("---")

        # 数据统计
        st.caption(f"当前数据: {len(df):,} 行 × {len(df.columns)} 列")
        if st.session_state.cleaned:
            st.caption(f"清洗操作: {'; '.join(st.session_state.clean_log[:3])}")

    st.markdown("---")
    st.markdown('<p style="color:rgba(255,255,255,0.3);font-size:0.7rem;text-align:center;">⚡ EV Charging Analytics Platform v2.0</p>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# 主内容区 - 数据未加载时的提示
# ═══════════════════════════════════════════════════════════

if not st.session_state.file_loaded or st.session_state.df is None:
    st.markdown("""
    <div class="main-header">
        <div class="title-group">
            <h1>⚡ 新能源汽车充电桩空间布局与使用效率分析系统</h1>
            <p>SPATIAL LAYOUT & USAGE EFFICIENCY ANALYSIS · EV CHARGING INFRASTRUCTURE</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns([1, 2, 1])
    with col_b:
        st.markdown("""
        <div style="text-align:center; padding: 60px 20px; color: rgba(255,255,255,0.6);">
            <div style="font-size: 5rem; margin-bottom: 20px;">📂</div>
            <h2 style="color:#00d2ff;">欢迎使用充电桩分析平台</h2>
            <p style="font-size:1.1rem;">请通过左侧边栏上传数据文件，或加载默认数据集开始分析</p>
            <p style="font-size:0.85rem;color:rgba(255,255,255,0.4);">
                支持 CSV / Excel (.xlsx) / JSON 格式<br>
                系统将自动识别数据结构并适配分析功能
            </p>
        </div>
        """, unsafe_allow_html=True)
    st.stop()

# ═══════════════════════════════════════════════════════════
# 数据已加载 - 预处理
# ═══════════════════════════════════════════════════════════
df_raw = st.session_state.df.copy()
num_cols, cat_cols, text_cols, geo_cols = auto_infer_columns(df_raw)
lat_col, lon_col = detect_lat_lon(df_raw)

# 对于默认数据集的特征工程
is_default_ds = all(c in df_raw.columns for c in ["Station ID", "Charger Type", "Usage Stats (avg users/day)", "Cost (USD/kWh)"])

if is_default_ds:
    for col in ["Cost (USD/kWh)", "Distance to City (km)", "Usage Stats (avg users/day)",
                 "Charging Capacity (kW)", "Reviews (Rating)", "Parking Spots", "Installation Year"]:
        if col in df_raw.columns:
            df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")
    if "Address" in df_raw.columns:
        df_raw["City"] = df_raw["Address"].apply(lambda x: x.split(",")[-2].strip() if isinstance(x, str) else "Unknown")
    cap_bins = [0, 50, 150, 350, 1000]
    cap_labels = ["低容量 (<=50kW)", "中容量 (50-150kW)", "高容量 (150-350kW)", "超高容量 (>350kW)"]
    if "Charging Capacity (kW)" in df_raw.columns:
        df_raw["Capacity Level"] = pd.cut(df_raw["Charging Capacity (kW)"], bins=cap_bins, labels=cap_labels)
    usage_bins = [0, 30, 60, 100]
    usage_labels = ["低使用率", "中使用率", "高使用率"]
    if "Usage Stats (avg users/day)" in df_raw.columns:
        df_raw["Usage Level"] = pd.cut(df_raw["Usage Stats (avg users/day)"], bins=usage_bins, labels=usage_labels)

st.session_state.df = df_raw
df = df_raw

# 应用筛选器（从 session_state 读取）
for key, val in st.session_state.items():
    if key.startswith("filter_cat_"):
        col = key.replace("filter_cat_", "")
        if col in df.columns and len(val) < len(df[col].unique()):
            df = df[df[col].isin(val)]
    elif key.startswith("filter_num_"):
        col = key.replace("filter_num_", "")
        if col in df.columns:
            lo, hi = val
            df = df[(df[col] >= lo) & (df[col] <= hi)]

# ═══════════════════════════════════════════════════════════
# 顶部标题
# ═══════════════════════════════════════════════════════════
st.markdown("""
<div class="main-header">
    <div class="title-group">
        <h1>⚡ 新能源汽车充电桩空间布局与使用效率分析系统</h1>
        <p>数据可视化课程大作业 · 交互式充电基础设施分析平台</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# 5 Tabs: 数据清洗 + 4 个原有分析模块
# ═══════════════════════════════════════════════════════════
tab_clean, tab1, tab2, tab3, tab4 = st.tabs([
    "🧹 数据清洗",
    "🗺️ 空间布局",
    "📈 使用效率分析",
    "🔬 统计建模",
    "📋 数据下载与管理"
])

# ═══════════════════════════════════════════════════════════
# TAB 0: 数据清洗
# ═══════════════════════════════════════════════════════════
with tab_clean:
    st.markdown('<p class="section-title">🧹 数据质量诊断与智能清洗</p>', unsafe_allow_html=True)

    # ── 数据质量诊断 ──
    col_diag1, col_diag2, col_diag3, col_diag4 = st.columns(4)

    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isnull().sum().sum()
    missing_pct = (missing_cells / total_cells) * 100
    dup_count = df.duplicated().sum()
    dup_pct = (dup_count / len(df)) * 100

    with col_diag1:
        quality_class = "quality-good" if missing_pct < 1 else ("quality-warn" if missing_pct < 5 else "quality-bad")
        st.markdown(f"""
        <div class="clean-card" style="text-align:center;">
            <div style="font-size:2rem;">🔍</div>
            <div class="{quality_class}" style="font-size:1.5rem;">{missing_pct:.2f}%</div>
            <div style="color:rgba(255,255,255,0.5);">缺失值比例 ({missing_cells:,} / {total_cells:,})</div>
        </div>
        """, unsafe_allow_html=True)

    with col_diag2:
        quality_class = "quality-good" if dup_pct < 1 else ("quality-warn" if dup_pct < 5 else "quality-bad")
        st.markdown(f"""
        <div class="clean-card" style="text-align:center;">
            <div style="font-size:2rem;">📋</div>
            <div class="{quality_class}" style="font-size:1.5rem;">{dup_pct:.2f}%</div>
            <div style="color:rgba(255,255,255,0.5);">重复行比例 ({dup_count:,} 行)</div>
        </div>
        """, unsafe_allow_html=True)

    with col_diag3:
        mixed_types = sum(1 for col in df.columns if df[col].apply(type).nunique() > 1)
        quality_class = "quality-good" if mixed_types == 0 else "quality-warn"
        st.markdown(f"""
        <div class="clean-card" style="text-align:center;">
            <div style="font-size:2rem;">🔀</div>
            <div class="{quality_class}" style="font-size:1.5rem;">{mixed_types}</div>
            <div style="color:rgba(255,255,255,0.5);">混合类型列数</div>
        </div>
        """, unsafe_allow_html=True)

    with col_diag4:
        outlier_cols = 0
        for col in num_cols[:10]:
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                outliers = ((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum()
                if outliers > 0:
                    outlier_cols += 1
        quality_class = "quality-good" if outlier_cols <= 3 else "quality-warn"
        st.markdown(f"""
        <div class="clean-card" style="text-align:center;">
            <div style="font-size:2rem;">📊</div>
            <div class="{quality_class}" style="font-size:1.5rem;">{outlier_cols}</div>
            <div style="color:rgba(255,255,255,0.5);">含离群值列数 (IQR法)</div>
        </div>
        """, unsafe_allow_html=True)

    # ── 各列缺失值详情 ──
    st.markdown('<p class="section-title" style="margin-top:20px;">📋 各列数据质量明细</p>', unsafe_allow_html=True)
    missing_detail = []
    for col in df.columns:
        miss = df[col].isnull().sum()
        miss_p = miss / len(df) * 100
        dtype = str(df[col].dtype)
        nuniq = df[col].nunique()
        missing_detail.append([col, dtype, nuniq, miss, f"{miss_p:.1f}%", "✅" if miss == 0 else ("⚠️" if miss_p < 5 else "❌")])
    detail_df = pd.DataFrame(missing_detail, columns=["列名", "数据类型", "唯一值数", "缺失数", "缺失率", "状态"])
    st.dataframe(detail_df, use_container_width=True, height=280)

    # ── 清洗操作 ──
    st.markdown('<p class="section-title" style="margin-top:20px;">🛠️ 数据清洗操作</p>', unsafe_allow_html=True)

    col_clean1, col_clean2 = st.columns(2)

    with col_clean1:
        st.markdown("#### 缺失值处理")
        missing_method = st.selectbox(
            "选择缺失值处理策略",
            options=["不处理", "删除含缺失值的行", "用均值填充 (数值列)", "用中位数填充 (数值列)", "用众数填充 (分类列)", "用0填充", "向前填充 (ffill)", "向后填充 (bfill)"],
            key="missing_method"
        )

        st.markdown("#### 重复值处理")
        drop_duplicates = st.checkbox("删除完全重复的行", value=False, key="drop_dup")

        st.markdown("#### 列名标准化")
        clean_col_names = st.checkbox("去除列名前后的空格", value=True, key="clean_cols")
        lower_col_names = st.checkbox("列名转小写", value=False, key="lower_cols")

    with col_clean2:
        st.markdown("#### 离群值处理")
        outlier_method = st.selectbox(
            "选择离群值处理策略 (IQR方法)",
            options=["不处理", "删除离群值行 (数值列)", "用中位数替换离群值 (数值列)", "用边界值截断 (Winsorize)"],
            key="outlier_method"
        )

        st.markdown("#### 数据类型优化")
        auto_convert_types = st.checkbox("自动优化数据类型 (节省内存)", value=True, key="auto_types")

        st.markdown("#### 自定义操作")
        drop_cols = st.multiselect(
            "选择要删除的列",
            options=df.columns.tolist(),
            default=[],
            key="drop_cols"
        )

    # ── 执行清洗 ──
    if st.button("⚡ 执行数据清洗", use_container_width=True, type="primary"):
        df_clean = df.copy()
        log = []
        n_before = len(df_clean)

        # 列名处理
        if clean_col_names:
            df_clean.columns = df_clean.columns.str.strip()
            log.append("列名前后的空格已去除")
        if lower_col_names:
            df_clean.columns = [c.lower() for c in df_clean.columns]
            log.append("列名转为小写")

        # 删除指定列
        if drop_cols:
            df_clean = df_clean.drop(columns=[c for c in drop_cols if c in df_clean.columns], errors='ignore')
            log.append(f"已删除 {len(drop_cols)} 列: {', '.join(drop_cols[:5])}")

        # 缺失值处理
        if missing_method == "删除含缺失值的行":
            df_clean = df_clean.dropna()
            log.append(f"删除含缺失值行: {n_before - len(df_clean)} 行")
        elif missing_method == "用均值填充 (数值列)":
            for c in df_clean.select_dtypes(include=np.number).columns:
                df_clean[c] = df_clean[c].fillna(df_clean[c].mean())
            log.append("数值列缺失值用均值填充")
        elif missing_method == "用中位数填充 (数值列)":
            for c in df_clean.select_dtypes(include=np.number).columns:
                df_clean[c] = df_clean[c].fillna(df_clean[c].median())
            log.append("数值列缺失值用中位数填充")
        elif missing_method == "用众数填充 (分类列)":
            for c in df_clean.select_dtypes(exclude=np.number).columns:
                if not df_clean[c].mode().empty:
                    df_clean[c] = df_clean[c].fillna(df_clean[c].mode()[0])
            log.append("分类列缺失值用众数填充")
        elif missing_method == "用0填充":
            df_clean = df_clean.fillna(0)
            log.append("缺失值用0填充")
        elif missing_method == "向前填充 (ffill)":
            df_clean = df_clean.ffill()
            log.append("缺失值向前填充")
        elif missing_method == "向后填充 (bfill)":
            df_clean = df_clean.bfill()
            log.append("缺失值向后填充")

        # 重复值处理
        if drop_duplicates:
            before_dup = len(df_clean)
            df_clean = df_clean.drop_duplicates()
            log.append(f"删除重复行: {before_dup - len(df_clean)} 行")

        # 离群值处理
        if outlier_method != "不处理":
            num_cols_clean = df_clean.select_dtypes(include=np.number).columns
            outlier_count = 0
            for col in num_cols_clean[:15]:
                q1, q3 = df_clean[col].quantile(0.25), df_clean[col].quantile(0.75)
                iqr = q3 - q1
                if iqr > 0:
                    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                    outliers = (df_clean[col] < low) | (df_clean[col] > high)
                    if outliers.any():
                        outlier_count += outliers.sum()
                        if outlier_method == "删除离群值行 (数值列)":
                            df_clean = df_clean[~outliers]
                        elif outlier_method == "用中位数替换离群值 (数值列)":
                            df_clean.loc[outliers, col] = df_clean[col].median()
                        elif outlier_method == "用边界值截断 (Winsorize)":
                            df_clean[col] = df_clean[col].clip(low, high)
            if outlier_count > 0:
                log.append(f"处理离群值: {outlier_count} 个数据点")

        # 数据类型优化
        if auto_convert_types:
            df_clean = df_clean.convert_dtypes()
            log.append("数据类型已优化")

        # 更新状态
        st.session_state.df = df_clean
        st.session_state.cleaned = True
        st.session_state.clean_log = log

        # 展示清洗结果
        st.success(f"✅ 清洗完成！{n_before:,} → {len(df_clean):,} 行 (变化 {len(df_clean)-n_before:+,})")
        for l in log:
            st.caption(f"  • {l}")

        st.rerun()

    # 重置清洗
    if st.session_state.cleaned:
        if st.button("🔄 撤销清洗 (恢复原始数据)", use_container_width=True):
            if st.session_state.raw_df is not None:
                st.session_state.df = st.session_state.raw_df.copy()
                st.session_state.cleaned = False
                st.session_state.clean_log = []
                st.rerun()

# ═══════════════════════════════════════════════════════════
# TAB 1: 空间布局
# ═══════════════════════════════════════════════════════════
with tab1:
    df_cn = cn_df(df)  # 中文化副本用于图表

    # ── KPI 摘要卡片 ──
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">🔌</div>
            <div class="metric-value">{len(df):,}</div>
            <div class="metric-label">当前站点数</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi2:
        n_cities = df["City"].nunique() if "City" in df.columns else df_cn["城市"].nunique() if "城市" in df.columns else "-"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">🏙️</div>
            <div class="metric-value">{n_cities}</div>
            <div class="metric-label">覆盖城市</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi3:
        avg_usage = df["Usage Stats (avg users/day)"].mean() if "Usage Stats (avg users/day)" in df.columns else "-"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">👥</div>
            <div class="metric-value">{avg_usage:.0f}</div>
            <div class="metric-label">日均用户 (均值)</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi4:
        avg_cost = df["Cost (USD/kWh)"].mean() if "Cost (USD/kWh)" in df.columns else "-"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">💰</div>
            <div class="metric-value">${avg_cost:.2f}</div>
            <div class="metric-label">平均充电成本/kWh</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    if lat_col and lon_col:
        col_left, col_right = st.columns([5, 3])

        with col_left:
            st.markdown('<p class="section-title">🗺️ 充电站地理空间分布</p>', unsafe_allow_html=True)

            # 使用颜色和大小映射
            color_col = None
            size_col = None
            for c in cat_cols:
                if any(k in c.lower() for k in ['type', 'charger', '类型', 'charger type']):
                    color_col = c; break
            if not color_col and cat_cols:
                color_col = cat_cols[0]
            for c in num_cols:
                if any(k in c.lower() for k in ['usage', 'user', '使用', 'user']):
                    size_col = c; break
            if not size_col and num_cols:
                size_col = num_cols[0] if num_cols else None

            # 用中文副本绘图，但经纬度保留原列名
            map_df_cn = df_cn.copy()
            map_df_cn[lat_col] = df[lat_col]  # lat/lon数值不变
            map_df_cn[lon_col] = df[lon_col]

            fig_map = px.scatter_geo(
                map_df_cn, lat=lat_col, lon=lon_col,
                color=cn_label(color_col) if color_col else None,
                size=cn_label(size_col) if size_col else None,
                hover_name=cn_label(text_cols[0]) if text_cols else None,
                color_discrete_sequence=px.colors.qualitative.Bold,
                projection="natural earth",
                height=500,
                opacity=0.8,
            )
            fig_map.update_geos(
                showcountries=True, countrycolor="rgba(255,255,255,0.15)",
                showocean=True, oceancolor="rgba(20,30,70,0.8)",
                showland=True, landcolor="rgba(15,20,45,0.9)",
                bgcolor="rgba(0,0,0,0)",
            )
            fig_map.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="white",
                legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5, font=dict(size=11)),
                margin=dict(l=0, r=0, t=20, b=10),
            )
            st.plotly_chart(fig_map, use_container_width=True, config={"displayModeBar": False})

        with col_right:
            if cat_cols:
                st.markdown('<p class="section-title">🏙️ 分类分布 TOP10</p>', unsafe_allow_html=True)
                main_cat = cat_cols[0] if cat_cols else None
                if main_cat:
                    counts = df_cn[cn_label(main_cat)].value_counts().head(10)
                    fig_bar = px.bar(
                        x=counts.values, y=counts.index.astype(str), orientation="h",
                        color=counts.values, color_continuous_scale="Blues",
                        text=counts.values,
                    )
                    fig_bar.update_traces(textposition="outside", marker=dict(line=dict(width=0)))
                    fig_bar.update_layout(
                        height=370, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font_color="white",
                        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)", zeroline=False, title=None),
                        yaxis=dict(title=None, tickfont=dict(size=12)),
                        margin=dict(l=0, r=40, t=10, b=0), coloraxis_showscale=False,
                    )
                    st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

        if num_cols:
            st.markdown('<p class="section-title">📐 数值分布</p>', unsafe_allow_html=True)
            cols_dist = st.columns(min(3, len(num_cols[:6])))
            for i, col in enumerate(num_cols[:6]):
                with cols_dist[i % 3]:
                    fig = px.histogram(
                        df_cn, x=cn_label(col), nbins=30,
                        color_discrete_sequence=[f"rgba({50+i*30}, {180-i*20}, {255-i*30}, 0.7)"],
                    )
                    fig.update_layout(
                        height=250, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font_color="white", title=cn_label(col), title_font_size=11,
                        xaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
                        yaxis=dict(title="频次", gridcolor="rgba(255,255,255,0.08)"),
                        margin=dict(l=0, r=0, t=30, b=0),
                    )
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    else:
        # 没有经纬度列时，显示替代视图
        st.info("💡 当前数据中未检测到经纬度字段，无法显示地图。以下展示数据的整体分布概览。")

        if num_cols:
            cols = st.columns(min(4, len(num_cols)))
            for i, col in enumerate(num_cols[:8]):
                with cols[i % 4]:
                    fig = px.histogram(df_cn, x=cn_label(col), nbins=30, title=cn_label(col),
                                       color_discrete_sequence=px.colors.qualitative.Pastel)
                    fig.update_layout(height=250, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                      font_color="white", title_font_size=10, margin=dict(l=0,r=0,t=25,b=0))
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        if cat_cols:
            st.markdown('<p class="section-title">📊 分类变量分布</p>', unsafe_allow_html=True)
            cols_cat = st.columns(min(3, len(cat_cols)))
            for i, col in enumerate(cat_cols[:6]):
                with cols_cat[i % 3]:
                    counts = df_cn[cn_label(col)].value_counts().head(10)
                    fig = px.bar(x=counts.values, y=counts.index.astype(str), orientation="h",
                                 title=cn_label(col), color=counts.values, color_continuous_scale="Blues")
                    fig.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                      font_color="white", title_font_size=10, margin=dict(l=0,r=0,t=25,b=0),
                                      coloraxis_showscale=False)
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ═══════════════════════════════════════════════════════════
# TAB 2: 使用效率分析 (保留原有逻辑，自适应列名)
# ═══════════════════════════════════════════════════════════
with tab2:
    df_cn = cn_df(df)
    st.markdown('<p class="section-title">📈 变量关系与效率分析</p>', unsafe_allow_html=True)

    # ── KPI 摘要卡片 ──
    if len(num_cols) >= 2:
        ke1, ke2, ke3, ke4 = st.columns(4)
        with ke1:
            avg_cap = df["Charging Capacity (kW)"].mean() if "Charging Capacity (kW)" in df.columns else 0
            st.markdown(f"""<div class="metric-card"><div class="metric-icon">⚡</div>
                <div class="metric-value">{avg_cap:.0f} kW</div>
                <div class="metric-label">平均充电容量</div></div>""", unsafe_allow_html=True)
        with ke2:
            avg_use = df["Usage Stats (avg users/day)"].mean() if "Usage Stats (avg users/day)" in df.columns else 0
            st.markdown(f"""<div class="metric-card"><div class="metric-icon">👥</div>
                <div class="metric-value">{avg_use:.0f}</div>
                <div class="metric-label">日均用户 (均值)</div></div>""", unsafe_allow_html=True)
        with ke3:
            avg_rate = df["Reviews (Rating)"].mean() if "Reviews (Rating)" in df.columns else 0
            st.markdown(f"""<div class="metric-card"><div class="metric-icon">⭐</div>
                <div class="metric-value">{avg_rate:.2f}</div>
                <div class="metric-label">平均评分</div></div>""", unsafe_allow_html=True)
        with ke4:
            n_ops = df["Station Operator"].nunique() if "Station Operator" in df.columns else 0
            st.markdown(f"""<div class="metric-card"><div class="metric-icon">🏢</div>
                <div class="metric-value">{n_ops}</div>
                <div class="metric-label">运营商数量</div></div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    if len(num_cols) >= 2:
        col_eff1, col_eff2 = st.columns(2)

        with col_eff1:
            # 数值列两两散点图 - 用中文标签
            x_col = st.selectbox("X 轴变量", options=[cn_label(c) for c in num_cols], index=0, key="eff_x")
            y_col = st.selectbox("Y 轴变量", options=[cn_label(c) for c in num_cols], index=min(1, len(num_cols)-1), key="eff_y")
            color_opt_cn = st.selectbox("颜色分组", options=["无"] + [cn_label(c) for c in cat_cols], index=0, key="eff_color")

            # 找回原始列名
            x_orig = next((c for c in num_cols if cn_label(c) == x_col), x_col)
            y_orig = next((c for c in num_cols if cn_label(c) == y_col), y_col)
            color_orig = None if color_opt_cn == "无" else next((c for c in cat_cols if cn_label(c) == color_opt_cn), color_opt_cn)

            fig_sc = px.scatter(
                df_cn, x=x_col, y=y_col,
                color=None if color_opt_cn == "无" else color_opt_cn,
                color_discrete_sequence=px.colors.qualitative.Bold,
                opacity=0.7,
                trendline="ols",
            )
            fig_sc.update_layout(
                height=400, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="white",
                xaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                margin=dict(l=0, r=0, t=10, b=0),
            )
            st.plotly_chart(fig_sc, use_container_width=True, config={"displayModeBar": False})

        with col_eff2:
            if cat_cols:
                cat_choice_cn = st.selectbox("分组变量", options=[cn_label(c) for c in cat_cols], key="eff_cat")
                cat_orig = next((c for c in cat_cols if cn_label(c) == cat_choice_cn), cat_choice_cn)
                agg_choice_cn = st.selectbox("聚合指标", options=[cn_label(c) for c in num_cols], index=min(1, len(num_cols)-1), key="eff_agg")
                agg_orig = next((c for c in num_cols if cn_label(c) == agg_choice_cn), agg_choice_cn)
                agg_func_cn = st.selectbox("聚合方式", options=["均值", "中位数", "总和", "最大值", "最小值", "计数"], key="eff_func")
                func_map = {"均值": "mean", "中位数": "median", "总和": "sum", "最大值": "max", "最小值": "min", "计数": "count"}

                grouped = df_cn.groupby(cat_choice_cn)[agg_choice_cn].agg(func_map[agg_func_cn]).reset_index().sort_values(agg_choice_cn)

                fig_grp = px.bar(
                    grouped, x=agg_choice_cn, y=cat_choice_cn, orientation="h",
                    color=cat_choice_cn, color_discrete_sequence=px.colors.qualitative.Pastel,
                    text=grouped[agg_choice_cn].round(1),
                )
                fig_grp.update_traces(textposition="outside", marker=dict(line=dict(width=0)))
                fig_grp.update_layout(
                    height=400, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_color="white",
                    xaxis=dict(showgrid=False, showticklabels=False, title=None),
                    yaxis=dict(title=None),
                    margin=dict(l=0, r=60, t=10, b=0),
                    coloraxis_showscale=False,
                )
                st.plotly_chart(fig_grp, use_container_width=True, config={"displayModeBar": False})

    # 相关性热力图
    if len(num_cols) >= 3:
        st.markdown('<p class="section-title">🔗 数值变量相关性热力图</p>', unsafe_allow_html=True)
        corr = df[num_cols[:12]].corr()
        # 用中文标签
        corr.index = [cn_label(c) for c in corr.index]
        corr.columns = [cn_label(c) for c in corr.columns]
        fig_heat = px.imshow(
            corr, text_auto=".2f", aspect="auto",
            color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        )
        fig_heat.update_layout(
            height=500, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="white", margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig_heat, use_container_width=True, config={"displayModeBar": False})

    else:
        st.info("数值变量不足（需要 3 个以上），部分分析图不可用。")

# ═══════════════════════════════════════════════════════════
# TAB 3: 统计建模 (保留K-Means和回归)
# ═══════════════════════════════════════════════════════════
with tab3:
    df_cn = cn_df(df)
    st.markdown('<p class="section-title">🔬 机器学习建模分析</p>', unsafe_allow_html=True)

    # ── KPI 摘要卡片 ──
    if len(num_cols) >= 2:
        km1, km2, km3, km4 = st.columns(4)
        with km1:
            st.markdown(f"""<div class="metric-card"><div class="metric-icon">📊</div>
                <div class="metric-value">{len(num_cols)}</div>
                <div class="metric-label">可用数值特征</div></div>""", unsafe_allow_html=True)
        with km2:
            st.markdown(f"""<div class="metric-card"><div class="metric-icon">🏷️</div>
                <div class="metric-value">{len(cat_cols)}</div>
                <div class="metric-label">可用分类特征</div></div>""", unsafe_allow_html=True)
        with km3:
            st.markdown(f"""<div class="metric-card"><div class="metric-icon">📈</div>
                <div class="metric-value">{len(df):,}</div>
                <div class="metric-label">建模样本数</div></div>""", unsafe_allow_html=True)
        with km4:
            r2_disp = f"{reg.score(X_reg, y_reg):.3f}" if 'X_reg' in dir() and len(num_cols) >= 4 else "-"
            st.markdown(f"""<div class="metric-card"><div class="metric-icon">🎯</div>
                <div class="metric-value">{r2_disp}</div>
                <div class="metric-label">默认模型 R²</div></div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    if len(num_cols) >= 4 and len(df) >= 20:
        col_m1, col_m2 = st.columns(2)

        with col_m1:
            st.markdown("#### 📌 K-Means 聚类分析")

            # 选择聚类特征 - 显示中文
            cn_feat_options = [cn_label(c) for c in num_cols]
            cn_feat_default = [cn_label(c) for c in num_cols[:min(5, len(num_cols))]]
            cluster_features_cn = st.multiselect(
                "选择聚类特征 (建议3-6个)",
                options=cn_feat_options,
                default=cn_feat_default,
                key="cluster_feats"
            )
            n_clusters = st.slider("聚类数量", 2, 8, 4, key="n_kmeans")

            if len(cluster_features_cn) >= 2:
                # 找回原始列名
                cluster_features_orig = [next((c for c in num_cols if cn_label(c) == f), f) for f in cluster_features_cn]
                X_cl = df[cluster_features_orig].dropna()
                if len(X_cl) >= n_clusters * 10:
                    scaler = StandardScaler()
                    X_sc = scaler.fit_transform(X_cl)
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    labels = kmeans.fit_predict(X_sc)

                    X_plot = X_cl.copy()
                    X_plot.columns = [cn_label(c) for c in X_plot.columns]
                    X_plot["聚类"] = [f"第{c+1}类" for c in labels]

                    fig_cl = px.scatter(
                        X_plot, x=cluster_features_cn[0], y=cluster_features_cn[1],
                        color="聚类", size=cluster_features_cn[2] if len(cluster_features_cn) > 2 else None,
                        color_discrete_sequence=px.colors.qualitative.Bold, opacity=0.7,
                    )
                    centers = scaler.inverse_transform(kmeans.cluster_centers_)
                    fig_cl.add_trace(go.Scatter(
                        x=centers[:, 0], y=centers[:, 1],
                        mode="markers", name="聚类中心",
                        marker=dict(size=16, color="white", symbol="star", line=dict(width=3, color="gold")),
                    ))
                    fig_cl.update_layout(
                        height=400, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font_color="white",
                        xaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
                        yaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02),
                        margin=dict(l=0, r=0, t=10, b=0),
                    )
                    st.plotly_chart(fig_cl, use_container_width=True, config={"displayModeBar": False})

                    # 聚类统计 - 中文列名
                    cluster_stats = X_plot.groupby("聚类").agg(
                        **{f"平均{cn_label(c)}": (cn_label(c), "mean") for c in cluster_features_orig},
                        样本数=("聚类", "count"),
                    ).round(2)
                    st.dataframe(cluster_stats, use_container_width=True)
                else:
                    st.warning("清洗后有效样本不足，无法进行聚类分析")

        with col_m2:
            st.markdown("#### 📈 线性回归分析")

            y_col_cn = st.selectbox("目标变量", options=[cn_label(c) for c in num_cols], key="reg_y")
            y_orig = next((c for c in num_cols if cn_label(c) == y_col_cn), y_col_cn)
            x_opts_cn = [cn_label(c) for c in num_cols if c != y_orig]
            x_default_cn = [cn_label(c) for c in num_cols[:min(4, len(num_cols))] if c != y_orig]
            x_cols_cn = st.multiselect("特征变量", options=x_opts_cn, default=x_default_cn, key="reg_x")

            if x_cols_cn and y_col_cn:
                x_orig_list = [next((c for c in num_cols if cn_label(c) == f), f) for f in x_cols_cn]
                X_r = df[x_orig_list].dropna()
                y_r = df.loc[X_r.index, y_orig].dropna()
                X_r = X_r.loc[y_r.index]

                if len(X_r) > 20:
                    reg = LinearRegression()
                    reg.fit(X_r, y_r)

                    coef_df = pd.DataFrame({
                        "特征": x_cols_cn,
                        "影响系数": reg.coef_.round(4),
                    }).sort_values("影响系数", ascending=True)

                    fig_cf = px.bar(
                        coef_df, x="影响系数", y="特征", orientation="h",
                        color="影响系数", color_continuous_scale="RdBu",
                        text=coef_df["影响系数"].round(4), color_continuous_midpoint=0,
                    )
                    fig_cf.update_traces(textposition="outside", marker=dict(line=dict(width=0)))
                    fig_cf.update_layout(
                        height=320, title=f"各特征对「{y_col_cn}」的影响系数",
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font_color="white",
                        xaxis=dict(title="回归系数", zeroline=True, zerolinecolor="rgba(255,255,255,0.3)", gridcolor="rgba(255,255,255,0.08)"),
                        yaxis=dict(title=None),
                        margin=dict(l=0, r=50, t=30, b=0), coloraxis_showscale=False,
                    )
                    st.plotly_chart(fig_cf, use_container_width=True, config={"displayModeBar": False})

                    r2 = reg.score(X_r, y_r)
                    st.markdown(f"""
                    <div style="background:rgba(0,210,255,0.08); border:1px solid rgba(0,210,255,0.2);
                                border-radius:10px; padding:14px 18px; margin-top:8px;">
                        <strong>📊 模型摘要</strong><br>
                        决定系数 R²：<strong>{r2:.4f}</strong><br>
                        截距：<strong>{reg.intercept_:.2f}</strong><br>
                        训练样本数：<strong>{len(X_r):,}</strong>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("有效样本不足，无法建立回归模型")

    else:
        st.info(f"需要至少 4 个数值列和 20 条样本才能进行建模分析。当前：{len(num_cols)} 个数值列，{len(df)} 条样本。")

# ═══════════════════════════════════════════════════════════
# TAB 4: 数据下载与管理
# ═══════════════════════════════════════════════════════════
with tab4:
    st.markdown('<p class="section-title">📋 数据预览与多格式导出</p>', unsafe_allow_html=True)

    col_dl1, col_dl2 = st.columns([3, 1])

    with col_dl1:
        st.markdown("#### 📊 数据预览")
        n_preview = st.slider("预览行数", 5, min(100, len(df)), min(20, len(df)), key="preview_n")
        st.dataframe(df.head(n_preview), use_container_width=True, height=400)

        st.markdown("#### 📈 统计摘要")
        st.dataframe(df.describe(include='all').round(2), use_container_width=True, height=250)

    with col_dl2:
        st.markdown("#### 📥 数据导出")

        # CSV 导出
        csv_data = df.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="📥 下载 CSV",
            data=csv_data,
            file_name="ev_charging_data_cleaned.csv",
            mime="text/csv",
            use_container_width=True,
        )

        # JSON 导出
        json_data = df.to_json(orient="records", force_ascii=False)
        st.download_button(
            label="📥 下载 JSON",
            data=json_data,
            file_name="ev_charging_data_cleaned.json",
            mime="application/json",
            use_container_width=True,
        )

        # Excel 导出
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Cleaned Data')
        excel_data = buffer.getvalue()
        st.download_button(
            label="📥 下载 Excel (.xlsx)",
            data=excel_data,
            file_name="ev_charging_data_cleaned.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

        # Parquet 导出
        parquet_buffer = BytesIO()
        df.to_parquet(parquet_buffer, index=False)
        parquet_data = parquet_buffer.getvalue()
        st.download_button(
            label="📥 下载 Parquet (.parquet)",
            data=parquet_data,
            file_name="ev_charging_data_cleaned.parquet",
            mime="application/octet-stream",
            use_container_width=True,
        )

        st.markdown("---")

        st.markdown("#### ℹ️ 数据信息")
        st.caption(f"行数: {len(df):,}")
        st.caption(f"列数: {len(df.columns)}")
        st.caption(f"缺失值: {df.isnull().sum().sum():,}")
        st.caption(f"重复行: {df.duplicated().sum():,}")
        mem_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        st.caption(f"内存占用: {mem_mb:.1f} MB")

        st.markdown("---")
        st.markdown("#### 📤 下载原始数据")
        if st.session_state.raw_df is not None:
            raw_csv = st.session_state.raw_df.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="📥 原始数据 (CSV)",
                data=raw_csv,
                file_name="ev_charging_data_raw.csv",
                mime="text/csv",
                use_container_width=True,
            )

        st.markdown("---")
        if st.button("🗑️ 清除所有数据", use_container_width=True):
            st.session_state.df = None
            st.session_state.raw_df = None
            st.session_state.file_loaded = False
            st.session_state.cleaned = False
            st.session_state.clean_log = []
            st.session_state.source_name = ""
            for key in list(st.session_state.keys()):
                if key.startswith("filter_"):
                    del st.session_state[key]
            st.rerun()

# ═══════════════════════════════════════════════════════════
# 页脚
# ═══════════════════════════════════════════════════════════
st.markdown("""
<div class="footer">
    ⚡ 新能源汽车充电桩空间布局与使用效率分析系统 &nbsp;|&nbsp;
    数据可视化课程大作业 &nbsp;|&nbsp;
    基于 Streamlit + Plotly 构建<br>
    © 2026 充电基础设施可视化分析平台 v2.0
</div>
""", unsafe_allow_html=True)
