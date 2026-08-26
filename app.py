"""
🛠️ NASA RUL Predictor — Predictive Maintenance Dashboard
============================================================
نسخة محسّنة من تطبيق التنبؤ بالعمر الافتراضي المتبقي (RUL) للمحركات.

المميزات الجديدة مقارنة بالنسخة الأصلية:
  • واجهة منظمة على شكل تبويبات (نظرة عامة / تفاصيل / رسوم تفاعلية / تنبيهات)
  • تصنيف تلقائي لحالة كل محرك (سليم 🟢 / تحذير 🟡 / حرج 🔴) حسب عتبات قابلة للتعديل من الشريط الجانبي
  • رسوم بيانية تفاعلية باستخدام Plotly بدل matplotlib (zoom / hover / تصدير كصورة)
  • بطاقات إحصائية (KPI cards) تلخص متوسط/أقل/أعلى RUL وعدد المحركات الحرجة
  • عرض معلومات الموديل (metadata) من الـ artifact نفسه
  • فلترة وترتيب تفاعلي للجدول (حسب الحالة أو حسب قيمة RUL)
  • دعم تحميل النتائج كـ CSV أو Excel
  • تحقق أكثر تفصيلاً من الأعمدة الناقصة + رسالة توضيحية للأعمدة المتاحة
  • رسم Trend لكل محرك عبر الزمن (إذا توفر عمود دوري cycle/time)
  • Progress bar ومؤشرات تحميل واضحة
  • تنبيهات صوتية/بصرية للمحركات اللي قربت من نهاية عمرها الافتراضي
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from rul_predictor.models.artifact import load_artifact

# ------------------------------------------------------------------
# إعدادات الصفحة العامة
# ------------------------------------------------------------------
st.set_page_config(
    page_title="NASA RUL Predictor",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# تنسيق CSS بسيط لتحسين شكل البطاقات
# ------------------------------------------------------------------
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 12px;
        padding: 18px;
        text-align: center;
        border: 1px solid #e9ecef;
    }
    .status-critical { color: #d62728; font-weight: bold; }
    .status-warning  { color: #ff9800; font-weight: bold; }
    .status-healthy  { color: #2ca02c; font-weight: bold; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🛠️ Predictive Maintenance — RUL Predictor")
st.caption("رفع ملف البيانات المعالجة لحساب المتبقي من العمر الافتراضي (RUL) لكل محرك، مع تحليل تفاعلي كامل.")

# ------------------------------------------------------------------
# 1) تحميل الـ Artifact (Model + Metadata)
# ------------------------------------------------------------------
@st.cache_resource(show_spinner="⏳ جاري تحميل الموديل...")
def get_model():
    return load_artifact("artifacts/rul_model.joblib")

try:
    artifact = get_model()
    st.success("✅ تم تحميل الموديل والـ Artifact بنجاح!")
except Exception as e:
    st.error(f"❌ تعذر تحميل الموديل: {e}")
    st.stop()

# ------------------------------------------------------------------
# الشريط الجانبي: إعدادات + معلومات الموديل
# ------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ الإعدادات")

    rul_cap_default = artifact.metadata.get("rul_cap", 125)
    rul_cap = st.number_input(
        "🔧 الحد الأقصى لـ RUL (Cap)",
        min_value=1,
        value=int(rul_cap_default),
        help="أي قيمة تنبؤ أعلى من هذا الرقم سيتم تقليمها لهذه القيمة.",
    )

    st.subheader("🚦 عتبات تصنيف الحالة")
    warning_threshold = st.slider("عتبة التحذير 🟡 (أقل من)", 0, int(rul_cap), min(30, int(rul_cap)))
    critical_threshold = st.slider("عتبة الحرج 🔴 (أقل من)", 0, warning_threshold, min(10, warning_threshold))

    st.divider()
    st.subheader("ℹ️ معلومات الموديل")
    model_name = artifact.metadata.get("model_name", type(artifact.model).__name__)
    trained_on = artifact.metadata.get("trained_on", "غير متوفر")
    version = artifact.metadata.get("version", "غير متوفر")
    st.write(f"**اسم الموديل:** {model_name}")
    st.write(f"**تاريخ/بيانات التدريب:** {trained_on}")
    st.write(f"**الإصدار:** {version}")
    st.write(f"**عدد الأعمدة المطلوبة:** {len(artifact.feature_columns)}")
    with st.expander("عرض الأعمدة المطلوبة كاملة"):
        st.code(", ".join(artifact.feature_columns))

# ------------------------------------------------------------------
# دالة تصنيف حالة المحرك
# ------------------------------------------------------------------
def classify_status(rul: float) -> str:
    if rul < critical_threshold:
        return "🔴 حرج"
    elif rul < warning_threshold:
        return "🟡 تحذير"
    return "🟢 سليم"


STATUS_COLOR_MAP = {"🔴 حرج": "#d62728", "🟡 تحذير": "#ff9800", "🟢 سليم": "#2ca02c"}

# ------------------------------------------------------------------
# 2) رفع الملف (يدعم أكثر من ملف)
# ------------------------------------------------------------------
uploaded_files = st.file_uploader(
    "اختر ملف/ملفات البيانات المعالجة (CSV)",
    type=["csv"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.info("👆 من فضلك ارفع ملف CSV واحد أو أكثر لبدء التحليل.")
    st.stop()

# دمج الملفات المرفوعة في DataFrame واحد مع عمود يوضح مصدر الملف
dfs = []
for f in uploaded_files:
    tmp = pd.read_csv(f)
    tmp["__source_file__"] = f.name
    dfs.append(tmp)
df = pd.concat(dfs, ignore_index=True)

st.write(f"📁 تم رفع **{len(uploaded_files)}** ملف / ملفات — إجمالي **{len(df)}** صف.")
with st.expander("📊 معاينة البيانات الخام"):
    st.dataframe(df.head(20), use_container_width=True)

# ------------------------------------------------------------------
# 3) التحقق من الأعمدة المطلوبة
# ------------------------------------------------------------------
missing_cols = [col for col in artifact.feature_columns if col not in df.columns]

if missing_cols:
    st.error(f"❌ الأعمدة التالية ناقصة في الملف: {missing_cols}")
    st.warning("الأعمدة المتاحة في ملفك حالياً:")
    st.code(", ".join(df.columns.tolist()))
    st.stop()

# ------------------------------------------------------------------
# 4) التنبؤ
# ------------------------------------------------------------------
with st.spinner("🔮 جاري حساب التنبؤات..."):
    X = df[artifact.feature_columns]
    raw_preds = artifact.model.predict(X)
    df["Predicted_RUL"] = np.minimum(raw_preds.astype(float), rul_cap)
    df["Status"] = df["Predicted_RUL"].apply(classify_status)

st.toast("تم حساب التنبؤات بنجاح ✅", icon="🎉")

# ------------------------------------------------------------------
# تجهيز جدول ملخّص لكل محرك (آخر قراءة لكل unit_id)
# ------------------------------------------------------------------
has_unit_id = "unit_id" in df.columns
if has_unit_id:
    summary_df = df.groupby("unit_id", as_index=False).last()
    summary_df = summary_df[["unit_id", "Predicted_RUL", "Status"]].sort_values("Predicted_RUL")
else:
    summary_df = df[["Predicted_RUL", "Status"]].copy()

# ------------------------------------------------------------------
# بطاقات KPI
# ------------------------------------------------------------------
st.subheader("📌 نظرة عامة سريعة")
k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("متوسط RUL", f"{summary_df['Predicted_RUL'].mean():.1f}")
    st.markdown("</div>", unsafe_allow_html=True)

with k2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("أقل قيمة RUL", f"{summary_df['Predicted_RUL'].min():.1f}")
    st.markdown("</div>", unsafe_allow_html=True)

with k3:
    n_critical = (summary_df["Status"] == "🔴 حرج").sum()
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("🔴 محركات حرجة", int(n_critical))
    st.markdown("</div>", unsafe_allow_html=True)

with k4:
    n_warning = (summary_df["Status"] == "🟡 تحذير").sum()
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("🟡 محركات تحذير", int(n_warning))
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------------------------
# تنبيه فوري لو فيه محركات حرجة
# ------------------------------------------------------------------
if n_critical > 0:
    critical_units = (
        summary_df.loc[summary_df["Status"] == "🔴 حرج", "unit_id"].tolist()
        if has_unit_id
        else []
    )
    msg = "⚠️ يوجد محركات في حالة حرجة تحتاج صيانة عاجلة"
    if critical_units:
        msg += f": {critical_units}"
    st.error(msg)

# ------------------------------------------------------------------
# التبويبات الرئيسية
# ------------------------------------------------------------------
tab_overview, tab_details, tab_charts, tab_trend, tab_download = st.tabs(
    ["📋 الملخص", "🔎 التفاصيل الكاملة", "📈 رسوم تفاعلية", "⏱️ التطور الزمني", "📥 التحميل"]
)

# --- تبويب الملخص ---
with tab_overview:
    st.subheader("جدول ملخّص حالة كل محرك")

    colf1, colf2 = st.columns(2)
    with colf1:
        status_filter = st.multiselect(
            "فلترة حسب الحالة",
            options=["🔴 حرج", "🟡 تحذير", "🟢 سليم"],
            default=["🔴 حرج", "🟡 تحذير", "🟢 سليم"],
        )
    with colf2:
        sort_order = st.radio("ترتيب حسب RUL", ["تصاعدي (الأخطر أولاً)", "تنازلي"], horizontal=True)

    filtered = summary_df[summary_df["Status"].isin(status_filter)]
    filtered = filtered.sort_values(
        "Predicted_RUL", ascending=(sort_order == "تصاعدي (الأخطر أولاً)")
    )

    st.dataframe(
        filtered.style.applymap(
            lambda v: f"color: {STATUS_COLOR_MAP.get(v, 'black')}; font-weight:bold"
            if v in STATUS_COLOR_MAP
            else "",
            subset=["Status"],
        ),
        use_container_width=True,
        height=420,
    )

# --- تبويب التفاصيل الكاملة ---
with tab_details:
    st.subheader("كل الصفوف والأعمدة بعد التنبؤ")
    st.dataframe(df, use_container_width=True, height=500)

# --- تبويب الرسوم التفاعلية ---
with tab_charts:
    st.subheader("توزيع RUL حسب المحرك")

    if has_unit_id:
        bar_df = summary_df.sort_values("Predicted_RUL")
        fig_bar = px.bar(
            bar_df,
            x="unit_id",
            y="Predicted_RUL",
            color="Status",
            color_discrete_map=STATUS_COLOR_MAP,
            title="Predicted RUL حسب المحرك",
            labels={"unit_id": "Unit ID", "Predicted_RUL": "Predicted RUL"},
        )
        fig_bar.add_hline(
            y=warning_threshold, line_dash="dash", line_color="#ff9800",
            annotation_text="عتبة التحذير",
        )
        fig_bar.add_hline(
            y=critical_threshold, line_dash="dash", line_color="#d62728",
            annotation_text="عتبة الحرج",
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        fig_line = px.line(df, y="Predicted_RUL", title="Predicted RUL عبر الصفوف")
        st.plotly_chart(fig_line, use_container_width=True)

    st.subheader("توزيع القيم (Histogram)")
    fig_hist = px.histogram(
        summary_df, x="Predicted_RUL", nbins=20, color="Status",
        color_discrete_map=STATUS_COLOR_MAP, title="Histogram لقيم RUL المتوقعة",
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    st.subheader("نسبة المحركات لكل حالة")
    pie_data = summary_df["Status"].value_counts().reset_index()
    pie_data.columns = ["Status", "Count"]
    fig_pie = px.pie(
        pie_data, names="Status", values="Count",
        color="Status", color_discrete_map=STATUS_COLOR_MAP,
    )
    st.plotly_chart(fig_pie, use_container_width=True)

# --- تبويب التطور الزمني (Trend) ---
with tab_trend:
    st.subheader("تطور RUL عبر الزمن لكل محرك")
    time_cols = [c for c in df.columns if c.lower() in ("cycle", "time", "time_in_cycles", "timestamp")]

    if has_unit_id and time_cols:
        time_col = st.selectbox("اختر عمود الزمن/الدورة", time_cols)
        unit_options = sorted(df["unit_id"].unique().tolist())
        selected_units = st.multiselect(
            "اختر المحركات لعرض الترند", unit_options, default=unit_options[: min(5, len(unit_options))]
        )
        trend_df = df[df["unit_id"].isin(selected_units)]
        fig_trend = px.line(
            trend_df.sort_values(time_col),
            x=time_col, y="Predicted_RUL", color="unit_id",
            title="RUL Trend لكل محرك عبر الزمن",
        )
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.info("لا يوجد عمود زمني (cycle/time) في البيانات لعرض الترند، أو عمود unit_id غير موجود.")

# --- تبويب التحميل ---
with tab_download:
    st.subheader("تحميل النتائج")

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 تحميل النتائج الكاملة كملف CSV",
        data=csv_bytes,
        file_name=f"rul_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )

    summary_csv = summary_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 تحميل جدول الملخص فقط (CSV)",
        data=summary_csv,
        file_name=f"rul_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )

    try:
        import io
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
            df.to_excel(writer, index=False, sheet_name="Full_Predictions")
            summary_df.to_excel(writer, index=False, sheet_name="Summary")
        st.download_button(
            "📥 تحميل النتائج كملف Excel (بشيتين)",
            data=buffer.getvalue(),
            file_name=f"rul_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    except Exception:
        st.caption("ℹ️ لتفعيل التحميل بصيغة Excel ثبّت مكتبة `xlsxwriter` (pip install xlsxwriter).")

st.divider()
st.caption("تم التطوير بواسطة نموذج Predictive Maintenance — RUL Predictor Dashboard 🚀")