import streamlit as st
import joblib
import numpy as np
import pandas as pd
from preprocessing import preprocessing_clean_text

# Cấu hình trang
st.set_page_config(
    page_title="Toxic Comment Classifier",
    page_icon="🛡️",
    layout="wide"
)

# Load mô hình đã huấn luyện
@st.cache_resource
def load_model():
    """Load mô hình và pipeline đã được huấn luyện"""
    try:
        model = joblib.load('best_logistic_custom_model.joblib')
        return model
    except Exception as e:
        st.error(f"Lỗi khi load mô hình: {str(e)}")
        return None

# Các nhãn độc hại (6 nhãn)
LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# Ngưỡng tối ưu cho từng nhãn (được tìm từ validation set)
OPTIMAL_THRESHOLDS = [0.53, 0.73, 0.52, 0.84, 0.52, 0.69]

# Mô tả chi tiết cho từng nhãn
LABEL_DESCRIPTIONS = {
    'toxic': '☠️ Độc hại - Bình luận có nội dung tiêu cực, gây hại',
    'severe_toxic': '💀 Rất độc hại - Bình luận cực kỳ xúc phạm, nguy hiểm',
    'obscene': '🔞 Tục tĩu - Bình luận chứa nội dung khiêu dâm, tục tĩu',
    'threat': '⚠️ Đe dọa - Bình luận có tính chất đe dọa, khủng bố',
    'insult': '😠 Xúc phạm - Bình luận sỉ nhục, làm nhục người khác',
    'identity_hate': '🚫 Kỳ thị - Bình luận kỳ thị chủng tộc, tôn giáo, giới tính'
}

def predict_toxicity(text, model):
    """
    Dự đoán độ độc hại của bình luận
    
    Args:
        text: Văn bản đầu vào
        model: Mô hình đã được huấn luyện
    
    Returns:
        predictions: Dict chứa kết quả dự đoán
    """
    # Tiền xử lý văn bản
    cleaned_text = preprocessing_clean_text(text)
    
    # Dự đoán xác suất
    probas = model.predict_proba([cleaned_text])[0]
    
    # Tạo kết quả với ngưỡng tối ưu cho từng nhãn
    results = {}
    for i, label in enumerate(LABELS):
        optimal_threshold = OPTIMAL_THRESHOLDS[i]
        results[label] = {
            'probability': float(probas[i]),
            'threshold': optimal_threshold,
            'is_toxic': bool(probas[i] > optimal_threshold)
        }
    
    return results, cleaned_text

def main():
    # Header
    st.title("🛡️ Toxic Comment Classification System")
    st.markdown("### Hệ thống phân loại bình luận độc hại với 6 nhãn")
    st.markdown("---")
    
    # Load mô hình
    model = load_model()
    
    if model is None:
        st.error("Không thể tải mô hình. Vui lòng kiểm tra file 'best_logistic_custom_model.joblib'")
        return
    
    # Sidebar - Thông tin
    with st.sidebar:
        st.header("ℹThông tin mô hình")
        st.info("Mô hình: Custom Logistic Regression One-vs-Rest")
        
        st.markdown("---")
        st.markdown("### Danh sách nhãn & Ngưỡng tối ưu")
        st.markdown("*Mỗi nhãn có ngưỡng riêng được tối ưu từ tập validation*")
        for i, label in enumerate(LABELS):
            st.markdown(f"**{label}**: `{OPTIMAL_THRESHOLDS[i]:.2f}`")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Nhập bình luận")
        
        # Text input
        user_input = st.text_area(
            "Nhập văn bản cần phân loại:",
            height=200,
            placeholder="Ví dụ: This is a great comment!"
        )
        
        # Nút phân tích
        analyze_button = st.button("🔍 Phân tích bình luận", type="primary", use_container_width=True)
    
    with col2:
        st.subheader("Kết quả phân tích")
        
        if analyze_button and user_input:
            with st.spinner("Đang phân tích..."):
                try:
                    # Dự đoán
                    results, cleaned_text = predict_toxicity(user_input, model)
                    
                    # Hiển thị văn bản đã xử lý
                    with st.expander("🔧 Văn bản sau tiền xử lý"):
                        st.code(cleaned_text)
                    
                    # Kiểm tra có nhãn nào vượt ngưỡng không
                    toxic_labels = [label for label, data in results.items() if data['is_toxic']]
                    
                    if toxic_labels:
                        st.error(f"**Cảnh báo:** Bình luận này có dấu hiệu độc hại!")
                        st.markdown(f"**Phát hiện {len(toxic_labels)} nhãn độc hại:**")
                        for label in toxic_labels:
                            st.markdown(f"- {LABEL_DESCRIPTIONS[label]}")
                    else:
                        st.success("**Bình luận an toàn!** Không phát hiện dấu hiệu độc hại.")
                    
                    st.markdown("---")
                    st.markdown("### Chi tiết xác suất từng nhãn")
                    
                    # Tạo DataFrame để hiển thị
                    df_results = pd.DataFrame([
                        {
                            'Nhãn': LABEL_DESCRIPTIONS[label],
                            'Xác suất': f"{data['probability']:.2%}",
                            'Ngưỡng': f"{data['threshold']:.2f}",
                            'Trạng thái': '🔴 Độc hại' if data['is_toxic'] else '🟢 An toàn'
                        }
                        for label, data in results.items()
                    ])
                    
                    st.dataframe(df_results, use_container_width=True, hide_index=True)
                    
                    # Biểu đồ thanh
                    st.markdown("### Biểu đồ xác suất")
                    chart_data = pd.DataFrame({
                        'Nhãn': LABELS,
                        'Xác suất': [results[label]['probability'] for label in LABELS]
                    })
                    st.bar_chart(chart_data.set_index('Nhãn'))
                    
                except Exception as e:
                    st.error(f"Lỗi khi phân tích: {str(e)}")
        
        elif analyze_button and not user_input:
            st.warning("Vui lòng nhập văn bản cần phân tích!")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>🛡️ Toxic Comment Classification System | Phát triển bằng Streamlit</p>
            <p><small>Mô hình: Custom Logistic Regression với TF-IDF Vectorizer</small></p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
