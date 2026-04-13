import streamlit as st
from PIL import Image
import fitz  # PyMuPDF
import io
import base64
import requests
 
# Page config
st.set_page_config(
    page_title="MediScan AI",
    page_icon="🏥",
    layout="wide"
)
 
# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        color: white;
        margin-bottom: 30px;
    }
    .result-box {
        background: #f8f9fa;
        border-left: 5px solid #667eea;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .success-box {
        background: #d4edda;
        border-left: 5px solid #28a745;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)
 
# Header
st.markdown("""
<div class="main-header">
    <h1>🏥 MediScan AI</h1>
    <p>AI-Powered Medical Report Assistant | Engineering a Healthier Tomorrow</p>
</div>
""", unsafe_allow_html=True)
 
def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")
 
def analyze_with_huggingface(api_token, image, scan_type, symptoms, age, gender, language):
    img_base64 = image_to_base64(image)
    
    lang_instruction = {
        "Hinglish (Hindi + English)": "Reply in Hinglish (mix of Hindi and English in English script)",
        "English": "Reply in English",
        "Hindi": "Reply in Hindi"
    }[language]
 
    prompt = f"""You are an expert medical AI assistant. Analyze this medical report/scan image carefully.
 
Patient Info: Age {age}, Gender {gender}
Report Type: {scan_type}
Symptoms: {symptoms if symptoms else 'Not provided'}
 
{lang_instruction}
 
Please provide:
1. REPORT SUMMARY - Simple explanation for common person
2. KEY FINDINGS - Main observations (normal and abnormal)
3. ABNORMALITIES - Mark as HIGH CONCERN, MODERATE, or NORMAL
4. POSSIBLE CONDITIONS - What these findings might indicate
5. PREVENTIVE GUIDANCE - Lifestyle changes and precautions
6. NEXT STEPS - Which specialist to consult
7. DISCLAIMER - This is AI analysis, always consult a doctor"""
 
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json"
    }
 
    payload = {
        "model": "meta-llama/Llama-4-Scout-17B-16E-Instruct:groq",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_base64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ],
        "max_tokens": 1500
    }
 
    response = requests.post(
        "https://router.huggingface.co/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=120
    )
 
    if response.status_code == 200:
        result = response.json()
        return result["choices"][0]["message"]["content"]
    else:
        raise Exception(f"API Error {response.status_code}: {response.text}")
 
# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    api_token=("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")
    #api_token = st.text_input("🔑 Hugging Face API Token", type="password", placeholder="hf_...")
    
    st.divider()
    
    scan_type = st.selectbox("🔬 Report Type", [
        "🦴 X-Ray (Bone)",
        "🫁 X-Ray (Chest)",
        "🧠 MRI / CT Scan",
        "🩸 Blood Report",
        "🦷 Dental X-Ray",
        "📋 General Medical Report"
    ])
    
    language = st.selectbox("🌐 Language", [
        "Hinglish (Hindi + English)",
        "English",
        "Hindi"
    ])
    
    st.divider()
    st.info("⚕️ Always consult a doctor for final diagnosis.")
 
# Main content
col1, col2 = st.columns([1, 1])
 
with col1:
    st.header("📤 Upload Report")
    
    uploaded_file = st.file_uploader(
        "Upload your medical report",
        type=["jpg", "jpeg", "png", "pdf"],
        help="Supported: X-Ray, MRI, CT Scan, Blood Report"
    )
    
    symptoms = st.text_area(
        "💬 Symptoms (optional)",
        placeholder="e.g., chest pain, fever, knee pain...",
        height=100
    )
    
    patient_age = st.number_input("👤 Patient Age", min_value=1, max_value=120, value=25)
    patient_gender = st.selectbox("👥 Gender", ["Male", "Female", "Other"])
    
    analyze_btn = st.button("🔍 Analyze Report", type="primary", use_container_width=True)
 
with col2:
    st.header("📊 Analysis Results")
    
    if analyze_btn:
        if not api_token:
            st.error("⚠️ Please enter your Hugging Face API Token in the sidebar!")
        elif not uploaded_file:
            st.error("⚠️ Please upload a medical report!")
        else:
            with st.spinner("🤖 AI analyzing your report... Please wait (30-60 seconds)..."):
                try:
                    if uploaded_file.type == "application/pdf":
                        pdf_bytes = uploaded_file.read()
                        pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                        page = pdf_doc[0]
                        mat = fitz.Matrix(2, 2)
                        pix = page.get_pixmap(matrix=mat)
                        img_bytes = pix.tobytes("png")
                        image = Image.open(io.BytesIO(img_bytes))
                    else:
                        image = Image.open(uploaded_file)
                    
                    st.image(image, caption="Uploaded Report width=600")
                    
                    result = analyze_with_huggingface(
                        api_token, image, scan_type,
                        symptoms, patient_age, patient_gender, language
                    )
                    
                    st.markdown("---")
                    st.markdown(result)
                    
                    st.markdown("""
                    <div class="success-box">
                    ✅ Analysis complete! Please consult a qualified doctor for professional medical advice.
                    </div>
                    """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.info("💡 Check your API token and try again.")
    else:
        st.markdown("""
        <div class="result-box">
        <h4>👈 How to use MediScan AI:</h4>
        <ol>
            <li>Enter Hugging Face API Token in sidebar</li>
            <li>Select report type</li>
            <li>Upload medical report (image or PDF)</li>
            <li>Add symptoms if any</li>
            <li>Click Analyze Report</li>
        </ol>
        <p>⚕️ Supports: X-Ray, MRI, CT Scan, Blood Reports</p>
        </div>
        """, unsafe_allow_html=True)
 
# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; padding: 10px;'>
    🏥 MediScan AI | Engineering a Healthier Tomorrow | Hackathon Project<br>
    ⚠️ For educational purposes only. Always consult a qualified healthcare professional.
</div>
""", unsafe_allow_html=True)
 
