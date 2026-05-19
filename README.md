# MediAI Clinical Assistant

MediAI Clinical Assistant is an intelligent healthcare decision-support system designed to streamline patient treatment recommendations. By leveraging advanced machine learning models, the application analyzes critical patient metrics—including BMI, Blood Pressure, Cholesterol, Sugar Levels, and Sodium-Potassium ratios—to provide data-driven insights and suggest appropriate pharmaceutical treatments.

Built with a modern, dark-themed UI using Streamlit, this tool aims to assist healthcare professionals in making rapid, informed clinical decisions while ensuring detailed documentation through automated report generation.
---

## 🔑 Key Features
* **AI-Driven Diagnostics:** Utilizes pre-trained Scikit-Learn models to predict optimal medication based on patient vitals.
* **Streamlined UI:** A clean, intuitive dashboard built with Streamlit for seamless user experience.
* **Automated Reporting:** Generates downloadable clinical reports (PDF/TXT) for patient records.
* **Data Privacy:** Localized processing of clinical metrics.
* **Comprehensive Metrics:** Analyzes a wide range of factors, including Sodium/Potassium levels, BMI, and metabolic indicators.

---

## 🛠 Tech Stack
* **Language:** Python 3.x
* **Framework:** Streamlit
* **ML Library:** Scikit-Learn
* **Data Handling:** Pandas, NumPy
* **Report Generation:** FPDF / Custom text formatting

---

## ⚙️ Installation & Usage

### 1. Clone the repository
```bash
git clone [https://github.com/kgouthamkrishnan/MediAI-Clinical-Assistant.git]

python -m venv venv
# On Windows:
venv\Scripts\activate

pip install -r requirements.txt

streamlit run app.py

📋 Project Workflow
The following diagram illustrates the data flow from patient input to the final clinical recommendation:

⚠️ Disclaimer
Important: MediAI Clinical Assistant is intended for research and educational purposes only. It is designed to assist healthcare professionals and should not replace professional medical judgment, diagnosis, or treatment. Always consult with a qualified medical professional for health-related decisions.

👤 Author
Goutham Krishnan | GitHub Profile


