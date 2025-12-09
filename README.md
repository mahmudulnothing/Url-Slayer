# ⚔️ URL Slayer – Advanced Phishing Detection System

**URL Slayer** is a production-ready, machine learning–powered web application designed to detect phishing URLs through comprehensive security feature analysis and real-time website inspection.

The system combines **URL structure analysis**, **domain intelligence**, and **HTML content inspection** with a trained **Random Forest classifier** to accurately identify malicious phishing websites.

---

## 🚨 Problem Statement

Phishing attacks remain one of the most common and effective cyber threats, exploiting user trust through deceptive URLs and fake login pages. Manual identification is unreliable and error-prone.

**URL Slayer** addresses this problem by:
- Automatically analyzing URLs for phishing indicators
- Performing real-time security feature extraction
- Providing explainable results with confidence scores
- Delivering a secure and interactive web interface

---

## ✅ Key Features

- 🔍 **30-feature URL security analysis**
- 🌐 **Live HTML content inspection** (forms, keywords, links)
- ⚔️ **Machine Learning–based phishing classification**
- 📊 **Confidence score & detailed feature breakdown**
- 📥 **Downloadable CSV security report**
- 🧠 **Explainable indicators (legitimate / neutral / suspicious)**
- 💻 **Streamlit-powered web interface**
- 🚀 **Production-ready & cloud deployable**

---

## 🧠 Machine Learning Model

- **Algorithm:** Random Forest Classifier  
- **Estimators:** 100  
- **Dataset:** UCI Phishing Websites Dataset  
- **Total Samples:** 11,055 URLs  
- **Features:** 30 handcrafted security features  

### 📊 Model Performance
| Metric     | Score |
|-----------|-------|
| Accuracy  | 97.42% |
| Precision | 96.96% |
| Recall    | 98.46% |
| F1-Score  | 97.70% |

---

## 🔬 Feature Engineering Overview

URL Slayer extracts and evaluates **30 security indicators**, including:

- IP-based URLs
- URL length & structure
- URL shortening services
- SSL / HTTPS validation
- Subdomain complexity
- Brand impersonation patterns
- Suspicious query parameters
- Redirect behavior
- DNS resolution checks
- Domain abnormalities

In addition, **live HTML analysis** inspects:
- Login forms & password fields
- Suspicious form handlers
- Hidden input fields
- External link ratios
- Common phishing keywords

---

## 🛠️ Technology Stack

- **Programming Language:** Python
- **Machine Learning:** scikit-learn
- **Web Framework:** Streamlit
- **Data Handling:** Pandas, NumPy
- **Model Persistence:** joblib
- **Web Analysis:** Requests, BeautifulSoup

---

## 🧪 How It Works (Pipeline)

1. User submits a URL
2. URL structure is normalized and parsed
3. Website HTML (if accessible) is fetched and analyzed
4. 30 security features are extracted
5. Trained ML model predicts legitimacy
6. Results, confidence, and indicators are displayed
7. Optional CSV report can be downloaded

---

## ▶️ Running Locally

### 1. Clone the repository
```bash
git clone https://github.com/mahmudulnothing/Url-Slayer.git
cd url-slayer
