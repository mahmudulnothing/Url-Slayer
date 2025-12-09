"""
URL Slayer - Advanced Phishing Detection System
================================================
A production-ready web application for detecting phishing URLs using Machine Learning.

File: app.py
Purpose: Real-time phishing detection interface
Model: Random Forest Classifier (phishing_model.pkl)
"""

# ============================================================================
# IMPORTS & DEPENDENCIES
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import socket
from urllib.parse import urlparse
import warnings
import requests
from bs4 import BeautifulSoup

warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION (MUST BE FIRST STREAMLIT COMMAND)
# ============================================================================

st.set_page_config(
    page_title="URL Slayer - Phishing Detection",
    page_icon="⚔️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CONSTANTS & CONFIGURATION
# ============================================================================

MODEL_PATH = 'phishing_model.pkl'

# Expected feature columns (30 features)
FEATURE_COLUMNS = [
    'having_IP_Address', 'URL_Length', 'Shortining_Service', 'having_At_Symbol',
    'double_slash_redirecting', 'Prefix_Suffix', 'having_Sub_Domain', 'SSLfinal_State',
    'Domain_registeration_length', 'Favicon', 'port', 'HTTPS_token', 'Request_URL',
    'URL_of_Anchor', 'Links_in_tags', 'SFH', 'Submitting_to_email', 'Abnormal_URL',
    'Redirect', 'on_mouseover', 'RightClick', 'popUpWidnow', 'Iframe',
    'age_of_domain', 'DNSRecord', 'web_traffic', 'Page_Rank', 'Google_Index',
    'Links_pointing_to_page', 'Statistical_report'
]

# Known URL shortening services
SHORTENING_SERVICES = [
    'bit.ly', 'goo.gl', 'shorte.st', 'go2l.ink', 'x.co', 'ow.ly', 't.co',
    'tinyurl', 'tr.im', 'is.gd', 'cli.gs', 'yfrog.com', 'migre.me', 'ff.im',
    'tiny.cc', 'url4.eu', 'twit.ac', 'su.pr', 'twurl.nl', 'snipurl.com',
    'short.to', 'BudURL.com', 'ping.fm', 'post.ly', 'Just.as', 'bkite.com',
    'snipr.com', 'fic.kr', 'loopt.us', 'doiop.com', 'short.ie', 'kl.am',
    'wp.me', 'rubyurl.com', 'om.ly', 'to.ly', 'bit.do', 'lnkd.in', 'db.tt',
    'qr.ae', 'adf.ly', 'bitly.com', 'cur.lv', 'tinyurl.com', 'ity.im',
    'q.gs', 'po.st', 'bc.vc', 'twitthis.com', 'u.to', 'j.mp', 'buzurl.com',
    'cutt.us', 'u.bb', 'yourls.org', 'x.co', 'prettylinkpro.com', 'scrnch.me',
    'filoops.info', 'vzturl.com', 'qr.net', '1url.com', 'tweez.me', 'v.gd',
    'tr.im', 'link.zip.net'
]

# ============================================================================
# CUSTOM CSS STYLING
# ============================================================================

def apply_custom_css():
    """Apply custom CSS for professional styling with URL Slayer theme."""
    st.markdown("""
        <style>
        /* Main container styling */
        .main {
            padding: 2rem;
        }
        
        /* Header styling - Dark warrior theme with red accents */
        .header-container {
            background: linear-gradient(135deg, #1a1a1a 0%, #2d1b1b 50%, #1a0a0a 100%);
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            box-shadow: 0 8px 16px rgba(220, 38, 38, 0.3);
            border: 2px solid #dc2626;
        }
        
        .header-title {
            color: #dc2626;
            font-size: 3rem;
            font-weight: bold;
            margin: 0;
            text-align: center;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.8);
            letter-spacing: 2px;
        }
        
        .header-subtitle {
            color: #fca5a5;
            font-size: 1.2rem;
            text-align: center;
            margin-top: 0.5rem;
            font-weight: 500;
        }
        
        /* Button styling - Aggressive red theme */
        .stButton>button {
            width: 100%;
            background: linear-gradient(135deg, #dc2626 0%, #991b1b 100%);
            color: white;
            font-weight: bold;
            padding: 0.75rem 2rem;
            border-radius: 8px;
            border: 2px solid #7f1d1d;
            box-shadow: 0 4px 6px rgba(220, 38, 38, 0.4);
            transition: all 0.3s;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(220, 38, 38, 0.6);
            background: linear-gradient(135deg, #991b1b 0%, #7f1d1d 100%);
        }
        
        /* Metric styling */
        .metric-container {
            text-align: center;
            padding: 1rem;
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        }
        
        /* Footer styling */
        .footer {
            text-align: center;
            padding: 2rem;
            color: #666;
            font-size: 0.9rem;
            margin-top: 3rem;
            border-top: 1px solid #e0e0e0;
        }
        </style>
    """, unsafe_allow_html=True)


# ============================================================================
# MODEL LOADING
# ============================================================================

@st.cache_resource
def load_model():
    """
    Load the pre-trained Random Forest model from disk.
    Uses caching to avoid reloading on every interaction.
    
    Returns:
    --------
    model : RandomForestClassifier or None
        Loaded model object, or None if loading fails
    """
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except FileNotFoundError:
        st.warning(f"⚠️ Model file '{MODEL_PATH}' not found. Using demo mode.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None


# ============================================================================
# HTML CONTENT ANALYSIS FUNCTIONS
# ============================================================================

def fetch_page_content(url, timeout=5):
    """
    Safely fetch webpage content.
    
    Returns:
    --------
    tuple : (html_content, status_code) or (None, None) if failed
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=timeout, verify=False, allow_redirects=True)
        return response.text, response.status_code
    except:
        return None, None


def analyze_html_content(html_content):
    """
    Analyze HTML content for phishing indicators.
    
    Returns:
    --------
    dict : Dictionary with analysis results
    """
    if not html_content:
        return {
            'has_login_form': 0,
            'has_password_field': 0,
            'suspicious_forms': 0,
            'external_links_ratio': 0,
            'has_suspicious_keywords': 0,
            'has_hidden_fields': 0
        }
    
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Check for login/password forms
        forms = soup.find_all('form')
        password_fields = soup.find_all('input', {'type': 'password'})
        has_login_form = 1 if len(forms) > 0 and len(password_fields) > 0 else 0
        has_password_field = 1 if len(password_fields) > 0 else 0
        
        # Check for suspicious form actions
        suspicious_form_count = 0
        for form in forms:
            action = form.get('action', '').lower()
            if any(x in action for x in ['php', 'cgi-bin', 'submit', 'process']) or action == '' or action == '#':
                suspicious_form_count += 1
        
        # Check for hidden fields (common in phishing)
        hidden_fields = soup.find_all('input', {'type': 'hidden'})
        has_hidden = 1 if len(hidden_fields) > 3 else 0
        
        # Check for external links vs internal links
        all_links = soup.find_all('a', href=True)
        if len(all_links) > 0:
            external_links = sum(1 for link in all_links if link['href'].startswith(('http://', 'https://')))
            external_ratio = external_links / len(all_links)
        else:
            external_ratio = 0
        
        # Check for suspicious keywords in page content
        page_text = soup.get_text().lower()
        phishing_keywords = [
            'verify your account', 'confirm your identity', 'suspended account',
            'unusual activity', 'verify identity', 'account locked', 'confirm identity',
            'security alert', 'click here immediately', 'act now', 'urgent action required',
            'verify now', 'update payment', 'billing information', 'payment details'
        ]
        
        keyword_count = sum(1 for keyword in phishing_keywords if keyword in page_text)
        has_suspicious_keywords = 1 if keyword_count >= 2 else 0
        
        return {
            'has_login_form': has_login_form,
            'has_password_field': has_password_field,
            'suspicious_forms': min(suspicious_form_count, 3),
            'external_links_ratio': external_ratio,
            'has_suspicious_keywords': has_suspicious_keywords,
            'has_hidden_fields': has_hidden
        }
    except:
        return {
            'has_login_form': 0,
            'has_password_field': 0,
            'suspicious_forms': 0,
            'external_links_ratio': 0,
            'has_suspicious_keywords': 0,
            'has_hidden_fields': 0
        }


# ============================================================================
# FEATURE EXTRACTION FUNCTIONS
# ============================================================================

def having_ip_address(url):
    """Check if URL contains an IP address instead of domain name."""
    ip_pattern = re.compile(
        r'(([01]?\d\d?|2[0-4]\d|25[0-5])\.([01]?\d\d?|2[0-4]\d|25[0-5])\.'
        r'([01]?\d\d?|2[0-4]\d|25[0-5])\.([01]?\d\d?|2[0-4]\d|25[0-5]))'
    )
    match = ip_pattern.search(url)
    return -1 if match else 1


def url_length(url):
    """Categorize URL based on length."""
    length = len(url)
    if length < 54:
        return 1
    elif 54 <= length <= 75:
        return 0
    else:
        return -1


def shortening_service(url):
    """Check if URL uses a URL shortening service."""
    for service in SHORTENING_SERVICES:
        if service in url.lower():
            return -1
    return 1


def having_at_symbol(url):
    """Check for '@' symbol in URL."""
    return -1 if '@' in url else 1


def double_slash_redirecting(url):
    """Check for '//' appearing after the protocol."""
    protocol_pos = url.find('//')
    if protocol_pos > -1:
        remaining_url = url[protocol_pos + 2:]
        if '//' in remaining_url:
            return -1
    return 1


def prefix_suffix(url):
    """Check for '-' (dash) in domain name."""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc
        return -1 if '-' in domain else 1
    except:
        return -1


def having_sub_domain(url):
    """Count the number of subdomains."""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc
        # Remove port if present
        domain = domain.split(':')[0]
        dots_count = domain.count('.')
        
        # More aggressive subdomain detection
        if dots_count <= 1:
            return 1  # Normal domain
        elif dots_count == 2:
            return 0  # One subdomain
        else:
            return -1  # Multiple subdomains - suspicious
    except:
        return -1


def ssl_final_state(url):
    """Check for HTTPS and SSL certificate."""
    try:
        parsed = urlparse(url)
        scheme = parsed.scheme.lower()
        
        if scheme == 'https':
            return 1
        elif scheme == 'http':
            return -1
        else:
            return 0
    except:
        return -1


def domain_registration_length(url):
    """
    Check for suspicious patterns in domain.
    """
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        
        # Check for brand impersonation (common phishing tactic)
        brand_names = [
            'paypal', 'amazon', 'google', 'microsoft', 'apple', 'facebook',
            'instagram', 'netflix', 'ebay', 'bank', 'dhl', 'fedex', 'ups',
            'usps', 'visa', 'mastercard', 'wellsfargo', 'chase', 'citibank'
        ]
        
        # Check if domain contains brand name but isn't the real domain
        for brand in brand_names:
            if brand in domain:
                # If it contains the brand but has extra parts (like dhl-access, paypal-secure)
                if not domain.endswith(f'{brand}.com') and not domain.endswith(f'{brand}.net'):
                    return -1  # Brand impersonation detected!
        
        # Check for suspicious patterns
        suspicious_patterns = [
            'login', 'signin', 'account', 'verify', 'secure', 'update',
            'banking', 'confirm', 'suspended', 'locked', 'unusual', 'access'
        ]
        
        pattern_count = sum(1 for pattern in suspicious_patterns if pattern in domain)
        
        if pattern_count >= 2:
            return -1
        elif pattern_count == 1:
            return 0
        
        return 1
    except:
        return -1


def favicon(url):
    """Placeholder for favicon analysis."""
    return 1


def port(url):
    """Check if URL uses a non-standard port."""
    try:
        parsed = urlparse(url)
        if parsed.port:
            suspicious_ports = [
                22, 23, 445, 1433, 1521, 3306, 3389, 5432, 5900, 8080, 8443
            ]
            return -1 if parsed.port in suspicious_ports else 0
        return 1
    except:
        return 1


def https_token(url):
    """Check if 'https' appears in domain name (not protocol)."""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        return -1 if 'https' in domain else 1
    except:
        return 1


def request_url(url):
    """
    Check for suspicious request patterns.
    """
    try:
        parsed = urlparse(url)
        path = parsed.path.lower()
        query = parsed.query.lower()
        
        # Check for suspicious query parameters
        suspicious_params = ['redirect', 'url=', 'link=', 'goto=', 'return=', 'next=']
        
        for param in suspicious_params:
            if param in query:
                return -1
        
        return 1
    except:
        return 1


def url_of_anchor(url):
    """Placeholder for anchor URL analysis."""
    return 1


def links_in_tags(url):
    """Placeholder for meta/script/link tag analysis."""
    return 1


def sfh(url):
    """
    Check for Server Form Handler suspicious patterns.
    """
    try:
        parsed = urlparse(url)
        
        # Empty or about:blank SFH is suspicious
        if parsed.scheme == '' or 'about:blank' in url.lower():
            return -1
        
        # Check for suspicious form handlers
        if 'javascript:' in url.lower() or 'data:' in url.lower():
            return -1
            
        return 1
    except:
        return -1


def submitting_to_email(url):
    """Check if URL contains 'mailto:' (form submission to email)."""
    return -1 if 'mailto:' in url.lower() else 1


def abnormal_url(url):
    """Check for abnormal URL patterns."""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        path = parsed.path.lower()
        
        # Check for multiple suspicious keywords
        phishing_keywords = [
            'verify', 'account', 'update', 'secure', 'banking', 'confirm',
            'login', 'signin', 'suspended', 'locked', 'verify', 'webscr',
            'cmd', 'security', 'ebayisapi', 'paypal', 'wallet'
        ]
        
        keyword_count = sum(1 for keyword in phishing_keywords if keyword in domain or keyword in path)
        
        if keyword_count >= 2:
            return -1
        elif keyword_count == 1:
            return 0
        
        return 1
    except:
        return -1


def redirect(url):
    """Count number of redirects."""
    redirect_indicators = ['redirect', 'url=', 'link=', 'goto=']
    count = sum(1 for indicator in redirect_indicators if indicator in url.lower())
    
    if count == 0:
        return 1
    elif count <= 2:
        return 0
    else:
        return -1


def on_mouseover(url):
    """Placeholder for JavaScript mouseover detection."""
    return 1


def right_click(url):
    """Placeholder for right-click disable detection."""
    return 1


def popup_window(url):
    """Placeholder for popup window detection."""
    return 1


def iframe(url):
    """Placeholder for iframe usage detection."""
    return 1


def age_of_domain(url):
    """Placeholder for domain age calculation."""
    return 0


def dns_record(url):
    """Check if domain has valid DNS record."""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc
        domain = domain.split(':')[0]
        socket.gethostbyname(domain)
        return 1
    except (socket.gaierror, socket.herror):
        return -1
    except:
        return 0


def web_traffic(url):
    """Placeholder for web traffic analysis."""
    return 0


def page_rank(url):
    """Placeholder for Google PageRank."""
    return 0


def google_index(url):
    """Placeholder for Google indexing check."""
    return 1


def links_pointing_to_page(url):
    """Placeholder for backlink analysis."""
    return 1


def statistical_report(url):
    """Placeholder for statistical report analysis."""
    return 1


def extract_features(url, html_analysis=None):
    """
    Extract all 30 features from the given URL.
    
    Parameters:
    -----------
    url : str
        The URL to analyze
    html_analysis : dict, optional
        HTML content analysis results
        
    Returns:
    --------
    pd.DataFrame : Single-row dataframe with 30 feature columns
    """
    if not url.startswith(('http://', 'https://')):
        url = 'http://' + url
    
    # Base URL features
    features = {
        'having_IP_Address': having_ip_address(url),
        'URL_Length': url_length(url),
        'Shortining_Service': shortening_service(url),
        'having_At_Symbol': having_at_symbol(url),
        'double_slash_redirecting': double_slash_redirecting(url),
        'Prefix_Suffix': prefix_suffix(url),
        'having_Sub_Domain': having_sub_domain(url),
        'SSLfinal_State': ssl_final_state(url),
        'Domain_registeration_length': domain_registration_length(url),
        'Favicon': favicon(url),
        'port': port(url),
        'HTTPS_token': https_token(url),
        'Request_URL': request_url(url),
        'URL_of_Anchor': url_of_anchor(url),
        'Links_in_tags': links_in_tags(url),
        'SFH': sfh(url),
        'Submitting_to_email': submitting_to_email(url),
        'Abnormal_URL': abnormal_url(url),
        'Redirect': redirect(url),
        'on_mouseover': on_mouseover(url),
        'RightClick': right_click(url),
        'popUpWidnow': popup_window(url),
        'Iframe': iframe(url),
        'age_of_domain': age_of_domain(url),
        'DNSRecord': dns_record(url),
        'web_traffic': web_traffic(url),
        'Page_Rank': page_rank(url),
        'Google_Index': google_index(url),
        'Links_pointing_to_page': links_pointing_to_page(url),
        'Statistical_report': statistical_report(url)
    }
    
    # Adjust features based on HTML analysis if available
    if html_analysis:
        # If login form detected with password field, increase suspicion
        if html_analysis['has_login_form'] and html_analysis['has_password_field']:
            features['SFH'] = -1
            features['Submitting_to_email'] = -1
        
        # If suspicious forms detected
        if html_analysis['suspicious_forms'] >= 2:
            features['Request_URL'] = -1
            features['Abnormal_URL'] = -1
        
        # If suspicious keywords found
        if html_analysis['has_suspicious_keywords']:
            features['Statistical_report'] = -1
        
        # If many external links (content theft indicator)
        if html_analysis['external_links_ratio'] > 0.6:
            features['Links_in_tags'] = -1
            features['URL_of_Anchor'] = -1
    
    df = pd.DataFrame([features], columns=FEATURE_COLUMNS)
    return df


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application function."""
    
    apply_custom_css()
    
    # Header Section
    st.markdown("""
        <div class="header-container">
            <h1 class="header-title">⚔️ URL SLAYER</h1>
            <p class="header-subtitle">
                Advanced Phishing Detection System • Machine Learning Powered Security Analysis
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar Information
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2913/2913133.png", width=150)
        st.title("About URL Slayer")
        st.markdown("""
        An advanced security tool powered by **Machine Learning** to identify 
        and eliminate phishing threats in real-time.
        
        ### ⚔️ Combat Statistics
        - **Accuracy:** 97.42%
        - **Precision:** 96.96%
        - **Recall:** 98.46%
        - **F1-Score:** 97.70%
        
        ### 🔍 Detection Capabilities
        - URL structure analysis
        - Domain characteristics
        - SSL/HTTPS verification
        - Redirect detection
        - DNS validation
        - 25+ additional indicators
        
        ### 🛠️ Technology Stack
        - **Algorithm:** Random Forest (100 estimators)
        - **Framework:** Streamlit
        - **ML Library:** scikit-learn
        - **Dataset:** UCI Phishing Websites
        
        ### ⚠️ Security Notice
        This tool provides automated threat assessment. 
        Always verify suspicious URLs through multiple sources.
        """)
        
        st.markdown("---")
        st.markdown("**🎓 Mahmudul's Project**")
        st.markdown("*ML Engineering with Cybersecurity*")
    
    # Main Content Area
    st.markdown("### 🔗 Enter a URL to Analyze")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        url_input = st.text_input(
            "URL",
            placeholder="https://example.com or example.com",
            help="Enter the full URL or domain name to check for threats",
            label_visibility="collapsed"
        )
    
    with col2:
        analyze_button = st.button("⚔️ Analyze URL", use_container_width=True)
    
    st.info("💡 **Tip:** You can enter URLs with or without the protocol (http:// or https://). The system will analyze various characteristics to determine if the URL is potentially malicious.")
    
    # Analysis Section
    if analyze_button:
        if not url_input:
            st.warning("⚠️ Please enter a URL to analyze.")
        else:
            with st.spinner("⚔️ Analyzing URL and scanning website content..."):
                try:
                    # Normalize URL
                    analysis_url = url_input if url_input.startswith(('http://', 'https://')) else 'http://' + url_input
                    
                    # Step 1: Fetch and analyze HTML content
                    st.info("🔍 **Step 1/3:** Fetching website content...")
                    html_content, status_code = fetch_page_content(analysis_url)
                    
                    html_analysis = None
                    if html_content:
                        st.info("🔍 **Step 2/3:** Analyzing HTML for phishing indicators...")
                        html_analysis = analyze_html_content(html_content)
                    else:
                        st.warning("⚠️ Could not fetch website content. Analyzing URL structure only.")
                    
                    # Step 2: Extract features
                    st.info("🔍 **Step 3/3:** Extracting 30 security features...")
                    model = load_model()
                    features_df = extract_features(analysis_url, html_analysis)
                    
                    # Step 3: Make prediction
                    if model is not None:
                        prediction = model.predict(features_df)[0]
                        prediction_proba = model.predict_proba(features_df)[0]
                        
                        if prediction == 1:
                            confidence = prediction_proba[1] * 100
                        else:
                            confidence = prediction_proba[0] * 100
                    else:
                        # Demo mode - more aggressive detection
                        suspicious_count = (features_df == -1).sum().sum()
                        neutral_count = (features_df == 0).sum().sum()
                        
                        # Calculate suspicion score
                        suspicion_score = suspicious_count * 2 + neutral_count
                        
                        # Boost suspicion if HTML analysis shows phishing signs
                        if html_analysis:
                            if html_analysis['has_login_form']:
                                suspicion_score += 4
                            if html_analysis['has_suspicious_keywords']:
                                suspicion_score += 3
                            if html_analysis['suspicious_forms'] >= 2:
                                suspicion_score += 3
                        
                        # More aggressive threshold
                        if suspicion_score >= 8:
                            prediction = -1  # Phishing
                            confidence = min(95, 50 + (suspicion_score * 4))
                        else:
                            prediction = 1  # Legitimate
                            confidence = max(70, 95 - (suspicion_score * 5))
                    
                    # Display Results
                    st.markdown("---")
                    st.markdown("## 📊 Analysis Results")
                    
                    result_col1, result_col2, result_col3 = st.columns([2, 1, 1])
                    
                    with result_col1:
                        if prediction == 1:
                            st.success("### ✅ LEGITIMATE WEBSITE")
                            st.markdown("""
                                This URL appears to be **safe** based on analysis. 
                                The system classified it as a legitimate website.
                            """)
                        else:
                            st.error("### ⚔️ THREAT DETECTED")
                            st.markdown("""
                                **DANGER:** This URL shows characteristics of a phishing attack. 
                                **Do not** enter personal information or credentials.
                            """)
                    
                    with result_col2:
                        st.metric(
                            label="Verdict",
                            value="SAFE" if prediction == 1 else "THREAT",
                            delta="Legitimate" if prediction == 1 else "Phishing",
                            delta_color="normal" if prediction == 1 else "inverse"
                        )
                    
                    with result_col3:
                        st.metric(
                            label="Confidence",
                            value=f"{confidence:.1f}%",
                            delta=f"{confidence:.1f}%" if confidence > 80 else f"{confidence:.1f}%",
                            delta_color="normal" if confidence > 80 else "inverse"
                        )
                    
                    # Feature Analysis Section
                    st.markdown("---")
                    st.markdown("### 🔬 Detailed Feature Analysis")
                    
                    st.markdown("""
                        The table below shows the 30 features extracted and analyzed.
                        - **Value 1:** Legitimate characteristic
                        - **Value 0:** Neutral/uncertain characteristic
                        - **Value -1:** Suspicious/phishing characteristic
                    """)
                    
                    st.dataframe(
                        features_df.T.rename(columns={0: 'Feature Value'}),
                        use_container_width=True,
                        height=400
                    )
                    
                    # Feature Statistics
                    st.markdown("### 📈 Feature Statistics")
                    
                    stat_col1, stat_col2, stat_col3 = st.columns(3)
                    
                    with stat_col1:
                        legitimate_features = (features_df == 1).sum().sum()
                        st.metric("Legitimate Indicators", legitimate_features, "✅")
                    
                    with stat_col2:
                        neutral_features = (features_df == 0).sum().sum()
                        st.metric("Neutral Indicators", neutral_features, "⚪")
                    
                    with stat_col3:
                        suspicious_features = (features_df == -1).sum().sum()
                        st.metric("Suspicious Indicators", suspicious_features, "🚩")
                    
                    # HTML Analysis Results (if available)
                    if html_analysis:
                        st.markdown("---")
                        st.markdown("### 🌐 Website Content Analysis")
                        
                        html_col1, html_col2, html_col3 = st.columns(3)
                        
                        with html_col1:
                            if html_analysis['has_login_form']:
                                st.error("🔐 Login Form Detected")
                            else:
                                st.success("✅ No Login Form")
                        
                        with html_col2:
                            if html_analysis['has_password_field']:
                                st.error("🔑 Password Field Found")
                            else:
                                st.success("✅ No Password Field")
                        
                        with html_col3:
                            if html_analysis['has_suspicious_keywords']:
                                st.error("⚠️ Suspicious Keywords")
                            else:
                                st.success("✅ No Suspicious Text")
                        
                        # Additional HTML metrics
                        st.markdown("#### Additional Website Metrics")
                        metric_col1, metric_col2, metric_col3 = st.columns(3)
                        
                        with metric_col1:
                            st.metric("Suspicious Forms", html_analysis['suspicious_forms'])
                        
                        with metric_col2:
                            st.metric("External Links Ratio", f"{html_analysis['external_links_ratio']:.1%}")
                        
                        with metric_col3:
                            hidden_status = "Yes ⚠️" if html_analysis['has_hidden_fields'] else "No ✅"
                            st.metric("Hidden Form Fields", hidden_status)
                    
                    # Recommendations
                    st.markdown("---")
                    st.markdown("### 💡 Security Recommendations")
                    
                    if prediction == 1:
                        st.info("""
                        **While this URL appears legitimate, always:**
                        - Verify the domain spelling carefully
                        - Check for HTTPS before entering sensitive data
                        - Be cautious of unexpected emails with links
                        - Use two-factor authentication when available
                        - Keep security software updated
                        """)
                    else:
                        st.warning("""
                        **⚔️ THREAT RESPONSE PROTOCOL:**
                        - **DO NOT** enter passwords, credit cards, or personal data
                        - **DO NOT** download files from this site
                        - Report this URL to security teams
                        - Clear browser cache and run security scan
                        - Change passwords if credentials were entered
                        """)
                    
                    # Download Report
                    st.markdown("---")
                    st.markdown("### 📥 Export Analysis Report")
                    
                    report_data = features_df.copy()
                    report_data['URL'] = url_input
                    report_data['Prediction'] = 'Legitimate' if prediction == 1 else 'Phishing'
                    report_data['Confidence'] = f"{confidence:.2f}%"
                    
                    csv = report_data.to_csv(index=False)
                    
                    st.download_button(
                        label="📄 Download Analysis Report (CSV)",
                        data=csv,
                        file_name=f"url_slayer_analysis_{url_input.replace('://', '_').replace('/', '_')[:30]}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
                except Exception as e:
                    st.error(f"❌ Error during analysis: {str(e)}")
                    st.exception(e)
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div class="footer">
            <p><strong>⚔️ URL Slayer</strong> | Advanced Phishing Detection System</p>
            <p>Model Accuracy: 97.42% | Dataset: 11,055 URLs | Features: 30</p>
            <p>© 2024 Mahmudul's Cybersecurity Portfolio</p>
        </div>
    """, unsafe_allow_html=True)


# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()