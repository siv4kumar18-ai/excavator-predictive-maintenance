import streamlit as st
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, fftconvolve
from scipy.fft import rfft, rfftfreq
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

# ==========================================
# PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="Predictive Maintenance & Diagnostic Simulation",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .metric-card {
        background-color: #1e2129;
        border-radius: 10px;
        padding: 18px;
        border: 1px solid #2e3340;
    }
    .status-normal {
        background-color: rgba(40, 167, 69, 0.15);
        border-left: 6px solid #28a745;
        padding: 15px;
        border-radius: 6px;
        color: #28a745;
    }
    .status-warning {
        background-color: rgba(255, 193, 7, 0.15);
        border-left: 6px solid #ffc107;
        padding: 15px;
        border-radius: 6px;
        color: #d39e00;
    }
    .status-critical {
        background-color: rgba(220, 53, 69, 0.15);
        border-left: 6px solid #dc3545;
        padding: 15px;
        border-radius: 6px;
        color: #dc3545;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 0. MULTI-CATEGORY MACHINERY DATABASE
# ==========================================
EQUIPMENT_DATABASE = {
    "Heavy Duty Machinery (Excavators)": {
        "Heavy Duty Excavators": [
            "CAT 336D — Main Hydraulic Pump & Drive",
            "Komatsu PC400 — Swing Motor & Planetary Gearbox",
            "Hitachi EX1200 — Final Drive & Track Roller Bearing",
            "Volvo EC480D — Diesel Engine & Turbocharger Assembly"
        ]
    },
    "Cranes & Lifts": {
        "Mobile Cranes": [
            "Liebherr LTM 1100 — Telescopic Boom Hoist Winch & Brake",
            "Grove GMK5250L — Slewing Ring Bearing & Pinion Drive",
            "Tadano ATF 220G — Outrigger & Main Hydraulic Cylinder"
        ],
        "Tower Cranes": [
            "Potain MDT 219 — Hoist Gearbox & Drum Bearing",
            "Liebherr 280 EC-H — Trolley Travel Motor & Slewing Drive",
            "Terex CTL 272 — High-Torque Winch & Reduction Gear"
        ],
        "Industrial Lifts": [
            "Konecranes 50T Overhead — Bridge Hoist Motor & Rope Drum",
            "JLG Heavy Scissor Lift — Hydraulic Lift Pump & Actuator",
            "Hyster Forklift H360 — Mast Lift Cylinder & Drive Axle"
        ]
    },
    "Hydraulic Machines": {
        "Hydraulic Presses": [
            "Schuler 2500T Forging Press — Main Ram Cylinder & Seals",
            "Beckwood Stamping Press — High-Pressure Proportional Valve",
            "industrial Deep-Draw Press — Variable Displacement Piston Pump"
        ],
        "Industrial Pumps": [
            "Bosch Rexroth Axial Piston Pump — Swashplate & Port Plate",
            "Parker Denison Vane Pump — Rotor Cartridge & Shaft Bearing",
            "Sulzer High-Pressure Centrifugal — Impeller & Mechanical Seal"
        ]
    },
    "Motors & Generators": {
        "Heavy Duty Motors": [
            "ABB 500kW Induction Motor — Drive-End (DE) Roller Bearing",
            "Siemens Simotics HV Motor — Stator Winding & Rotor Shaft",
            "WEG W22 Industrial Crusher Motor — Non-Drive-End (NDE) Bearing"
        ],
        "Air Compressors": [
            "Atlas Copco GA90 Rotary Screw — Male/Female Airend Rotors",
            "Ingersoll Rand Two-Stage Compressor — High-Speed Thrust Bearing",
            "Kaeser CSD 125 — Cooling Oil Separator & Drive Coupling"
        ],
        "Power Generators": [
            "Cummins C1000D5 (1000 kVA) — Alternator Rotor & Exciter Bearing",
            "Caterpillar C32 Diesel Genset — Crankshaft Damper & Turbo",
            "MTU Onsite Energy 16V — Main Alternator Coupling & Cooling Fan"
        ]
    }
}

# ==========================================
# 1. BASELINE ML ENGINE TRAINING (Cached)
# ==========================================
@st.cache_resource
def initialize_ai_models():
    """
    Trains the Random Forest and SVM models based on multi-machinery synthetic
    condition monitoring data (vibration, temperature, working_hours,
    noise_level, fluid_pressure, dominant_freq).
    """
    np.random.seed(42)
    n = 1800

    df = pd.DataFrame({
        "vibration_rms": np.random.uniform(1.0, 12.0, n),
        "vibration_p2p": np.random.uniform(3.0, 35.0, n),
        "dominant_freq": np.random.uniform(10.0, 1200.0, n),
        "temperature": np.random.uniform(40.0, 130.0, n),
        "fluid_pressure": np.random.uniform(1500.0, 5000.0, n),
        "acoustic_db": np.random.uniform(40.0, 115.0, n),
        "working_hours": np.random.uniform(1.0, 16.0, n)
    })

    # Multi-class target assignment based on domain diagnostic rules:
    # 0: Normal Operation
    # 1: Bearing Wear / Gear Meshing Fault (High Freq + High Acoustic)
    # 2: Structural Looseness / Misalignment / Unbalance (Low Freq + High Amp + Normal Temp)
    # 3: Lubrication Failure / Imminent Component Damage (High Temp + Abnormal Pressure)
    conditions = [
        (df["dominant_freq"] > 450) & (df["acoustic_db"] > 85),
        (df["dominant_freq"] < 180) & ((df["vibration_p2p"] > 16) | (df["vibration_rms"] > 7.0)) & (df["temperature"] <= 90),
        (df["temperature"] > 90) & ((df["fluid_pressure"] < 2200) | (df["fluid_pressure"] > 4200)),
        (df["vibration_rms"] > 7.0) | (df["temperature"] > 95) | (df["acoustic_db"] > 88)
    ]
    choices = [1, 2, 3, 1]
    df["fault_class"] = np.select(conditions, choices, default=0)

    features = [
        "vibration_rms", "vibration_p2p", "dominant_freq",
        "temperature", "fluid_pressure", "acoustic_db", "working_hours"
    ]
    X = df[features]
    y = df["fault_class"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Random Forest Model (Ensemble Tree Diagnostics)
    rf_model = RandomForestClassifier(n_estimators=120, max_depth=10, random_state=42)
    rf_model.fit(X_scaled, y)

    # Support Vector Machine (Hyperplane Anomaly Boundary)
    svm_model = SVC(kernel="rbf", probability=True, C=1.5, random_state=42)
    svm_model.fit(X_scaled, y)

    return rf_model, svm_model, scaler, features

rf_model, svm_model, scaler, feature_cols = initialize_ai_models()

# ==========================================
# 2. SIGNAL GENERATION & PREPROCESSING
# ==========================================
def generate_and_filter_signals(p2p_amp, rms_val, dom_freq, acoustic_db, filter_method, cutoff_freq, noise_injection):
    """
    Simulates a high-frequency accelerometer time-series waveform with mechanical
    harmonics and random background Gaussian noise, then applies digital denoising.
    """
    fs = 3000.0  # Sampling frequency (3000 Hz)
    duration = 0.5  # 0.5 seconds window
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)

    # Base primary harmonic + secondary gear/bearing sidebands
    primary_wave = (p2p_amp / 2.0) * np.sin(2 * np.pi * dom_freq * t)
    second_harmonic = (rms_val * 0.35) * np.sin(2 * np.pi * (dom_freq * 2.0) * t + np.pi / 4)
    sub_harmonic = (rms_val * 0.20) * np.sin(2 * np.pi * (dom_freq * 0.5) * t)

    # High-frequency bearing micro-impacts scaled by Acoustic Emission (dB)
    impact_modulation = 0.0
    if acoustic_db > 80:
        impact_severity = (acoustic_db - 80) / 30.0
        impact_modulation = impact_severity * 3.5 * np.sin(2 * np.pi * 850 * t) * np.exp(-15 * (t % 0.05))

    clean_signal = primary_wave + second_harmonic + sub_harmonic + impact_modulation

    # Inject ambient machine & sensor noise
    np.random.seed(101)
    ambient_noise = np.random.normal(0, noise_injection, len(t))
    high_freq_interference = (noise_injection * 0.6) * np.sin(2 * np.pi * 1350 * t)
    raw_signal = clean_signal + ambient_noise + high_freq_interference

    # Apply selected Preprocessing / Denoising Filter
    if filter_method == "Butterworth Low-Pass Filter":
        nyquist = 0.5 * fs
        normal_cutoff = min(cutoff_freq / nyquist, 0.99)
        b, a = butter(N=4, Wn=normal_cutoff, btype='low', analog=False)
        filtered_signal = filtfilt(b, a, raw_signal)
    elif filter_method == "Wavelet Soft-Thresholding (Simulated)":
        fft_coeffs = rfft(raw_signal)
        freqs = rfftfreq(len(t), 1 / fs)
        threshold = np.std(raw_signal) * 1.8 * np.sqrt(2 * np.log(len(t)))
        fft_filtered = np.where(
            (np.abs(fft_coeffs) > threshold) | (np.abs(freqs - dom_freq) < 30),
            fft_coeffs * (1 - 0.15 * (threshold / (np.abs(fft_coeffs) + 1e-6))),
            0.0
        )
        filtered_signal = np.fft.irfft(fft_filtered, n=len(t))
    else:  # Moving Average Kernel
        kernel_size = 7
        kernel = np.ones(kernel_size) / kernel_size
        filtered_signal = fftconvolve(raw_signal, kernel, mode='same')

    # Compute Fast Fourier Transform (FFT) Spectrum
    N = len(t)
    yf_raw = 2.0 / N * np.abs(rfft(raw_signal))
    yf_clean = 2.0 / N * np.abs(rfft(filtered_signal))
    xf = rfftfreq(N, 1 / fs)

    # Compute Signal-to-Noise Ratio (SNR) improvement
    estimated_noise = raw_signal - filtered_signal
    snr_db = 10 * np.log10(np.var(filtered_signal) / (np.var(estimated_noise) + 1e-8))
    denoised_rms = np.sqrt(np.mean(filtered_signal**2))

    return t, raw_signal, filtered_signal, xf, yf_raw, yf_clean, snr_db, denoised_rms

# ==========================================
# 3. HYBRID AI DIAGNOSTIC ENGINE
# ==========================================
def run_hybrid_diagnostics(
    machine_type, equipment_sub, p2p_amp, rms_val, dom_freq, temp, pressure, acoustic_db, working_hrs, denoised_rms
):
    """
    Combines rule-based physics thresholds with 1D-CNN spectral feature extraction,
    Random Forest classification, and SVM boundary scoring to output fault diagnosis,
    Anomaly Score (0-100%), and Remaining Useful Life (RUL).
    """
    input_df = pd.DataFrame([{
        "vibration_rms": rms_val,
        "vibration_p2p": p2p_amp,
        "dominant_freq": dom_freq,
        "temperature": temp,
        "fluid_pressure": pressure,
        "acoustic_db": acoustic_db,
        "working_hours": working_hrs
    }])
    X_in = scaler.transform(input_df[feature_cols])

    rf_probs = rf_model.predict_proba(X_in)[0]
    svm_probs = svm_model.predict_proba(X_in)[0]

    # Simulate 1D-CNN Spectral Feature Map Confidence
    cnn_probs = np.array([0.70, 0.10, 0.10, 0.10])
    if dom_freq >= 450 and acoustic_db >= 82:
        cnn_probs = np.array([0.05, 0.82, 0.08, 0.05])
    elif dom_freq <= 180 and (p2p_amp >= 15.0 or rms_val >= 6.5):
        cnn_probs = np.array([0.08, 0.07, 0.78, 0.07])
    elif temp > 88 and (pressure < 2300 or pressure > 4100):
        cnn_probs = np.array([0.05, 0.08, 0.07, 0.80])

    # Ensemble probability fusion (40% CNN, 35% Random Forest, 25% SVM)
    ensemble_probs = (0.40 * cnn_probs) + (0.35 * rf_probs) + (0.25 * svm_probs)
    ensemble_probs = ensemble_probs / np.sum(ensemble_probs)

    # Evaluate Explicit Physics & Domain Diagnostic Rules
    fault_triggers = []
    recommendations = []
    severity = "Normal"

    # Rule A: High-Frequency Vibration + High Acoustic Emission
    is_bearing_gear_fault = (dom_freq >= 400 and acoustic_db >= 82)
    # Rule B: Low-Frequency High Amplitude Vibration + Normal Temp
    is_structural_fault = (dom_freq < 220 and (p2p_amp >= 14.0 or rms_val >= 6.5) and temp <= 90)
    # Rule C: High Temperature + Abnormal Fluid/System Pressure
    is_lubrication_fault = (temp > 90 and (pressure < 2200 or pressure > 4200))

    if is_bearing_gear_fault:
        fault_triggers.append("Outer/Inner Race Bearing Wear or Gear Meshing Fault")
        recommendations.append(
            f"Schedule immediate bearing raceway inspection & gear/rotor clearance check on <strong>{machine_type} ({equipment_sub})</strong>. "
            "Perform ultrasonic grease replenishment and inspect magnetic drain plugs for metallic debris."
        )
        ensemble_probs[1] = max(ensemble_probs[1], 0.78)

    if is_structural_fault:
        fault_triggers.append("Structural Looseness, Misalignment, or Unbalance")
        recommendations.append(
            f"Check mounting bolts, drive coupling alignment, and structural base frame on <strong>{machine_type} ({equipment_sub})</strong>. "
            "Re-torque foundation fasteners and perform dynamic shaft balancing."
        )
        ensemble_probs[2] = max(ensemble_probs[2], 0.76)

    if is_lubrication_fault:
        fault_triggers.append("Lubrication Failure or Imminent Component Damage")
        recommendations.append(
            f"HALT HIGH-LOAD OPERATION: Inspect lubrication/hydraulic oil viscosity, relief valves, and clogged filters on <strong>{machine_type} ({equipment_sub})</strong>. "
            "Check heat exchanger cooling efficiency and pressure regulator immediately."
        )
        ensemble_probs[3] = max(ensemble_probs[3], 0.84)

    # Secondary Threshold Warnings
    if not fault_triggers:
        if rms_val > 7.0 or temp > 90 or acoustic_db > 85 or p2p_amp > 16.0 or (pressure < 2200 or pressure > 4200):
            fault_triggers.append("Early-Stage Parametric Drift / Elevated Stress")
            recommendations.append(
                f"Increase condition monitoring frequency for <strong>{machine_type} ({equipment_sub})</strong>. "
                "Verify fluid/lubrication levels, inspect cooling pathways, and schedule preventive maintenance within 48 operating hours."
            )
        else:
            fault_triggers.append("Optimal Mechanical & Hydraulic Operation")
            recommendations.append(
                f"Continue standard preventive maintenance schedule for <strong>{machine_type} ({equipment_sub})</strong>. "
                "All vibration, thermal, acoustic, and pressure signatures are within ISO 10816 baseline limits."
            )

    ensemble_probs = ensemble_probs / np.sum(ensemble_probs)

    # Calculate Exact Anomaly Score (0.0 to 100.0)
    vib_stress = min(100.0, max(0.0, ((rms_val - 2.0) / 8.0) * 100.0))
    p2p_stress = min(100.0, max(0.0, ((p2p_amp - 5.0) / 25.0) * 100.0))
    temp_stress = min(100.0, max(0.0, ((temp - 65.0) / 45.0) * 100.0))
    acoustic_stress = min(100.0, max(0.0, ((acoustic_db - 60.0) / 40.0) * 100.0))

    pressure_dev = abs(pressure - 3200.0)
    pressure_stress = min(100.0, max(0.0, ((pressure_dev - 400.0) / 1600.0) * 100.0))

    base_anomaly = (
        0.25 * vib_stress +
        0.20 * p2p_stress +
        0.20 * temp_stress +
        0.20 * acoustic_stress +
        0.15 * pressure_stress
    )

    active_rule_count = sum([is_bearing_gear_fault, is_structural_fault, is_lubrication_fault])
    if active_rule_count >= 1:
        anomaly_score = min(99.4, max(68.0, base_anomaly * 1.35 + (active_rule_count * 12.0)))
    elif rms_val > 7.0 or temp > 90 or acoustic_db > 85:
        anomaly_score = min(67.9, max(42.0, base_anomaly * 1.15))
    else:
        anomaly_score = min(39.0, max(4.2, base_anomaly * 0.75))

    if anomaly_score >= 65.0 or active_rule_count >= 1:
        severity = "Critical"
    elif anomaly_score >= 38.0:
        severity = "Warning"
    else:
        severity = "Normal"

    # Estimate Remaining Useful Life (RUL)
    nominal_life_hrs = 2000.0
    duty_cycle_factor = 1.0 + ((working_hrs - 8.0) / 16.0) * 0.35
    degradation_rate = (anomaly_score / 100.0) ** 1.65
    rul_hours = max(4.0, nominal_life_hrs * (1.0 - degradation_rate) / max(0.7, duty_cycle_factor))
    rul_days = rul_hours / max(1.0, working_hrs)

    return {
        "fault_diagnoses": fault_triggers,
        "recommendations": recommendations,
        "severity": severity,
        "anomaly_score": round(anomaly_score, 1),
        "rul_hours": round(rul_hours, 1),
        "rul_days": round(rul_days, 1),
        "cnn_probs": cnn_probs,
        "rf_probs": rf_probs,
        "svm_probs": svm_probs,
        "ensemble_probs": ensemble_probs
    }

# ==========================================
# 4. SIDEBAR: EQUIPMENT & SENSOR INPUTS
# ==========================================
with st.sidebar:
    st.title("🚜 Equipment & Sensors")
    st.markdown("Configure machinery unit and real-time multi-sensor telemetry.")

    # 1. Equipment Selection Module (Updated with all categories & machines)
    st.subheader("1. Equipment Selection")
    equipment_category = st.selectbox(
        "Machinery Category",
        options=list(EQUIPMENT_DATABASE.keys()),
        index=0
    )

    machine_type = st.selectbox(
        "Select Machine Type",
        options=list(EQUIPMENT_DATABASE[equipment_category].keys()),
        index=0
    )

    monitored_subsystem = st.selectbox(
        "Monitored Machine Unit & Subsystem",
        options=EQUIPMENT_DATABASE[equipment_category][machine_type],
        index=0
    )

    # Quick Diagnostic Preset Scenarios for Testing
    st.markdown("---")
    st.subheader("⚡ Quick Fault Presets")
    preset_mode = st.radio(
        "Load Telemetry Scenario:",
        options=[
            "Custom Manual Sliders",
            "Normal Baseline Operation",
            "Scenario A: Bearing / Gear Fault",
            "Scenario B: Structural Looseness",
            "Scenario C: Lubrication Failure"
        ],
        index=0
    )

    if preset_mode == "Normal Baseline Operation":
        def_p2p, def_rms, def_freq = 6.5, 2.8, 120.0
        def_temp, def_press, def_db, def_hrs = 68.0, 3200.0, 62.0, 8.0
    elif preset_mode == "Scenario A: Bearing / Gear Fault":
        def_p2p, def_rms, def_freq = 14.5, 6.2, 680.0
        def_temp, def_press, def_db, def_hrs = 78.0, 3350.0, 94.0, 11.0
    elif preset_mode == "Scenario B: Structural Looseness":
        def_p2p, def_rms, def_freq = 24.0, 8.5, 65.0
        def_temp, def_press, def_db, def_hrs = 72.0, 3150.0, 74.0, 9.0
    elif preset_mode == "Scenario C: Lubrication Failure":
        def_p2p, def_rms, def_freq = 12.0, 5.4, 240.0
        def_temp, def_press, def_db, def_hrs = 108.0, 1750.0, 81.0, 14.0
    else:
        def_p2p, def_rms, def_freq = 8.0, 3.5, 150.0
        def_temp, def_press, def_db, def_hrs = 70.0, 3200.0, 65.0, 8.0

    st.markdown("---")
    st.subheader("2. Vibration Telemetry")
    p2p_amp = st.slider(
        "Peak-to-Peak Amplitude (mm/s)",
        min_value=1.0, max_value=35.0, value=def_p2p, step=0.5,
        help="Maximum excursion of the time-domain vibration wave. Safe < 14 mm/s."
    )
    rms_val = st.slider(
        "RMS Vibration Velocity (mm/s)",
        min_value=1.0, max_value=12.0, value=def_rms, step=0.1,
        help="Root Mean Square vibration energy (ISO 10816). Threshold > 7.0 mm/s."
    )
    dom_freq = st.slider(
        "Dominant Frequency (Hz)",
        min_value=10.0, max_value=1200.0, value=def_freq, step=10.0,
        help="Primary spectral frequency spike. Low (<180 Hz) = unbalance/looseness; High (>400 Hz) = bearing/gear mesh."
    )

    st.markdown("---")
    st.subheader("3. Multi-Sensor Telemetry")
    temp_val = st.slider(
        "Component Temperature (°C)",
        min_value=40.0, max_value=130.0, value=def_temp, step=1.0,
        help="Surface/oil/winding temperature. Safe operating range: 50°C - 90°C."
    )
    pressure_val = st.slider(
        "Hydraulic / Lube Fluid Pressure (PSI)",
        min_value=1200.0, max_value=5000.0, value=def_press, step=50.0,
        help="Nominal main hydraulic/lubrication pressure baseline: 2,200 - 4,200 PSI."
    )
    acoustic_val = st.slider(
        "Acoustic Emission (dB)",
        min_value=40.0, max_value=115.0, value=def_db, step=1.0,
        help="High-frequency ultrasonic/acoustic decibel level. Threshold > 85 dB."
    )
    working_hrs = st.slider(
        "Daily Shift Working Hours (hrs/day)",
        min_value=1.0, max_value=16.0, value=def_hrs, step=0.5,
        help="Active daily duty cycle used to weight RUL degradation."
    )

    st.markdown("---")
    st.subheader("4. Signal Preprocessing")
    filter_type = st.selectbox(
        "Denoising Algorithm",
        options=[
            "Butterworth Low-Pass Filter",
            "Wavelet Soft-Thresholding (Simulated)",
            "Moving Average Kernel"
        ]
    )
    cutoff_freq = st.slider(
        "Filter Cutoff Frequency (Hz)",
        min_value=150, max_value=1400, value=950, step=50
    )
    noise_level_sim = st.slider(
        "Simulated Background Sensor Noise (σ)",
        min_value=0.2, max_value=5.0, value=1.6, step=0.2
    )

# ==========================================
# 5. EXECUTE SIGNAL PROCESSING & AI ENGINE
# ==========================================
t, raw_sig, clean_sig, xf, yf_raw, yf_clean, snr_db, denoised_rms = generate_and_filter_signals(
    p2p_amp, rms_val, dom_freq, acoustic_val, filter_type, cutoff_freq, noise_level_sim
)

diag_results = run_hybrid_diagnostics(
    machine_type, monitored_subsystem, p2p_amp, rms_val, dom_freq,
    temp_val, pressure_val, acoustic_val, working_hrs, denoised_rms
)

# ==========================================
# 6. MAIN DASHBOARD UI & VISUALIZATIONS
# ==========================================
st.title("⚙️ Predictive Maintenance & AI Diagnostic Simulation")
st.caption(
    f"**Category:** `{equipment_category}`  |  **Machine Type:** `{machine_type}`  |  "
    f"**Subsystem:** `{monitored_subsystem}`  |  **Filter:** `{filter_type}`"
)

# Top KPI Row
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

with kpi1:
    st.metric(
        label="System Health Status",
        value=diag_results["severity"].upper(),
        delta="Normal Baseline" if diag_results["severity"] == "Normal" else "Action Required",
        delta_color="normal" if diag_results["severity"] == "Normal" else "inverse"
    )
with kpi2:
    st.metric(
        label="AI Anomaly Score",
        value=f"{diag_results['anomaly_score']}%",
        delta=f"{round(diag_results['anomaly_score'] - 20.0, 1)}% vs Baseline",
        delta_color="inverse"
    )
with kpi3:
    st.metric(
        label="Estimated RUL (Hours)",
        value=f"{diag_results['rul_hours']} hrs",
        delta=f"~{diag_results['rul_days']} Shifts Left",
        delta_color="normal" if diag_results["rul_hours"] > 500 else "inverse"
    )
with kpi4:
    st.metric(
        label="Denoised RMS Velocity",
