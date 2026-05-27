import joblib
import numpy as np

try:
    print("🔄 Loading models for structural diagnostics...")
    audio_model = joblib.load('dog_audio_xgb.joblib')
    vitals_model = joblib.load('dog_vitals_xgb.joblib')
    
    print("\n==================================================")
    print("📊 DIAGNOSTIC RESULTS")
    print("==================================================")
    
    # Check if the models are XGBoost and have feature names saved
    if hasattr(audio_model, 'feature_names_in_'):
        print(f"✅ Audio Model expects {len(audio_model.feature_names_in_)} features.")
        print("First 10 feature names expected by your model:")
        print(audio_model.feature_names_in_[:10])
    else:
        print(f"✅ Audio Model expects exactly {audio_model.n_features_in_} features (No names exported).")
        
    if hasattr(vitals_model, 'feature_names_in_'):
        print(f"\n✅ Vitals Model expects {len(vitals_model.feature_names_in_)} features.")
        print("Vitals feature names expected:", vitals_model.feature_names_in_)
    else:
        print(f"\n✅ Vitals Model expects exactly {vitals_model.n_features_in_} features.")
    print("==================================================")

except Exception as e:
    print(f"❌ Error inspecting models: {e}")