import sys
import json
sys.path.append('.')

# Import fungsi-fungsi core dari main.py
from main import get_academic_evaluation, get_status, get_analysis

def run_diagnostics():
    print("🔍 MENJALANKAN DIAGNOSTIK API (LightGBM Engine)\n" + "="*55)

    try:
        print("\n1️⃣  SYSTEM STATUS (/api/status)")
        status = get_status()
        print(json.dumps(status, indent=2))
        if status.get("model_degraded"):
            print("⚠️  WARNING: Performa model menurun (MAPE tinggi)!")
    except Exception as e:
        print(f"❌ ERROR di get_status(): {e}")

    try:
        print("\n2️⃣  ACADEMIC EVALUATION (/api/academic-evaluation)")
        eval_metrics = get_academic_evaluation()
        print(json.dumps(eval_metrics, indent=2))
    except Exception as e:
        print(f"❌ ERROR di get_academic_evaluation(): {e}")

    try:
        print("\n3️⃣  PREDICTIVE ANALYSIS & RULE ENGINE (/api/analysis)")
        analysis = get_analysis()
        
        # Ekstrak data krusial untuk log singkat
        cond = analysis["current_conditions"]
        rule = analysis["rule_engine_state"]
        savings = analysis["savings_potential"]
        
        print(f"  ➜ Fee Saat Ini : {cond['fee_gwei']} Gwei ({cond['fee_zone']})")
        print(f"  ➜ Rekomendasi  : {rule['action']} ({rule['message']})")
        print(f"  ➜ Potensi Hemat: {savings['savings_pct']}%")
        
        # Print JSON utuh untuk melihat struktur data lengkap
        print("\n[Detail JSON Analysis]")
        print(json.dumps(analysis, indent=2))
        
    except Exception as e:
        print(f"❌ ERROR di get_analysis(): {e}")

    print("\n" + "="*55)
    print("✅ Diagnostik selesai! Jika tidak ada error, backend 100% siap untuk production.")

if __name__ == "__main__":
    run_diagnostics()