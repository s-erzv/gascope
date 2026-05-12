import { useState, useEffect, useCallback } from "react";
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine,
} from "recharts";
import {
  Zap, RefreshCw, AlertTriangle, TrendingDown,
  Clock, Activity, ArrowDown, Flame,
  CheckCircle2, Eye,
} from "lucide-react";
import "./App.css";

const API_BASE = "https://nupers-gascopez.hf.space";
const REFRESH_MS = 5 * 60 * 1000;

function useIsMobile(bp = 768) {
  const [mobile, setMobile] = useState(() => window.innerWidth < bp);
  useEffect(() => {
    const fn = () => setMobile(window.innerWidth < bp);
    window.addEventListener("resize", fn);
    return () => window.removeEventListener("resize", fn);
  }, [bp]);
  return mobile;
}

interface ActionConfig {
  label: string;
  short: string;
  dot: string;
  bg: string;
  text: string;
  border: string;
  Icon: any;
}

const ACTION_CFG: Record<string, ActionConfig> = {
  EXECUTE_NOW:  { label: "Eksekusi Sekarang", short: "Eksekusi", dot: "#16a34a", bg: "#F0FDF4", text: "#15803d", border: "#BBF7D0", Icon: CheckCircle2 },
  EXECUTE_SOON: { label: "Eksekusi Segera",   short: "Segera",   dot: "#2563EB", bg: "#EFF6FF", text: "#1d4ed8", border: "#BFDBFE", Icon: Clock },
  SPIKE_ALERT:  { label: "Spike Aktif!",      short: "Spike!",   dot: "#DC2626", bg: "#FEF2F2", text: "#b91c1c", border: "#FECACA", Icon: Flame },
  HOLD:         { label: "Tahan Dulu",         short: "Tahan",    dot: "#D97706", bg: "#FFFBEB", text: "#b45309", border: "#FDE68A", Icon: TrendingDown },
  MONITOR:      { label: "Pantau",             short: "Pantau",   dot: "#6B7280", bg: "#F9FAFB", text: "#374151", border: "#E5E7EB", Icon: Eye },
};

interface ZoneConfig { label: string; color: string; bg: string; }
const ZONE_CFG: Record<string, ZoneConfig> = {
  FLOOR:    { label: "Floor",    color: "#16a34a", bg: "#F0FDF4" },
  NORMAL:   { label: "Normal",   color: "#2563EB", bg: "#EFF6FF" },
  ELEVATED: { label: "Elevated", color: "#D97706", bg: "#FFFBEB" },
  SPIKE:    { label: "Spike",    color: "#DC2626", bg: "#FEF2F2" },
};

const fmt  = (n: any, d = 5) => typeof n === "number" ? n.toFixed(d) : "—";
const fmtU = (n: any) => n == null ? "—" : n < 0.01 ? `$${n.toFixed(6)}` : `$${n.toFixed(4)}`;

export default function App() {
  const [data, setData]       = useState<any>(null);
  const [metrics, setMet]     = useState<any>(null);
  const [loading, setLoad]    = useState(true);
  const [error, setErr]       = useState<string | null>(null);
  const [syncing, setSyncing] = useState(false);

  const fetchData = useCallback(async () => {
    setSyncing(true); setErr(null);
    try {
      const [r1, r2] = await Promise.all([
        fetch(`${API_BASE}/api/predict`),
        fetch(`${API_BASE}/api/metrics`),
      ]);
      if (!r1.ok) throw new Error(`Server error (${r1.status})`);
      setData(await r1.json());
      if (r2.ok) setMet(await r2.json());
    } catch (e: any) { setErr(e.message || "An error occurred"); }
    finally { setLoad(false); setSyncing(false); }
  }, []);

  useEffect(() => {
    fetchData();
    const t = setInterval(fetchData, REFRESH_MS);
    return () => clearInterval(t);
  }, [fetchData]);

  return (
    <>
      {loading         ? <Splash /> :
       error || !data  ? <Err msg={error} retry={fetchData} /> :
                         <Dashboard data={data} metrics={metrics} syncing={syncing} onRefresh={fetchData} />}
    </>
  );
}

function Splash() {
  return (
    <div style={{ height: "100dvh", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 16, background: "#F5F4F0" }}>
      <div style={{ width: 48, height: 48, borderRadius: 14, background: "#18181B", display: "flex", alignItems: "center", justifyContent: "center" }}>
        <Activity size={22} color="#F5F4F0" style={{ animation: "spin 1.5s linear infinite" }} />
      </div>
      <p className="mono" style={{ fontSize: 11, color: "#8C8A84", letterSpacing: ".12em", textTransform: "uppercase" }}>Menghubungkan ke Base L2…</p>
    </div>
  );
}

function Err({ msg, retry }: { msg: string | null; retry: () => void }) {
  return (
    <div style={{ height: "100dvh", display: "flex", alignItems: "center", justifyContent: "center", padding: 24 }}>
      <div className="card" style={{ padding: 32, maxWidth: 340, width: "100%", textAlign: "center" }}>
        <div style={{ width: 48, height: 48, borderRadius: 14, background: "#FEF2F2", display: "flex", alignItems: "center", justifyContent: "center", margin: "0 auto 16px" }}>
          <AlertTriangle size={22} color="#DC2626" />
        </div>
        <h2 style={{ fontSize: 17, fontWeight: 700, marginBottom: 8 }}>Koneksi Gagal</h2>
        <p style={{ fontSize: 13, color: "#71717A", marginBottom: 24, lineHeight: 1.6 }}>{msg || "Tidak dapat mengambil data."}</p>
        <button onClick={retry} className="btn-refresh"
          style={{ width: "100%", justifyContent: "center", padding: "11px 24px", background: "#18181B", color: "#F5F4F0", border: "none", borderRadius: 10, fontSize: 13 }}>
          Coba Lagi
        </button>
      </div>
    </div>
  );
}

function Dashboard({ data, metrics, syncing, onRefresh }: { data: any; metrics: any; syncing: boolean; onRefresh: () => void }) {
  const isMobile = useIsMobile();
  const rec      = data.recommendation;
  const cfg      = ACTION_CFG[rec.action] || ACTION_CFG.MONITOR;
  const zone     = ZONE_CFG[rec.zone]     || ZONE_CFG.NORMAL;
  const mape     = metrics?.metrics_overall?.mape;
  const zScore   = data.chain_stats?.z_score   ?? 0;
  const floorGwei = data.chain_stats?.floor_gwei ?? 10;

  return isMobile
    ? <MobileDashboard data={data} metrics={metrics} syncing={syncing} onRefresh={onRefresh} rec={rec} cfg={cfg} zone={zone} mape={mape} zScore={zScore} floorGwei={floorGwei} />
    : <DesktopDashboard data={data} metrics={metrics} syncing={syncing} onRefresh={onRefresh} rec={rec} cfg={cfg} zone={zone} mape={mape} zScore={zScore} floorGwei={floorGwei} />;
}

function MobileDashboard({ data, metrics, syncing, onRefresh, rec, cfg, zone, mape, zScore, floorGwei }: any) {
  return (
    <div style={{ minHeight: "100dvh", background: "#F5F4F0", display: "flex", flexDirection: "column" }}>

      <header style={{ position: "sticky", top: 0, zIndex: 50, background: "#fff", borderBottom: "1.5px solid #ECEAE3", height: 52, display: "flex", alignItems: "center", justifyContent: "space-between", padding: "0 16px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <div style={{ width: 32, height: 32, borderRadius: 10, background: "#18181B", display: "flex", alignItems: "center", justifyContent: "center", position: "relative", flexShrink: 0 }}>
            <Zap size={15} color="#F5F4F0" fill="#F5F4F0" />
            <div style={{ position: "absolute", bottom: 4, right: 4, width: 5, height: 5, borderRadius: "50%", background: cfg.dot, border: "1.5px solid #fff" }} />
          </div>
          <div>
            <div style={{ display: "flex", alignItems: "baseline", gap: 5 }}>
              <span style={{ fontSize: 15, fontWeight: 800, letterSpacing: "-.03em" }}>Gascope</span>
              <span style={{ fontSize: 9, fontWeight: 600, color: "#A8A5A0", letterSpacing: ".08em", textTransform: "uppercase" }}>Base L2</span>
            </div>
          </div>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 5, padding: "4px 9px", borderRadius: 7, background: "#F5F4F0", border: "1.5px solid #ECEAE3", fontSize: 10, fontWeight: 600, color: "#52524E" }}>
            <span className="pulse-dot" style={{ width: 5, height: 5, borderRadius: "50%", background: data.data_quality === "fresh" ? "#16a34a" : "#D97706", display: "inline-block" }} />
            {data.data_quality === "fresh" ? "Live" : "Stale"}
          </div>
          <button className="btn-refresh" onClick={onRefresh} disabled={syncing} style={{ padding: "5px 10px" }}>
            <RefreshCw size={11} className={syncing ? "spin" : ""} />
            {syncing ? "…" : "Perbarui"}
          </button>
        </div>
      </header>

      <div style={{ flex: 1, overflowY: "auto", padding: "12px 14px", display: "flex", flexDirection: "column", gap: 10 }}>

        <div className="card fade-up" style={{ padding: "16px", background: cfg.bg, borderColor: cfg.border }}>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 10 }}>
            <span style={{ fontSize: 9, fontWeight: 700, textTransform: "uppercase", letterSpacing: ".1em", color: cfg.text, opacity: .7 }}>AI Rekomendasi</span>
            <span className="mono" style={{ fontSize: 9, padding: "2px 7px", borderRadius: 5, background: "rgba(255,255,255,.6)", color: cfg.text, fontWeight: 600, border: `1px solid ${cfg.border}` }}>
              {rec.rule_triggered?.split("_").slice(-2).join("_")}
            </span>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 10 }}>
            <div style={{ width: 40, height: 40, borderRadius: 12, background: "rgba(255,255,255,.7)", display: "flex", alignItems: "center", justifyContent: "center", border: `1.5px solid ${cfg.border}`, flexShrink: 0 }}>
              <cfg.Icon size={19} color={cfg.text} />
            </div>
            <div>
              <div style={{ fontSize: 20, fontWeight: 800, letterSpacing: "-.04em", color: cfg.text, lineHeight: 1 }}>{cfg.label}</div>
              <div style={{ display: "flex", gap: 6, marginTop: 5, flexWrap: "wrap" }}>
                <Pill label={`Confidence ${(rec.confidence_score * 100).toFixed(0)}%`} color={cfg.text} bg="rgba(255,255,255,.5)" border={cfg.border} />
                <Pill label={`Z-score ${zScore.toFixed(2)}σ`} color={cfg.text} bg="rgba(255,255,255,.5)" border={cfg.border} />
              </div>
            </div>
          </div>
          <p style={{ fontSize: 12, color: cfg.text, lineHeight: 1.65, opacity: .85, margin: 0 }}>{rec.message}</p>
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
          <Ticker label="Gas Sekarang"    value={fmt(rec.current_fee_gwei)}    unit="Gwei" accent hi />
          <Ticker label="Prediksi Min"    value={fmt(rec.lowest_future_gwei)}  unit="Gwei" />
          <Ticker label="Potensi Hemat"   value={rec.savings_estimate_pct > 0 ? `${rec.savings_estimate_pct.toFixed(1)}%` : "—"} valueColor="#16a34a" />
          <Ticker label="Tunggu Optimal"  value={rec.optimal_wait_minutes != null ? `${rec.optimal_wait_minutes} mnt` : "—"} valueColor="#2563EB" />
          <Ticker label="Network Zone"    value={zone.label} valueColor={zone.color} />
          {mape != null && (
            <div className="card-sm" style={{ padding: "13px 15px" }}>
              <div style={{ fontSize: 9, fontWeight: 700, textTransform: "uppercase", letterSpacing: ".09em", color: "#A8A5A0", marginBottom: 7 }}>MAPE</div>
              <span className="mono" style={{ fontSize: 18, fontWeight: 600 }}>{mape.toFixed(1)}%</span>
            </div>
          )}
        </div>

        <ZonePanel zone={rec.zone} zoneCfg={zone} p25={data.percentiles?.p25} p75={data.percentiles?.p75} zScore={zScore} floorGwei={floorGwei} currentFee={rec.current_fee_gwei} />

        <div className="card" style={{ padding: "16px 14px 12px" }}>
          <div style={{ marginBottom: 10 }}>
            <div style={{ fontSize: 13, fontWeight: 700, letterSpacing: "-.02em" }}>Proyeksi Gas Hari Ini</div>
            <div style={{ fontSize: 9, color: "#A8A5A0", marginTop: 2, fontWeight: 500 }}>
              Forecast 24 jam · Prophet AI · {data.data_source}
            </div>
          </div>
          <div style={{ display: "flex", gap: 10, flexWrap: "wrap", marginBottom: 10 }}>
            <Legend color="#10b981" label="Aktual" />
            <Legend color="#3b82f6" dashed label="Prediksi" />
            <Legend color="#10b981" dashed label="P25 (murah)" />
          </div>
          <div style={{ height: 200 }}>
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={data.chart_data} margin={{ top: 4, right: 4, left: -28, bottom: 0 }}>
                <defs>
                  <linearGradient id="gActual" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#10b981" stopOpacity={.18} />
                    <stop offset="100%" stopColor="#10b981" stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="gForecast" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#3b82f6" stopOpacity={.1} />
                    <stop offset="100%" stopColor="#3b82f6" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid stroke="#F0EEE8" strokeDasharray="0" vertical={false} />
                <XAxis dataKey="time_label" tick={{ fontSize: 8, fill: "#B5B3AD", fontFamily: "'JetBrains Mono',monospace" }} axisLine={false} tickLine={false} dy={6} interval={95} />
                <YAxis tick={{ fontSize: 8, fill: "#B5B3AD", fontFamily: "'JetBrains Mono',monospace" }} tickFormatter={v => v.toFixed(1)} axisLine={false} tickLine={false} domain={["auto", "auto"]} />
                <Tooltip content={<ChartTip />} />
                <ReferenceLine y={data.percentiles?.p25} stroke="#10b981" strokeDasharray="4 3" strokeOpacity={.6} strokeWidth={1.5} />
                <Area type="monotone" dataKey="upper_gwei" stroke="none" fill="#3b82f6" fillOpacity={.05} isAnimationActive={false} />
                <Area type="monotone" dataKey="fee_gwei" stroke="#3b82f6" strokeWidth={1.5} strokeDasharray="5 3" fill="url(#gForecast)" dot={false} isAnimationActive={false} />
                <Area type="monotone" dataKey="actual_fee_gwei" stroke="#10b981" strokeWidth={2} fill="url(#gActual)" dot={false} connectNulls isAnimationActive animationDuration={600} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
          <div style={{ marginTop: 10, paddingTop: 10, borderTop: "1.5px solid #F0EEE8", display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 6 }}>
            <div style={{ display: "flex", gap: 12 }}>
              <StatChip label="Train rows" value={metrics?.train_rows?.toLocaleString() ?? "—"} />
              <StatChip label="Coverage CI" value={metrics ? `${metrics.ci_coverage_pct?.toFixed(1) ?? "—"}%` : "—"} />
            </div>
            <div style={{ fontSize: 9, color: data.data_quality === "fresh" ? "#16a34a" : "#D97706", fontWeight: 700, display: "flex", alignItems: "center", gap: 4 }}>
              <span style={{ width: 5, height: 5, borderRadius: "50%", background: data.data_quality === "fresh" ? "#16a34a" : "#D97706", display: "inline-block" }} />
              {data.latest_data_ts_wib ?? "—"}
            </div>
          </div>
        </div>

        <div className="card" style={{ padding: "14px 14px" }}>
          <div style={{ fontSize: 9, fontWeight: 700, textTransform: "uppercase", letterSpacing: ".1em", color: "#A8A5A0", marginBottom: 10 }}>Estimasi Biaya Transaksi</div>
          <div style={{ display: "flex", flexDirection: "column", gap: 7 }}>
            {data.tx_costs?.map((tx: any) => <TxRow key={tx.tx_type} tx={tx} />)}
          </div>
        </div>

      </div>

      <footer style={{ flexShrink: 0, background: "#fff", borderTop: "1.5px solid #ECEAE3", height: 30, display: "flex", alignItems: "center", justifyContent: "center", padding: "0 16px" }}>
        <span style={{ fontSize: 8, fontWeight: 600, color: "#C4C2BC", textTransform: "uppercase", letterSpacing: ".1em" }} className="mono">
          Gascope · Base Mainnet · {new Date().toLocaleTimeString("id-ID")} WIB
        </span>
      </footer>
    </div>
  );
}

function DesktopDashboard({ data, metrics, syncing, onRefresh, rec, cfg, zone, mape, zScore, floorGwei }: any) {
  return (
    <div style={{ height: "100vh", display: "flex", flexDirection: "column", background: "#F5F4F0", overflow: "hidden" }}>

      <header style={{ flexShrink: 0, background: "#fff", borderBottom: "1.5px solid #ECEAE3", height: 58, display: "flex", alignItems: "center", justifyContent: "space-between", padding: "0 24px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <div style={{ width: 36, height: 36, borderRadius: 11, background: "#18181B", display: "flex", alignItems: "center", justifyContent: "center", position: "relative" }}>
            <Zap size={17} color="#F5F4F0" fill="#F5F4F0" />
            <div style={{ position: "absolute", bottom: 5, right: 5, width: 6, height: 6, borderRadius: "50%", background: cfg.dot, border: "1.5px solid #fff" }} />
          </div>
          <div>
            <div style={{ display: "flex", alignItems: "baseline", gap: 7 }}>
              <span style={{ fontSize: 17, fontWeight: 800, letterSpacing: "-.03em", lineHeight: 1 }}>Gascope</span>
              <span style={{ fontSize: 10, fontWeight: 600, color: "#A8A5A0", letterSpacing: ".08em", textTransform: "uppercase" }}>Base L2</span>
            </div>
            <div style={{ fontSize: 10, color: "#A8A5A0", marginTop: 2, fontWeight: 500 }}>Pantau &amp; Optimalkan Biaya Transaksi</div>
          </div>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 6, padding: "5px 11px", borderRadius: 8, background: "#F5F4F0", border: "1.5px solid #ECEAE3", fontSize: 11, fontWeight: 600, color: "#52524E" }}>
            <span className="pulse-dot" style={{ width: 6, height: 6, borderRadius: "50%", background: data.data_quality === "fresh" ? "#16a34a" : "#D97706", display: "inline-block" }} />
            {data.data_quality === "fresh" ? "Live" : "Stale"}
            <span style={{ color: "#B5B3AD", marginLeft: 2 }}>·</span>
            <span className="mono" style={{ color: "#8C8A84" }}>{data.timestamp_wib?.split(" ")[1] ?? "—"}</span>
          </div>
          {mape != null && (
            <div style={{ padding: "5px 10px", borderRadius: 8, background: "#F5F4F0", border: "1.5px solid #ECEAE3", fontSize: 10, fontWeight: 600, color: "#8C8A84" }} className="mono">
              MAPE {mape.toFixed(1)}%
            </div>
          )}
          <button className="btn-refresh" onClick={onRefresh} disabled={syncing}>
            <RefreshCw size={11} className={syncing ? "spin" : ""} />
            {syncing ? "Sinkron…" : "Perbarui"}
          </button>
        </div>
      </header>

      <main style={{ flex: 1, minHeight: 0, padding: "14px 20px", display: "flex", flexDirection: "column", gap: 12 }}>

        <div style={{ flexShrink: 0, display: "grid", gridTemplateColumns: "repeat(5,1fr)", gap: 10 }}>
          <Ticker label="Gas Sekarang"   value={fmt(rec.current_fee_gwei)}   unit="Gwei" accent hi />
          <Ticker label="Prediksi Minimum" value={fmt(rec.lowest_future_gwei)} unit="Gwei" />
          <Ticker label="Potensi Hemat"  value={rec.savings_estimate_pct > 0 ? `${rec.savings_estimate_pct.toFixed(1)}%` : "—"} valueColor="#16a34a" />
          <Ticker label="Tunggu Optimal" value={rec.optimal_wait_minutes != null ? `${rec.optimal_wait_minutes} menit` : "—"} valueColor="#2563EB" />
          <Ticker label="Network Zone"   value={zone.label} valueColor={zone.color} />
        </div>

        <div style={{ flex: 1, minHeight: 0, display: "grid", gridTemplateColumns: "300px 1fr", gap: 12 }}>

          <div style={{ display: "flex", flexDirection: "column", gap: 10, minHeight: 0 }}>

            <div className="card fade-up" style={{ padding: "18px 18px 16px", flexShrink: 0, background: cfg.bg, borderColor: cfg.border }}>
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 12 }}>
                <span style={{ fontSize: 9, fontWeight: 700, textTransform: "uppercase", letterSpacing: ".1em", color: cfg.text, opacity: .7 }}>AI Rekomendasi</span>
                <span className="mono" style={{ fontSize: 9, padding: "2px 7px", borderRadius: 5, background: "rgba(255,255,255,.6)", color: cfg.text, fontWeight: 600, border: `1px solid ${cfg.border}` }}>
                  {rec.rule_triggered?.split("_").slice(-2).join("_")}
                </span>
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 12 }}>
                <div style={{ width: 44, height: 44, borderRadius: 13, background: "rgba(255,255,255,.7)", display: "flex", alignItems: "center", justifyContent: "center", border: `1.5px solid ${cfg.border}`, flexShrink: 0 }}>
                  <cfg.Icon size={21} color={cfg.text} />
                </div>
                <div>
                  <div style={{ fontSize: 22, fontWeight: 800, letterSpacing: "-.04em", color: cfg.text, lineHeight: 1 }}>{cfg.label}</div>
                  <div style={{ display: "flex", gap: 8, marginTop: 5 }}>
                    <Pill label={`Confidence ${(rec.confidence_score * 100).toFixed(0)}%`} color={cfg.text} bg="rgba(255,255,255,.5)" border={cfg.border} />
                    <Pill label={`Z-score ${zScore.toFixed(2)}σ`} color={cfg.text} bg="rgba(255,255,255,.5)" border={cfg.border} />
                  </div>
                </div>
              </div>
              <p style={{ fontSize: 12, color: cfg.text, lineHeight: 1.65, opacity: .85 }}>{rec.message}</p>
            </div>

            <ZonePanel zone={rec.zone} zoneCfg={zone} p25={data.percentiles?.p25} p75={data.percentiles?.p75} zScore={zScore} floorGwei={floorGwei} currentFee={rec.current_fee_gwei} />

            <div className="card" style={{ flex: 1, minHeight: 0, padding: "14px 16px", display: "flex", flexDirection: "column" }}>
              <div style={{ fontSize: 9, fontWeight: 700, textTransform: "uppercase", letterSpacing: ".1em", color: "#A8A5A0", marginBottom: 12, flexShrink: 0 }}>Estimasi Biaya Transaksi</div>
              <div style={{ flex: 1, overflowY: "auto", display: "flex", flexDirection: "column", gap: 7 }}>
                {data.tx_costs?.map((tx: any) => <TxRow key={tx.tx_type} tx={tx} />)}
              </div>
            </div>
          </div>

          <div className="card" style={{ padding: "18px 20px 14px", display: "flex", flexDirection: "column", minHeight: 0 }}>
            <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", marginBottom: 16, flexShrink: 0 }}>
              <div>
                <div style={{ fontSize: 14, fontWeight: 700, letterSpacing: "-.02em" }}>Proyeksi Gas Hari Ini</div>
                <div style={{ fontSize: 10, color: "#A8A5A0", marginTop: 3, fontWeight: 500 }}>
                  Forecast 24 jam · Prophet AI · interval 5 menit · {data.data_source}
                </div>
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                <Legend color="#10b981" label="Aktual" />
                <Legend color="#3b82f6" dashed label="Prediksi" />
                <Legend color="#10b981" dashed label="P25 (murah)" />
              </div>
            </div>

            <div style={{ flex: 1, minHeight: 0 }}>
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={data.chart_data} margin={{ top: 4, right: 4, left: -22, bottom: 0 }}>
                  <defs>
                    <linearGradient id="gActual" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#10b981" stopOpacity={.18} />
                      <stop offset="100%" stopColor="#10b981" stopOpacity={0} />
                    </linearGradient>
                    <linearGradient id="gForecast" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#3b82f6" stopOpacity={.1} />
                      <stop offset="100%" stopColor="#3b82f6" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid stroke="#F0EEE8" strokeDasharray="0" vertical={false} />
                  <XAxis dataKey="time_label" tick={{ fontSize: 9, fill: "#B5B3AD", fontFamily: "'JetBrains Mono',monospace" }} axisLine={false} tickLine={false} dy={8} interval={47} />
                  <YAxis tick={{ fontSize: 9, fill: "#B5B3AD", fontFamily: "'JetBrains Mono',monospace" }} tickFormatter={v => v.toFixed(1)} axisLine={false} tickLine={false} domain={["auto", "auto"]} />
                  <Tooltip content={<ChartTip />} />
                  <ReferenceLine y={data.percentiles?.p25} stroke="#10b981" strokeDasharray="4 3" strokeOpacity={.6} strokeWidth={1.5} />
                  <Area type="monotone" dataKey="upper_gwei" stroke="none" fill="#3b82f6" fillOpacity={.05} isAnimationActive={false} />
                  <Area type="monotone" dataKey="fee_gwei" stroke="#3b82f6" strokeWidth={1.5} strokeDasharray="5 3" fill="url(#gForecast)" dot={false} isAnimationActive={false} />
                  <Area type="monotone" dataKey="actual_fee_gwei" stroke="#10b981" strokeWidth={2} fill="url(#gActual)" dot={false} connectNulls isAnimationActive animationDuration={600} />
                </AreaChart>
              </ResponsiveContainer>
            </div>

            <div style={{ flexShrink: 0, marginTop: 12, paddingTop: 12, borderTop: "1.5px solid #F0EEE8", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <div style={{ display: "flex", gap: 16 }}>
                <StatChip label="Train rows"  value={metrics?.train_rows?.toLocaleString() ?? "—"} />
                <StatChip label="Coverage CI" value={metrics ? `${metrics.ci_coverage_pct?.toFixed(1) ?? "—"}%` : "—"} />
              </div>
              <div style={{ fontSize: 10, color: data.data_quality === "fresh" ? "#16a34a" : "#D97706", fontWeight: 700, display: "flex", alignItems: "center", gap: 4 }}>
                <span style={{ width: 6, height: 6, borderRadius: "50%", background: data.data_quality === "fresh" ? "#16a34a" : "#D97706", display: "inline-block" }} />
                {data.data_quality === "fresh" ? "Live" : "Stale"} · sumber: {data.latest_data_ts_wib ?? "—"}
              </div>
            </div>
          </div>
        </div>
      </main>

      <footer style={{ flexShrink: 0, background: "#fff", borderTop: "1.5px solid #ECEAE3", height: 34, display: "flex", alignItems: "center", justifyContent: "space-between", padding: "0 24px" }}>
        <span style={{ fontSize: 9, fontWeight: 600, color: "#C4C2BC", textTransform: "uppercase", letterSpacing: ".1em" }}>Gascope · Non-Custodial Monitoring</span>
        <span style={{ fontSize: 9, fontWeight: 600, color: "#C4C2BC", textTransform: "uppercase", letterSpacing: ".1em" }} className="mono">Base Mainnet · {new Date().toLocaleTimeString("id-ID")} WIB</span>
      </footer>
    </div>
  );
}


function ChartTip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  const d = payload[0]?.payload;
  return (
    <div style={{ background: "#18181B", color: "#F5F4F0", padding: "10px 14px", borderRadius: 10, fontSize: 11, fontFamily: "'JetBrains Mono',monospace", lineHeight: 1.7 }}>
      <div style={{ color: "#8C8A84", fontSize: 9, marginBottom: 4, letterSpacing: ".05em" }}>{label} WIB</div>
      {d?.actual_fee_gwei != null && <div style={{ color: "#34d399" }}>Aktual:   {d.actual_fee_gwei.toFixed(5)} Gwei</div>}
      <div style={{ color: "#93c5fd" }}>Prediksi: {d?.fee_gwei?.toFixed(5) ?? "—"} Gwei</div>
      {d?.ratio_forecast != null && <div style={{ color: "#fcd34d", fontSize: 9, marginTop: 2 }}>Ratio: {(d.ratio_forecast * 100).toFixed(1)}%</div>}
    </div>
  );
}

function ZonePanel({ zone, zoneCfg, p25, p75, zScore }: any) {
  const zones = ["FLOOR", "NORMAL", "ELEVATED", "SPIKE"];
  const idx = zones.indexOf(zone);
  return (
    <div className="card-sm" style={{ padding: "12px 14px", flexShrink: 0 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 9 }}>
        <span style={{ fontSize: 9, fontWeight: 700, textTransform: "uppercase", letterSpacing: ".1em", color: "#A8A5A0" }}>Network Zone</span>
        <span style={{ fontSize: 10, fontWeight: 700, color: zoneCfg.color, background: zoneCfg.bg, padding: "2px 8px", borderRadius: 5 }}>{zoneCfg.label}</span>
      </div>
      <div style={{ display: "flex", gap: 3, marginBottom: 9 }}>
        {["FLOOR", "NORMAL", "ELEVATED", "SPIKE"].map((z, i) => {
          const colors: Record<string, string> = { FLOOR: "#16a34a", NORMAL: "#2563EB", ELEVATED: "#D97706", SPIKE: "#DC2626" };
          const active = i === idx;
          return <div key={z} style={{ flex: 1, height: 5, borderRadius: 3, background: active ? colors[z] : "#ECEAE3", transition: "background .3s" }} />;
        })}
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 6 }}>
        <MicroStat label="Z-Score" value={`${zScore?.toFixed(2)}σ`} unit="" />
        <MicroStat label="P25"     value={p25 != null ? fmt(p25, 3) : "—"} unit="Gwei" />
        <MicroStat label="P75"     value={p75 != null ? fmt(p75, 3) : "—"} unit="Gwei" />
      </div>
    </div>
  );
}

function Ticker({ label, value, unit = "", valueColor = "#18181B", accent = false, hi = false }: any) {
  return (
    <div className="card-sm" style={{ padding: "13px 15px", background: accent ? "#18181B" : "#fff", borderColor: accent ? "#18181B" : "#ECEAE3" }}>
      <div style={{ fontSize: 9, fontWeight: 700, textTransform: "uppercase", letterSpacing: ".09em", color: accent ? "#6B6966" : "#A8A5A0", marginBottom: 7 }}>{label}</div>
      <div style={{ display: "flex", alignItems: "baseline", gap: 5 }}>
        <span className="mono" style={{ fontSize: hi ? 22 : 18, fontWeight: 600, letterSpacing: "-.02em", color: accent ? "#F5F4F0" : (valueColor ?? "#18181B") }}>
          {value}
        </span>
        {unit && <span style={{ fontSize: 10, fontWeight: 500, color: accent ? "#4B4846" : "#C4C2BC" }}>{unit}</span>}
      </div>
    </div>
  );
}

function Pill({ label, color, bg, border }: any) {
  return (
    <span style={{ fontSize: 9, fontWeight: 700, padding: "2px 7px", borderRadius: 5, background: bg, color, border: `1px solid ${border}`, letterSpacing: ".02em" }} className="mono">
      {label}
    </span>
  );
}

function MicroStat({ label, value, unit }: any) {
  return (
    <div style={{ background: "#F8F7F3", borderRadius: 8, padding: "7px 9px" }}>
      <div style={{ fontSize: 8, fontWeight: 700, textTransform: "uppercase", letterSpacing: ".08em", color: "#B5B3AD", marginBottom: 3 }}>{label}</div>
      <span className="mono" style={{ fontSize: 12, fontWeight: 600, color: "#18181B" }}>{value}</span>
      {unit && <span style={{ fontSize: 9, color: "#A8A5A0", marginLeft: 2 }}>{unit}</span>}
    </div>
  );
}

function StatChip({ label, value }: any) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 5 }}>
      <span style={{ fontSize: 9, color: "#B5B3AD", fontWeight: 500 }}>{label}</span>
      <span className="mono" style={{ fontSize: 9, fontWeight: 700, color: "#52524E" }}>{value}</span>
    </div>
  );
}

function Legend({ color, dashed = false, label }: any) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 5, fontSize: 9, fontWeight: 600, color: "#A8A5A0" }}>
      <svg width="16" height="4" viewBox="0 0 16 4">
        {dashed
          ? <line x1="0" y1="2" x2="16" y2="2" stroke={color} strokeWidth="1.5" strokeDasharray="4 2" />
          : <line x1="0" y1="2" x2="16" y2="2" stroke={color} strokeWidth="2" />}
      </svg>
      {label}
    </div>
  );
}

function TxRow({ tx }: any) {
  return (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "9px 11px", borderRadius: 9, background: "#F8F7F3", border: "1.5px solid #ECEAE3" }}>
      <div>
        <div style={{ fontSize: 11, fontWeight: 700, marginBottom: 2 }}>{tx.tx_type}</div>
        <div style={{ fontSize: 9, color: "#A8A5A0", fontWeight: 500 }}>{tx.gas_units.toLocaleString()} gas</div>
      </div>
      <div style={{ textAlign: "right" }}>
        <div className="mono" style={{ fontSize: 15, fontWeight: 600 }}>{fmtU(tx.cost_now_usd)}</div>
        <div style={{ display: "flex", alignItems: "center", gap: 4, justifyContent: "flex-end", marginTop: 2 }}>
          <ArrowDown size={9} color="#16a34a" />
          <span style={{ fontSize: 9, fontWeight: 700, color: "#16a34a" }}>Hemat {tx.saving_pct.toFixed(0)}%</span>
          <span style={{ fontSize: 9, color: "#B5B3AD" }}>({fmtU(tx.cost_wait_usd)} jika tunggu)</span>
        </div>
      </div>
    </div>
  );
}