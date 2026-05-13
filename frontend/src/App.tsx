import { useState, useEffect, useCallback, useMemo } from "react";
import {
  AreaChart, Area, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine,
} from "recharts";
import {
  Zap, RefreshCw, AlertTriangle, TrendingDown,
  Clock, Activity, ArrowDown, Flame,
  CheckCircle2, Eye, Info,
  TrendingUp, BarChart2, Layers,
} from "lucide-react";
import "./App.css";

const API_BASE = "https://nupers-gascopez.hf.space";  
// const API_BASE = "http://localhost:8000";
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
  HOLD:         { label: "Tahan Dulu",        short: "Tahan",    dot: "#D97706", bg: "#FFFBEB", text: "#b45309", border: "#FDE68A", Icon: TrendingDown },
  MONITOR:      { label: "Pantau",            short: "Pantau",   dot: "#6B7280", bg: "#F9FAFB", text: "#374151", border: "#E5E7EB", Icon: Eye },
};

interface ZoneConfig { label: string; color: string; bg: string; desc: string; }
const ZONE_CFG: Record<string, ZoneConfig> = {
  FLOOR:    { label: "Termurah",   color: "#16a34a", bg: "#F0FDF4", desc: "Fee di harga terendah — waktu terbaik" },
  NORMAL:   { label: "Normal",     color: "#2563EB", bg: "#EFF6FF", desc: "Fee dalam kisaran wajar"               },
  ELEVATED: { label: "Meningkat",  color: "#D97706", bg: "#FFFBEB", desc: "Fee mulai tinggi, pertimbangkan tunggu" },
  SPIKE:    { label: "Lonjakan!",  color: "#DC2626", bg: "#FEF2F2", desc: "Fee sedang lonjak — sebaiknya tunggu"  },
};

const fmt  = (n: any, d = 5) => typeof n === "number" ? n.toFixed(d) : "—";
const fmtU = (n: any) => n == null ? "—" : n < 0.01 ? `$${n.toFixed(6)}` : `$${n.toFixed(4)}`;

function formatGweiAxisTick(v: number, span: number) {
  if (!Number.isFinite(v)) return "—";
  if (span < 0.0002) return v.toFixed(8);
  if (span < 0.002) return v.toFixed(6);
  if (span < 0.02) return v.toFixed(5);
  if (span < 0.15) return v.toFixed(4);
  if (span < 1.5) return v.toFixed(3);
  return v.toFixed(2);
}

function buildChartYScale(chartData: any[] | undefined) {
  const vals: number[] = [];
  for (const d of chartData ?? []) {
    for (const k of ["actual_fee_gwei", "fee_gwei", "upper_gwei"] as const) {
      const v = d[k];
      if (typeof v === "number" && Number.isFinite(v)) vals.push(v);
    }
  }
  let lo = vals.length ? Math.min(...vals) : 0;
  let hi = vals.length ? Math.max(...vals) : 0.02;
  const pad = Math.max((hi - lo) * 0.12, hi * 0.035, 1e-9);
  lo = Math.max(0, lo - pad);
  hi = hi + pad;
  if (hi <= lo) hi = lo + 0.0005;
  const span = hi - lo;
  return {
    domain: [lo, hi] as [number, number],
    tickFmt: (v: number) => formatGweiAxisTick(v, span),
    span,
  };
}

const JKT_TZ = "Asia/Jakarta";

function jktAxisTick(ms: number, compact: boolean) {
  return new Intl.DateTimeFormat("id-ID", {
    timeZone: JKT_TZ,
    day: "numeric",
    month: compact ? "numeric" : "short",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  }).format(new Date(ms));
}

function chartEpochTicks(chartData: any[], tickCount: number): number[] {
  const xs = chartData.map((d) => d.time_epoch_ms).filter((x) => typeof x === "number");
  if (!xs.length) return [];
  if (xs.length === 1) return xs;
  const lo = Math.min(...xs);
  const hi = Math.max(...xs);
  if (hi <= lo) return [lo];
  const n = Math.max(2, tickCount);
  // Snap ticks to round WIB hour boundaries (1/2/3/6/12h step depending on span)
  const HR = 3600 * 1000;
  const spanHr = (hi - lo) / HR;
  const stepHr = spanHr <= 6 ? 1 : spanHr <= 12 ? 2 : spanHr <= 24 ? 3 : spanHr <= 48 ? 6 : 12;
  const stepMs = stepHr * HR;
  // WIB offset = +7h: shift epoch so floor() snaps to WIB hour
  const WIB_MS = 7 * HR;
  const firstSnap = Math.ceil((lo + WIB_MS) / stepMs) * stepMs - WIB_MS;
  const ticks: number[] = [];
  for (let t = firstSnap; t <= hi && ticks.length < n + 4; t += stepMs) ticks.push(t);
  if (!ticks.length) return [lo, hi];
  return ticks;
}

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

function GasProjectionChart({ chartData, compact, p90Ref }: { chartData: any[]; compact: boolean; p90Ref?: number }) {
  const gid = compact ? "chM" : "chD";
  const { domain, tickFmt } = useMemo(() => buildChartYScale(chartData), [chartData]);
  const xTicks = useMemo(() => chartEpochTicks(chartData, compact ? 5 : 7), [chartData, compact]);
  const yTick = compact ? 8 : 9;
  const yAxisW = compact ? 50 : 56;

  return (
    <AreaChart
      data={chartData}
      margin={compact ? { top: 4, right: 2, left: 0, bottom: 4 } : { top: 6, right: 6, left: 2, bottom: 6 }}
    >
      <defs>
        <linearGradient id={`${gid}-act`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#10b981" stopOpacity={0.14} />
          <stop offset="100%" stopColor="#10b981" stopOpacity={0} />
        </linearGradient>
        <linearGradient id={`${gid}-band`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#2563eb" stopOpacity={0.12} />
          <stop offset="100%" stopColor="#2563eb" stopOpacity={0.02} />
        </linearGradient>
      </defs>
      <CartesianGrid stroke="#EDEAE4" strokeDasharray="0" vertical={false} />
      <XAxis
        type="number"
        dataKey="time_epoch_ms"
        domain={["dataMin", "dataMax"]}
        ticks={xTicks}
        tickFormatter={(ms) => jktAxisTick(ms as number, compact)}
        tick={{ fontSize: compact ? 7 : 8, fill: "#A8A29E", fontFamily: "'JetBrains Mono',monospace" }}
        axisLine={false}
        tickLine={false}
        dy={compact ? 6 : 8}
      />
      <YAxis
        domain={domain}
        tickFormatter={tickFmt}
        tick={{ fontSize: yTick, fill: "#A8A29E", fontFamily: "'JetBrains Mono',monospace" }}
        axisLine={false}
        tickLine={false}
        width={yAxisW}
        tickCount={compact ? 5 : 6}
      />
      <Tooltip content={<ChartTip />} />
      {/* P90 spike reference line */}
      {p90Ref != null && (
        <ReferenceLine
          y={p90Ref}
          stroke="#D97706"
          strokeDasharray="3 3"
          strokeWidth={1}
          label={{ value: "P90 historis", position: "insideTopRight", fontSize: 8, fill: "#D97706", fontFamily: "'JetBrains Mono',monospace" }}
        />
      )}
      {/* Forecast confidence band (P10–P90) */}
      <Area
        type="monotone"
        dataKey="upper_gwei"
        stroke="none"
        fill={`url(#${gid}-band)`}
        dot={false}
        connectNulls
        isAnimationActive={false}
        legendType="none"
      />
      {/* Historical actual */}
      <Area
        type="monotone"
        dataKey="actual_fee_gwei"
        stroke="#0d9488"
        strokeWidth={compact ? 1.5 : 1.75}
        fill={`url(#${gid}-act)`}
        dot={false}
        connectNulls
        isAnimationActive
        animationDuration={compact ? 350 : 500}
      />
      {/* Forecast median line */}
      <Line
        type="monotone"
        dataKey="fee_gwei"
        stroke="#2563eb"
        strokeWidth={compact ? 1.5 : 1.75}
        strokeDasharray="5 4"
        dot={false}
        connectNulls
        isAnimationActive={false}
      />
    </AreaChart>
  );
}

function ModelTooltip({ metrics }: { metrics: any }) {
  if (!metrics) return null;
  const r2 = metrics?.metrics_overall?.r2;
  const mae = metrics?.metrics_overall?.mae;
  const arch = metrics?.architecture ?? "LightGBM";
  
  return (
    <div className="model-info-trigger" style={{ position: "relative", display: "flex", alignItems: "center", cursor: "help" }}>
      <Info size={12} color="#C4C2BC" />
      <div className="model-info-content" style={{
        position: "absolute",
        bottom: "100%",
        left: "50%",
        transform: "translateX(-50%)",
        marginBottom: 8,
        background: "#27272a",
        color: "#fff",
        padding: "10px 12px",
        borderRadius: 8,
        fontSize: 10,
        width: 180,
        zIndex: 100,
        pointerEvents: "none",
        lineHeight: 1.5,
        boxShadow: "0 10px 15px -3px rgba(0,0,0,0.1)"
      }}>
        <div style={{ fontWeight: 700, marginBottom: 4, borderBottom: "1px solid #3f3f46", paddingBottom: 4 }}>AI Model Summary</div>
        <div>Architecture: <span className="mono">{arch.split(" ")[0]}</span></div>
        {r2 != null && <div>R² Score: <span className="mono">{r2.toFixed(4)}</span></div>}
        {mae != null && <div>MAE: <span className="mono">{mae < 0.0001 ? mae.toExponential(2) : mae.toFixed(6)} Gwei</span></div>}
        <div style={{ marginTop: 4, fontSize: 8, color: "#a1a1aa", fontStyle: "italic" }}>
          LightGBM Quantile Regression
        </div>
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
  const zScore   = data.chain_stats?.z_score    ?? 0;

  const todayChartData = useMemo(() => {
    if (!data?.chart_data) return [];

    const now = data.timestamp_utc ? new Date(data.timestamp_utc + "Z") : new Date();

    const startOfDay = new Date(now);
    startOfDay.setUTCHours(0 - 7, 0, 0, 0);
    
    const endOfDay = new Date(startOfDay);
    endOfDay.setUTCHours(startOfDay.getUTCHours() + 23, 59, 59, 999);

    const sTs = startOfDay.getTime();
    const eTs = endOfDay.getTime();

    return data.chart_data.filter((d: any) => 
      d.time_epoch_ms >= sTs && d.time_epoch_ms <= eTs
    );
  }, [data]);

  const p90Ref = data.percentiles?.p90 as number | undefined;

  return isMobile
    ? <MobileDashboard  data={data} metrics={metrics} syncing={syncing} onRefresh={onRefresh} rec={rec} cfg={cfg} zone={zone} mape={mape} zScore={zScore} todayChartData={todayChartData} p90Ref={p90Ref} />
    : <DesktopDashboard data={data} metrics={metrics} syncing={syncing} onRefresh={onRefresh} rec={rec} cfg={cfg} zone={zone} mape={mape} zScore={zScore} todayChartData={todayChartData} p90Ref={p90Ref} />;
}

function MobileDashboard({ data, metrics, syncing, onRefresh, rec, cfg, zone, mape, zScore, todayChartData, p90Ref }: any) {
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
          <Ticker label="Gas Sekarang"   value={fmt(rec.current_fee_gwei)}   unit="Gwei" accent hi />
          <Ticker label="Prediksi Min"   value={fmt(rec.lowest_future_gwei)}  unit="Gwei" />
          <Ticker 
            label="Potensi Hemat"  
            value={zone.label === "Floor" ? "Maksimal" : `${rec.savings_estimate_pct.toFixed(1)}%`} 
            valueColor="#16a34a" 
          />
          <Ticker 
            label="Tunggu Optimal" 
            value={zone.label === "Floor" || rec.optimal_wait_minutes === 0 ? "Sekarang" : `~ ${rec.optimal_wait_minutes} mnt`} 
            valueColor={zone.label === "Floor" || rec.optimal_wait_minutes === 0 ? "#16a34a" : "#2563EB"} 
          />
          <Ticker label="Network Zone"   value={zone.label} valueColor={zone.color} />
          {mape != null && (
            <div className="card-sm" style={{ padding: "13px 15px" }}>
              <div style={{ fontSize: 9, fontWeight: 700, textTransform: "uppercase", letterSpacing: ".09em", color: "#A8A5A0", marginBottom: 7 }}>MAPE</div>
              <span className="mono" style={{ fontSize: 18, fontWeight: 600 }}>{mape.toFixed(1)}%</span>
            </div>
          )}
        </div>

        <ZonePanel zone={rec.zone} zoneCfg={zone} p25={data.percentiles?.p25} p75={data.percentiles?.p75} />

        <PerspectivePanel perspectives={data.perspectives} />

        <div className="card" style={{ padding: "16px 14px 12px" }}>
          <div style={{ marginBottom: 12 }}>
            <div style={{ fontSize: 13, fontWeight: 700, letterSpacing: "-.02em" }}>Proyeksi Gas Hari Ini</div>
          </div>
          <div style={{ display: "flex", gap: 12, flexWrap: "wrap", marginBottom: 8 }}>
            <Legend color="#0d9488" label="Aktual" />
            <Legend color="#2563eb" dashed label="Prediksi AI" />
            <Legend color="#2563eb" label="Batas Risiko" />
          </div>
          <div style={{ height: 200 }}>
            <ResponsiveContainer width="100%" height="100%">
              <GasProjectionChart chartData={todayChartData} compact p90Ref={p90Ref} />
            </ResponsiveContainer>
          </div>
          <div style={{ marginTop: 10, paddingTop: 10, borderTop: "1.5px solid #F0EEE8", display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 6 }}>
            <div style={{ display: "flex", gap: 12 }}>
              <StatChip label="AI Engine" value={metrics?.architecture?.split(" ")[0] ?? "LightGBM"} />
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

      <footer style={{ flexShrink: 0, background: "#fff", borderTop: "1.5px solid #ECEAE3", height: 32, display: "flex", alignItems: "center", justifyContent: "space-between", padding: "0 16px" }}>
        <span style={{ fontSize: 8, fontWeight: 600, color: "#C4C2BC", textTransform: "uppercase", letterSpacing: ".1em" }} className="mono">
          Gascope · Base · {new Date().toLocaleTimeString("id-ID")}
        </span>
        <ModelTooltip metrics={metrics} />
      </footer>
    </div>
  );
}

function DesktopDashboard({ data, metrics, syncing, onRefresh, rec, cfg, zone, mape, zScore, todayChartData, p90Ref }: any) {
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
          <Ticker 
            label="Potensi Hemat"  
            value={zone.label === "Floor" ? "Maksimal" : `${rec.savings_estimate_pct.toFixed(1)}%`} 
            valueColor="#16a34a" 
          />
          <Ticker 
            label="Waktu Optimal" 
            value={zone.label === "Floor" || rec.optimal_wait_minutes === 0 ? "Sekarang" : `~ ${rec.optimal_wait_minutes} menit`} 
            valueColor={zone.label === "Floor" || rec.optimal_wait_minutes === 0 ? "#16a34a" : "#2563EB"} 
          />
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

            <ZonePanel zone={rec.zone} zoneCfg={zone} p25={data.percentiles?.p25} p75={data.percentiles?.p75} />

            <div className="card" style={{ flex: 1, minHeight: 0, padding: "14px 16px", display: "flex", flexDirection: "column" }}>
              <div style={{ fontSize: 9, fontWeight: 700, textTransform: "uppercase", letterSpacing: ".1em", color: "#A8A5A0", marginBottom: 12, flexShrink: 0 }}>Estimasi Biaya Transaksi</div>
              <div style={{ flex: 1, overflowY: "auto", display: "flex", flexDirection: "column", gap: 7 }}>
                {data.tx_costs?.map((tx: any) => <TxRow key={tx.tx_type} tx={tx} />)}
              </div>
            </div>
          </div>

          <div style={{ flex: 1, minHeight: 0, display: "flex", flexDirection: "column", gap: 10 }}>
            <PerspectiveStrip perspectives={data.perspectives} />
            <div className="card" style={{ padding: "18px 20px 14px", display: "flex", flexDirection: "column", minHeight: 0, flex: 1 }}>
            <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", marginBottom: 16, flexShrink: 0 }}>
              <div>
                <div style={{ fontSize: 14, fontWeight: 700, letterSpacing: "-.02em" }}>Proyeksi Gas Hari Ini</div>
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: 14, flexWrap: "wrap", justifyContent: "flex-end" }}>
                <Legend color="#0d9488" label="Aktual" />
                <Legend color="#2563eb" dashed label="Prediksi" />
              </div>
            </div>

            <div style={{ flex: 1, minHeight: 0 }}>
              <ResponsiveContainer width="100%" height="100%">
                <GasProjectionChart chartData={todayChartData} compact={false} p90Ref={p90Ref} />
              </ResponsiveContainer>
            </div>

            <div style={{ flexShrink: 0, marginTop: 12, paddingTop: 12, borderTop: "1.5px solid #F0EEE8", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <div style={{ display: "flex", gap: 16 }}>
                <StatChip label="AI Engine"  value={metrics?.architecture?.split(" ")[0] ?? "LightGBM"} />
                <StatChip label="Coverage CI" value={metrics ? `${metrics.ci_coverage_pct?.toFixed(1) ?? "—"}%` : "—"} />
              </div>
              <div style={{ fontSize: 10, color: data.data_quality === "fresh" ? "#16a34a" : "#D97706", fontWeight: 700, display: "flex", alignItems: "center", gap: 4 }}>
                <span style={{ width: 6, height: 6, borderRadius: "50%", background: data.data_quality === "fresh" ? "#16a34a" : "#D97706", display: "inline-block" }} />
                {data.data_quality === "fresh" ? "Live" : "Stale"} · sumber: {data.latest_data_ts_wib ?? "—"}
              </div>
            </div>
            </div>
          </div>
        </div>
      </main>

      <footer style={{ flexShrink: 0, background: "#fff", borderTop: "1.5px solid #ECEAE3", height: 34, display: "flex", alignItems: "center", justifyContent: "space-between", padding: "0 24px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <span style={{ fontSize: 9, fontWeight: 600, color: "#C4C2BC", textTransform: "uppercase", letterSpacing: ".1em" }}>Gascope · Non-Custodial Monitoring</span>
          <ModelTooltip metrics={metrics} />
        </div>
        <span style={{ fontSize: 9, fontWeight: 600, color: "#C4C2BC", textTransform: "uppercase", letterSpacing: ".1em" }} className="mono">Base Mainnet · {new Date().toLocaleTimeString("id-ID")} WIB</span>
      </footer>
    </div>
  );
}

function ChartTip({ active, payload }: any) {
  if (!active || !payload?.length) return null;
  const d = payload[0]?.payload;
  if (!d) return null;

  const hasActual   = d.actual_fee_gwei != null;
  const hasForecast = d.fee_gwei != null;
  const hasBand     = d.upper_gwei != null && d.lower_gwei != null;
  const isHistorical = d.segment === "historical";

  const headerLabel = d.time_label
    ?? new Intl.DateTimeFormat("id-ID", {
         timeZone: JKT_TZ, day: "numeric", month: "short",
         hour: "2-digit", minute: "2-digit", hour12: false,
       }).format(new Date(d.time_epoch_ms));

  return (
    <div style={{ background: "#18181B", color: "#F5F4F0", padding: "10px 14px", borderRadius: 10, fontSize: 11, fontFamily: "'JetBrains Mono',monospace", lineHeight: 1.7 }}>
      <div style={{ color: "#8C8A84", fontSize: 9, marginBottom: 4, letterSpacing: ".05em" }}>
        {headerLabel}{!d.time_label?.includes("WIB") && " WIB"}
      </div>
      {hasActual   && <div style={{ color: "#34d399" }}>Aktual:    {d.actual_fee_gwei.toFixed(5)} Gwei</div>}
      {hasForecast && <div style={{ color: "#93c5fd" }}>{isHistorical ? "Model fit" : "Prediksi"}: {d.fee_gwei.toFixed(5)} Gwei</div>}
      {hasBand     && <div style={{ color: "#fbbf24", fontSize: 9 }}>Batas atas: {d.upper_gwei.toFixed(5)} Gwei</div>}
      {d?.ratio_forecast != null && <div style={{ color: "#fcd34d", fontSize: 9, marginTop: 2 }}>Ratio Jaringan: {(d.ratio_forecast * 100).toFixed(1)}%</div>}
    </div>
  );
}

function ZonePanel({ zone, zoneCfg, p25, p75 }: any) {
  const zones = ["FLOOR", "NORMAL", "ELEVATED", "SPIKE"];
  const idx = zones.indexOf(zone);
  const isP2575Equal = p25 != null && p75 != null && p25 === p75;

  return (
    <div className="card-sm" style={{ padding: "12px 14px", flexShrink: 0 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 4 }}>
        <span style={{ fontSize: 9, fontWeight: 700, textTransform: "uppercase", letterSpacing: ".1em", color: "#A8A5A0" }}>Status Jaringan</span>
        <span style={{ fontSize: 10, fontWeight: 700, color: zoneCfg.color, background: zoneCfg.bg, padding: "2px 8px", borderRadius: 5 }}>{zoneCfg.label}</span>
      </div>
      <div style={{ fontSize: 9, color: "#8C8A84", marginBottom: 8 }}>{zoneCfg.desc}</div>
      <div style={{ display: "flex", gap: 3, marginBottom: 9 }}>
        {(["FLOOR", "NORMAL", "ELEVATED", "SPIKE"] as const).map((z, i) => {
          const colors: Record<string, string> = { FLOOR: "#16a34a", NORMAL: "#2563EB", ELEVATED: "#D97706", SPIKE: "#DC2626" };
          const active = i === idx;
          return <div key={z} style={{ flex: 1, height: 5, borderRadius: 3, background: active ? colors[z] : "#ECEAE3", transition: "background .3s" }} />;
        })}
      </div>
      <div style={{ display: "grid", gridTemplateColumns: isP2575Equal ? "1fr" : "1fr 1fr", gap: 6 }}>
        {isP2575Equal ? (
          <MicroStat label="Kisaran Normal" value={fmt(p25, 4)} unit="Gwei" />
        ) : (
          <>
            <MicroStat label="Batas Bawah Normal" value={p25 != null ? fmt(p25, 4) : "—"} unit="Gwei" />
            <MicroStat label="Batas Atas Normal"  value={p75 != null ? fmt(p75, 4) : "—"} unit="Gwei" />
          </>
        )}
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

const SIGNAL_CFG: Record<string, { color: string; bg: string; label: string }> = {
  bullish: { color: "#16a34a", bg: "#F0FDF4", label: "Aman"    },
  neutral: { color: "#6B7280", bg: "#F9FAFB", label: "Normal"  },
  bearish: { color: "#DC2626", bg: "#FEF2F2", label: "Waspada" },
};

const PERSPECTIVE_META: Record<string, { label: string; Icon: any }> = {
  statistical: { label: "Harga Kini",  Icon: BarChart2   },
  trend:       { label: "Tren",        Icon: TrendingUp  },
  seasonality: { label: "Pola Jam",    Icon: Clock       },
  ml_forecast: { label: "Prediksi AI", Icon: Activity    },
  regime:      { label: "Kondisi",     Icon: Layers      },
};

function perspectiveInsight(key: string, p: any): string {
  switch (key) {
    case "statistical": {
      const pct = p.percentile_rank ?? 50;
      if (pct <= 25) return `Lebih murah dari ${(100 - pct).toFixed(0)}% waktu`;
      if (pct >= 75) return `Lebih mahal dari ${pct.toFixed(0)}% waktu`;
      return `Harga dalam kisaran normal`;
    }
    case "trend": {
      const abs1h = Math.abs(p.change_1h_pct ?? 0);
      if (p.direction === "rising")  return `Naik ${abs1h.toFixed(1)}% dalam 1 jam`;
      if (p.direction === "falling") return `Turun ${abs1h.toFixed(1)}% dalam 1 jam`;
      return `Stabil dalam 1 jam terakhir`;
    }
    case "seasonality": {
      const rate = p.hour_spike_rate_pct ?? 0;
      if (rate < 5)  return `Jam tenang — risiko lonjak ${rate.toFixed(0)}%`;
      if (rate > 15) return `Jam ramai — risiko lonjak ${rate.toFixed(0)}%`;
      return `Jam biasa — risiko lonjak ${rate.toFixed(0)}%`;
    }
    case "ml_forecast": {
      const riskTxt = p.spike_prob_1h != null ? ` · risiko lonjak ${(p.spike_prob_1h * 100).toFixed(0)}%` : "";
      if (p.forecast_trend === "rising")  return `AI: fee akan naik${riskTxt}`;
      if (p.forecast_trend === "falling") return `AI: fee akan turun${riskTxt}`;
      return `AI: fee stabil 1 jam${riskTxt}`;
    }
    case "regime": {
      if (p.type === "floor_sustained") return `Stabil di harga terendah ${p.consecutive_floor_hours}j`;
      if (p.type === "spike_active")    return `Fee sedang tinggi — coba tunggu`;
      return `Fee dalam transisi normal`;
    }
    default:
      return p.assessment ?? "";
  }
}

function PerspectiveStrip({ perspectives }: { perspectives: any }) {
  if (!perspectives) return null;
  const keys = ["statistical", "trend", "seasonality", "ml_forecast", "regime"] as const;

  const counts = { bullish: 0, neutral: 0, bearish: 0 };
  keys.forEach(k => { const s = perspectives[k]?.signal; if (s in counts) (counts as any)[s]++; });

  const verdictText  = counts.bearish >= 2
    ? `${counts.bearish} Waspada — pertimbangkan tunggu`
    : counts.bullish >= 3
    ? `${counts.bullish} Aman — oke untuk transaksi sekarang`
    : `Kondisi normal, fee wajar`;
  const verdictColor = counts.bearish >= 2 ? "#DC2626" : counts.bullish >= 3 ? "#16a34a" : "#6B7280";

  return (
    <div className="card" style={{ padding: "11px 16px", flexShrink: 0 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
        <div style={{ fontSize: 9, fontWeight: 700, textTransform: "uppercase", letterSpacing: ".1em", color: "#A8A5A0" }}>
          Analisis Multi-Sudut
        </div>
        <div style={{ fontSize: 9, fontWeight: 700, color: verdictColor }}>{verdictText}</div>
      </div>
      <div style={{ display: "flex", gap: 6 }}>
        {keys.map((key) => {
          const p = perspectives[key];
          if (!p) return null;
          const sig  = SIGNAL_CFG[p.signal] ?? SIGNAL_CFG.neutral;
          const meta = PERSPECTIVE_META[key];
          const Icon = meta.Icon;
          return (
            <div key={key} title={p.assessment} style={{
              flex: 1, display: "flex", flexDirection: "column", gap: 4,
              padding: "8px 10px", borderRadius: 8,
              background: sig.bg, border: `1.5px solid ${sig.color}22`,
              cursor: "default",
            }}>
              <div style={{ display: "flex", alignItems: "center", gap: 5 }}>
                <Icon size={11} color={sig.color} />
                <span style={{ fontSize: 8, fontWeight: 700, textTransform: "uppercase", letterSpacing: ".07em", color: sig.color }}>
                  {meta.label}
                </span>
              </div>
              <div style={{ fontSize: 9, fontWeight: 700, color: sig.color }}>{sig.label}</div>
              <span style={{ fontSize: 8, color: "#6B7280", lineHeight: 1.4 }}>{perspectiveInsight(key, p)}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function PerspectivePanel({ perspectives }: { perspectives: any }) {
  if (!perspectives) return null;
  const keys = ["statistical", "trend", "seasonality", "ml_forecast", "regime"] as const;

  const counts = { bullish: 0, neutral: 0, bearish: 0 };
  keys.forEach(k => { const s = perspectives[k]?.signal; if (s in counts) (counts as any)[s]++; });

  const verdictText  = counts.bearish >= 2
    ? "Pertimbangkan tunggu sebentar"
    : counts.bullish >= 3
    ? "Kondisi bagus untuk transaksi"
    : "Fee dalam kondisi normal";
  const verdictColor = counts.bearish >= 2 ? "#DC2626" : counts.bullish >= 3 ? "#16a34a" : "#6B7280";
  const verdictBg    = counts.bearish >= 2 ? "#FEF2F2" : counts.bullish >= 3 ? "#F0FDF4" : "#F9FAFB";

  return (
    <div className="card" style={{ padding: "14px 16px" }}>
      <div style={{ fontSize: 9, fontWeight: 700, textTransform: "uppercase", letterSpacing: ".1em", color: "#A8A5A0", marginBottom: 8 }}>
        Analisis Multi-Sudut
      </div>
      <div style={{
        display: "flex", alignItems: "center", gap: 8,
        padding: "9px 12px", borderRadius: 9, marginBottom: 10,
        background: verdictBg, border: `1.5px solid ${verdictColor}33`,
      }}>
        <div style={{ flex: 1 }}>
          <div style={{ fontSize: 11, fontWeight: 700, color: verdictColor, marginBottom: 2 }}>{verdictText}</div>
          <div style={{ fontSize: 9, color: "#8C8A84" }}>
            {counts.bullish} aman · {counts.neutral} normal · {counts.bearish} waspada dari 5 sudut pandang
          </div>
        </div>
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 7 }}>
        {keys.map((key) => {
          const p = perspectives[key];
          if (!p) return null;
          const sig  = SIGNAL_CFG[p.signal] ?? SIGNAL_CFG.neutral;
          const meta = PERSPECTIVE_META[key];
          const Icon = meta.Icon;
          return (
            <div key={key} style={{
              display: "flex", alignItems: "flex-start", gap: 10,
              padding: "9px 11px", borderRadius: 9,
              background: sig.bg, border: `1.5px solid ${sig.color}22`,
            }}>
              <div style={{
                width: 28, height: 28, borderRadius: 8, flexShrink: 0,
                background: "rgba(255,255,255,.7)", display: "flex", alignItems: "center", justifyContent: "center",
                border: `1.5px solid ${sig.color}44`,
              }}>
                <Icon size={13} color={sig.color} />
              </div>
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 3 }}>
                  <span style={{ fontSize: 9, fontWeight: 700, textTransform: "uppercase", letterSpacing: ".08em", color: "#8C8A84" }}>
                    {meta.label}
                  </span>
                  <span style={{
                    fontSize: 8, fontWeight: 700, padding: "1px 6px", borderRadius: 4,
                    background: sig.bg, color: sig.color, border: `1px solid ${sig.color}55`,
                    letterSpacing: ".04em",
                  }}>
                    {sig.label}
                  </span>
                </div>
                <p style={{ fontSize: 10, color: "#374151", lineHeight: 1.5, margin: 0 }}>
                  {perspectiveInsight(key, p)}
                </p>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function TxRow({ tx }: any) {
  const hasSavings = tx.saving_pct > 0.01;
  return (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "9px 11px", borderRadius: 9, background: "#F8F7F3", border: "1.5px solid #ECEAE3" }}>
      <div>
        <div style={{ fontSize: 11, fontWeight: 700, marginBottom: 2 }}>{tx.tx_type}</div>
        <div style={{ fontSize: 9, color: "#A8A5A0", fontWeight: 500 }}>{tx.gas_units.toLocaleString()} gas</div>
      </div>
      <div style={{ textAlign: "right" }}>
        <div className="mono" style={{ fontSize: 15, fontWeight: 600 }}>{fmtU(tx.cost_now_usd)}</div>
        {hasSavings ? (
          <div style={{ display: "flex", alignItems: "center", gap: 4, justifyContent: "flex-end", marginTop: 2 }}>
            <ArrowDown size={9} color="#16a34a" />
            <span style={{ fontSize: 9, fontWeight: 700, color: "#16a34a" }}>Hemat {tx.saving_pct.toFixed(0)}%</span>
            <span style={{ fontSize: 9, color: "#B5B3AD" }}>({fmtU(tx.cost_wait_usd)} jika tunggu)</span>
          </div>
        ) : (
          <div style={{ fontSize: 9, color: "#16a34a", fontWeight: 700, marginTop: 2 }}>
            Harga Terbaik Saat Ini
          </div>
        )}
      </div>
    </div>
  );
}