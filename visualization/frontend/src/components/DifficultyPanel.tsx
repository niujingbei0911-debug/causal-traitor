import React from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import type { EvolutionPoint } from "../types";

/* ── 难度仪表 ── */
function DifficultyGauge({ value }: { value: number }) {
  const pct = Math.round(value * 100);
  // 低难度绿色 → 中等黄色 → 高难度红色
  const hue = Math.round((1 - value) * 120);
  const radius = 50;
  const circumference = Math.PI * radius; // 半圆
  const offset = circumference * (1 - value);

  return (
    <div className="flex flex-col items-center gap-1">
      <svg viewBox="0 0 120 70" className="w-32">
        <path
          d="M10 60 A50 50 0 0 1 110 60"
          fill="none"
          stroke="#374151"
          strokeWidth="8"
          strokeLinecap="round"
        />
        <path
          d="M10 60 A50 50 0 0 1 110 60"
          fill="none"
          stroke={`hsl(${hue}, 75%, 50%)`}
          strokeWidth="8"
          strokeLinecap="round"
          strokeDasharray={`${value * 157} 157`}
        />
        <text
          x="60"
          y="55"
          textAnchor="middle"
          fill="white"
          fontSize="16"
          fontWeight="bold"
        >
          {pct}%
        </text>
      </svg>
      <span className="text-xs text-gray-400">当前难度</span>
    </div>
  );
}

/* ── 进化轨迹图 ── */
const LINE_CFG = [
  { key: "difficulty", color: "#f59e0b", label: "难度" },
  { key: "dsr", color: "#ef4444", label: "欺骗成功率" },
  { key: "strategy_diversity", color: "#3b82f6", label: "策略多样性" },
  { key: "arms_race_index", color: "#a855f7", label: "军备竞赛指数" },
] as const;

/* ── 主面板 ── */
export default function DifficultyPanel({
  points,
}: {
  points: EvolutionPoint[];
}) {
  const latest = points.length > 0 ? points[points.length - 1] : null;

  return (
    <section className="flex flex-col h-full">
      <header className="px-4 py-2 border-b border-gray-700">
        <h2 className="text-sm font-bold tracking-wide">📈 难度 &amp; 进化</h2>
      </header>

      {/* 仪表 + 关键数值 */}
      <div className="flex items-center justify-around py-3 px-4">
        <DifficultyGauge value={latest?.difficulty ?? 0} />
        {latest && (
          <div className="grid grid-cols-2 gap-x-6 gap-y-1 text-xs">
            <Stat label="DSR" value={latest.dsr} color="#ef4444" />
            <Stat label="策略多样性" value={latest.strategy_diversity} color="#3b82f6" />
            <Stat label="军备指数" value={latest.arms_race_index} color="#a855f7" />
            <Stat label="回合" value={latest.round_id} plain />
          </div>
        )}
      </div>

      {/* 折线图 */}
      <div className="flex-1 min-h-0 px-2 pb-2">
        {points.length < 2 ? (
          <div className="flex items-center justify-center h-full text-gray-600 text-xs">
            至少需要 2 个回合数据…
          </div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={points} margin={{ top: 4, right: 12, bottom: 4, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis
                dataKey="round_id"
                tick={{ fill: "#9ca3af", fontSize: 10 }}
                label={{ value: "回合", position: "insideBottomRight", fill: "#9ca3af", fontSize: 10 }}
              />
              <YAxis
                domain={[0, 1]}
                tick={{ fill: "#9ca3af", fontSize: 10 }}
                width={30}
              />
              <Tooltip
                contentStyle={{ backgroundColor: "#1f2937", border: "1px solid #374151", fontSize: 11 }}
                labelFormatter={(v) => `回合 ${v}`}
              />
              <Legend
                wrapperStyle={{ fontSize: 10 }}
              />
              {LINE_CFG.map((cfg) => (
                <Line
                  key={cfg.key}
                  type="monotone"
                  dataKey={cfg.key}
                  name={cfg.label}
                  stroke={cfg.color}
                  strokeWidth={1.5}
                  dot={false}
                  activeDot={{ r: 3 }}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>
    </section>
  );
}

/* ── 小数值展示 ── */
function Stat({
  label,
  value,
  color,
  plain,
}: {
  label: string;
  value: number;
  color?: string;
  plain?: boolean;
}) {
  return (
    <div className="flex items-center gap-1">
      {color && <span className="w-2 h-2 rounded-full" style={{ backgroundColor: color }} />}
      <span className="text-gray-400">{label}</span>
      <span className="font-semibold text-gray-200">
        {plain ? value : `${(value * 100).toFixed(0)}%`}
      </span>
    </div>
  );
}
