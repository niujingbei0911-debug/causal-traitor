import React from "react";
import type { JuryVoteData, JuryInfo } from "../types";

/* ── 颜色 ── */
const VERDICT_COLOR: Record<string, string> = {
  traitor: "#ef4444",
  scientist: "#3b82f6",
  uncertain: "#6b7280",
};

/* ── 单张投票卡片 ── */
function VoteCard({ vote }: { vote: JuryVoteData }) {
  const color = VERDICT_COLOR[vote.vote] ?? VERDICT_COLOR.uncertain;
  return (
    <div
      className="rounded-lg p-3 text-sm"
      style={{ backgroundColor: `${color}14`, borderLeft: `3px solid ${color}` }}
    >
      <div className="flex justify-between items-center mb-1">
        <span className="font-semibold" style={{ color }}>
          {vote.model}
        </span>
        <span className="text-xs text-gray-500">
          置信度 {(vote.confidence * 100).toFixed(0)}%
        </span>
      </div>
      <p className="text-gray-300 text-xs">{vote.reasoning}</p>
    </div>
  );
}

/* ── 共识仪表盘 ── */
function ConsensusGauge({ value }: { value: number }) {
  const pct = Math.round(value * 100);
  const hue = value > 0.7 ? 142 : value > 0.4 ? 45 : 0; // green / yellow / red
  return (
    <div className="flex flex-col items-center gap-1">
      <svg viewBox="0 0 120 70" className="w-28">
        {/* 背景弧 */}
        <path
          d="M10 60 A50 50 0 0 1 110 60"
          fill="none"
          stroke="#374151"
          strokeWidth="8"
          strokeLinecap="round"
        />
        {/* 前景弧 */}
        <path
          d="M10 60 A50 50 0 0 1 110 60"
          fill="none"
          stroke={`hsl(${hue}, 70%, 55%)`}
          strokeWidth="8"
          strokeLinecap="round"
          strokeDasharray={`${value * 157} 157`}
        />
        <text x="60" y="58" textAnchor="middle" fill="white" fontSize="18" fontWeight="bold">
          {pct}%
        </text>
      </svg>
      <span className="text-xs text-gray-400">共识度</span>
    </div>
  );
}

/* ── 陪审团面板 ── */
export default function JuryPanel({ jury }: { jury: JuryInfo | null }) {
  if (!jury) {
    return (
      <section className="flex flex-col h-full items-center justify-center text-gray-600 text-sm">
        等待陪审团投票…
      </section>
    );
  }

  return (
    <section className="flex flex-col h-full">
      <header className="flex items-center justify-between px-4 py-2 border-b border-gray-700">
        <h2 className="text-sm font-bold tracking-wide">⚖️ 陪审团</h2>
        <span
          className="text-xs font-semibold px-2 py-0.5 rounded"
          style={{
            color: VERDICT_COLOR[jury.verdict] ?? "#fff",
            backgroundColor: `${VERDICT_COLOR[jury.verdict] ?? "#6b7280"}22`,
          }}
        >
          {jury.verdict === "traitor" ? "🔴 叛徒" : jury.verdict === "scientist" ? "🔵 科学家" : "⚪ 未定"}
        </span>
      </header>

      {/* 共识仪表 */}
      <div className="flex justify-center py-3">
        <ConsensusGauge value={jury.consensus} />
      </div>

      {/* 投票列表 */}
      <div className="flex-1 overflow-y-auto px-4 pb-3 space-y-2 custom-scrollbar">
        {jury.votes.map((v, i) => (
          <VoteCard key={i} vote={v} />
        ))}
      </div>
    </section>
  );
}
