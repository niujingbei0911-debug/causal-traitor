import React, { useMemo } from "react";
import { useGameSocket } from "./useGameSocket";
import CausalGraph from "./components/CausalGraph";
import DebatePanel from "./components/DebatePanel";
import JuryPanel from "./components/JuryPanel";
import DifficultyPanel from "./components/DifficultyPanel";
import type { CausalGraphData, JuryInfo, EvolutionPoint } from "./types";

/**
 * 主布局 — 四象限面板
 * ┌──────────┬──────────┐
 * │ CausalGraph │ DebatePanel │
 * ├──────────┼──────────┤
 * │ DifficultyPanel │ JuryPanel │
 * └──────────┴──────────┘
 */
const WS_URL = `ws://${window.location.host}/ws/game`;

export default function App() {
  const { events, connected } = useGameSocket(WS_URL);

  /* ── 从事件流中提取各面板所需数据 ── */
  const graphData = useMemo<CausalGraphData>(() => {
    // 取最新的 causal_graph 事件
    for (let i = events.length - 1; i >= 0; i--) {
      const ev = events[i];
      if (ev.data?.causal_graph) {
        return ev.data.causal_graph as CausalGraphData;
      }
    }
    return { nodes: [], links: [] };
  }, [events]);

  const juryInfo = useMemo<JuryInfo | null>(() => {
    for (let i = events.length - 1; i >= 0; i--) {
      const ev = events[i];
      if (ev.event_type === "jury" && ev.data?.votes) {
        return {
          votes: ev.data.votes,
          consensus: ev.data.consensus ?? 0,
          verdict: ev.data.verdict ?? "uncertain",
        } as JuryInfo;
      }
    }
    return null;
  }, [events]);

  const evolutionPoints = useMemo<EvolutionPoint[]>(() => {
    const pts: EvolutionPoint[] = [];
    for (const ev of events) {
      if (ev.data?.difficulty !== undefined && ev.round_id !== undefined) {
        pts.push({
          round_id: ev.round_id,
          difficulty: ev.data.difficulty ?? 0,
          dsr: ev.data.dsr ?? 0,
          strategy_diversity: ev.data.strategy_diversity ?? 0,
          arms_race_index: ev.data.arms_race_index ?? 0,
        });
      }
    }
    // 去重：同一 round_id 只保留最后一条
    const map = new Map<number, EvolutionPoint>();
    for (const p of pts) map.set(p.round_id, p);
    return Array.from(map.values()).sort((a, b) => a.round_id - b.round_id);
  }, [events]);

  return (
    <div className="h-screen w-screen flex flex-col bg-gray-900 text-gray-100 overflow-hidden">
      {/* 顶栏 */}
      <header className="flex items-center justify-between px-4 py-2 bg-gray-800 border-b border-gray-700 shrink-0">
        <h1 className="text-base font-bold tracking-wider">🎭 因果叛徒 — 实时可视化</h1>
        <span className={`text-xs px-2 py-0.5 rounded ${connected ? "bg-green-900 text-green-300" : "bg-red-900 text-red-300"}`}>
          {connected ? "● 已连接" : "○ 断开"}
        </span>
      </header>

      {/* 四象限 */}
      <main className="flex-1 grid grid-cols-2 grid-rows-2 gap-px bg-gray-700 min-h-0">
        <div className="bg-gray-900 overflow-hidden">
          <CausalGraph data={graphData} />
        </div>
        <div className="bg-gray-900 overflow-hidden">
          <DebatePanel events={events} />
        </div>
        <div className="bg-gray-900 overflow-hidden">
          <DifficultyPanel points={evolutionPoints} />
        </div>
        <div className="bg-gray-900 overflow-hidden">
          <JuryPanel jury={juryInfo} />
        </div>
      </main>
    </div>
  );
}
