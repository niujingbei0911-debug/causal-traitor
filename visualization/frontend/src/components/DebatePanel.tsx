import React, { useRef, useEffect } from "react";
import type { GameEvent } from "../types";

/* ── 角色颜色映射 ── */
const ROLE_STYLE: Record<string, { color: string; label: string }> = {
  traitor:   { color: "#ef4444", label: "叛徒 A" },
  scientist: { color: "#3b82f6", label: "科学家 B" },
  auditor:   { color: "#a855f7", label: "审计员 C" },
  jury:      { color: "#f59e0b", label: "陪审团" },
  system:    { color: "#6b7280", label: "系统" },
};

function roleMeta(role: string) {
  return ROLE_STYLE[role] ?? ROLE_STYLE.system;
}

/* ── 单条消息气泡 ── */
function Bubble({ ev }: { ev: GameEvent }) {
  const role = (ev.data?.role as string) ?? "system";
  const meta = roleMeta(role);
  const isSystem = ev.event_type === "system" || role === "system";

  if (isSystem) {
    return (
      <div className="text-center text-xs text-gray-500 my-2 italic">
        {(ev.data?.narrative as string) ?? ev.event_type}
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-0.5 mb-3">
      {/* 角色标签 + 时间 */}
      <div className="flex items-center gap-2 text-xs">
        <span className="font-semibold" style={{ color: meta.color }}>
          {meta.label}
        </span>
        <span className="text-gray-600">
          R{ev.round_id ?? "-"}
        </span>
      </div>

      {/* 消息体 */}
      <div
        className="rounded-lg px-3 py-2 text-sm leading-relaxed max-w-[90%]"
        style={{
          backgroundColor: `${meta.color}18`,
          borderLeft: `3px solid ${meta.color}`,
        }}
      >
        {ev.data?.claim && (
          <p className="font-medium mb-1">
            🔗 因果声明: {ev.data.claim as string}
          </p>
        )}
        {ev.data?.narrative && (
          <p className="whitespace-pre-wrap">{ev.data.narrative as string}</p>
        )}
        {ev.data?.reasoning && (
          <p className="text-gray-400 text-xs mt-1">
            💡 {ev.data.reasoning as string}
          </p>
        )}
        {ev.data?.verdict && (
          <p className="mt-1">
            ⚖️ 裁定: <strong>{ev.data.verdict as string}</strong>
          </p>
        )}
        {/* 工具执行结果 */}
        {ev.data?.tool_results && (
          <pre className="mt-1 text-xs bg-gray-900 rounded p-2 overflow-x-auto">
            {JSON.stringify(ev.data.tool_results, null, 2)}
          </pre>
        )}
      </div>
    </div>
  );
}

/* ── 辩论流面板 ── */
export default function DebatePanel({ events }: { events: GameEvent[] }) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [events.length]);

  return (
    <section className="flex flex-col h-full">
      {/* 标题栏 */}
      <header className="flex items-center justify-between px-4 py-2 border-b border-gray-700">
        <h2 className="text-sm font-bold tracking-wide">💬 辩论流</h2>
        <span className="text-xs text-gray-500">{events.length} 条消息</span>
      </header>

      {/* 消息列表 */}
      <div className="flex-1 overflow-y-auto px-4 py-3 custom-scrollbar">
        {events.length === 0 ? (
          <p className="text-gray-600 text-sm text-center mt-8">
            等待辩论开始…
          </p>
        ) : (
          events.map((ev, i) => <Bubble key={i} ev={ev} />)
        )}
        <div ref={bottomRef} />
      </div>
    </section>
  );
}
