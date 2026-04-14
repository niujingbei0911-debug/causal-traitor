import { useEffect, useRef, useState, useCallback } from "react";
import type { GameEvent } from "./types";

/**
 * 管理与后端 /ws/game WebSocket 连接的 hook.
 * 自动重连, 暴露最新事件列表和连接状态.
 */
export function useGameSocket(url: string) {
  const [events, setEvents] = useState<GameEvent[]>([]);
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout>>();

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;
    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => setConnected(true);

    ws.onmessage = (msg) => {
      try {
        const evt: GameEvent = JSON.parse(msg.data);
        setEvents((prev) => [...prev, evt]);
      } catch {
        /* 忽略非 JSON 消息 */
      }
    };

    ws.onclose = () => {
      setConnected(false);
      reconnectTimer.current = setTimeout(connect, 2000);
    };

    ws.onerror = () => ws.close();
  }, [url]);

  useEffect(() => {
    connect();
    return () => {
      clearTimeout(reconnectTimer.current);
      wsRef.current?.close();
    };
  }, [connect]);

  const clear = useCallback(() => setEvents([]), []);

  return { events, connected, clear } as const;
}
