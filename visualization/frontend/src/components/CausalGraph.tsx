import { useEffect, useRef } from "react";
import * as d3 from "d3";
import dagre from "@dagrejs/dagre";
import type { CausalGraphData, CausalNode, CausalLink } from "../types";

/** 节点颜色映射 */
const NODE_COLOR: Record<CausalNode["type"], string> = {
  claimed: "#ef4444",   // 红 — Agent A 声称
  verified: "#22c55e",  // 绿 — Agent B/C 验证
  hidden: "#a855f7",    // 紫 — 隐变量
};

const LINK_DASH: Record<CausalLink["type"], string> = {
  claimed: "6,3",
  verified: "0",
  hidden: "4,4",
};

interface Props {
  data: CausalGraphData;
  width?: number;
  height?: number;
}

/**
 * 因果图面板 — 使用 dagre 做 DAG 布局, D3.js 渲染 SVG.
 * 隐变量用虚线标注, Agent A 声称的关系用红色, 验证的用绿色.
 */
export default function CausalGraph({
  data,
  width = 420,
  height = 360,
}: Props) {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current || data.nodes.length === 0) return;

    /* ---------- dagre 布局 ---------- */
    const g = new dagre.graphlib.Graph();
    g.setGraph({ rankdir: "TB", nodesep: 60, ranksep: 50 });
    g.setDefaultEdgeLabel(() => ({}));

    data.nodes.forEach((n) =>
      g.setNode(n.id, { label: n.label, width: 80, height: 36 }),
    );
    data.links.forEach((l) => g.setEdge(l.source, l.target));
    dagre.layout(g);

    /* ---------- D3 渲染 ---------- */
    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    // 箭头 marker
    const defs = svg.append("defs");
    (["claimed", "verified", "hidden"] as const).forEach((t) => {
      defs
        .append("marker")
        .attr("id", `arrow-${t}`)
        .attr("viewBox", "0 0 10 10")
        .attr("refX", 10)
        .attr("refY", 5)
        .attr("markerWidth", 8)
        .attr("markerHeight", 8)
        .attr("orient", "auto")
        .append("path")
        .attr("d", "M0,0 L10,5 L0,10 Z")
        .attr("fill", NODE_COLOR[t]);
    });

    const container = svg.append("g");

    // 边
    const linkMap = new Map(data.links.map((l) => [`${l.source}-${l.target}`, l]));
    g.edges().forEach((e) => {
      const pts = g.edge(e).points as { x: number; y: number }[];
      const key = `${e.v}-${e.w}`;
      const link = linkMap.get(key);
      const t = link?.type ?? "claimed";
      const line = d3.line<{ x: number; y: number }>()
        .x((d) => d.x)
        .y((d) => d.y)
        .curve(d3.curveBasis);

      container
        .append("path")
        .attr("d", line(pts))
        .attr("fill", "none")
        .attr("stroke", NODE_COLOR[t])
        .attr("stroke-width", 2)
        .attr("stroke-dasharray", LINK_DASH[t])
        .attr("marker-end", `url(#arrow-${t})`);
    });

    // 节点
    const nodeMap = new Map(data.nodes.map((n) => [n.id, n]));
    g.nodes().forEach((nId) => {
      const pos = g.node(nId);
      const node = nodeMap.get(nId);
      if (!pos || !node) return;

      const group = container.append("g").attr("transform", `translate(${pos.x},${pos.y})`);

      group
        .append("rect")
        .attr("x", -40)
        .attr("y", -18)
        .attr("width", 80)
        .attr("height", 36)
        .attr("rx", 6)
        .attr("fill", "#1f2937")
        .attr("stroke", NODE_COLOR[node.type])
        .attr("stroke-width", 2)
        .attr("stroke-dasharray", node.type === "hidden" ? "4,4" : "0");

      group
        .append("text")
        .attr("text-anchor", "middle")
        .attr("dy", "0.35em")
        .attr("fill", NODE_COLOR[node.type])
        .attr("font-size", 13)
        .text(node.label);
    });

    // 自动缩放
    const bbox = (container.node() as SVGGElement).getBBox();
    const pad = 20;
    svg.attr(
      "viewBox",
      `${bbox.x - pad} ${bbox.y - pad} ${bbox.width + pad * 2} ${bbox.height + pad * 2}`,
    );
  }, [data, width, height]);

  return (
    <div className="rounded-lg border border-gray-700 bg-gray-800 p-3">
      <h3 className="mb-2 text-sm font-semibold text-gray-300">
        因果图可视化
      </h3>
      <div className="flex items-center gap-3 mb-2 text-xs text-gray-400">
        <span className="flex items-center gap-1">
          <span className="inline-block w-3 h-3 rounded-sm bg-traitor" />
          Agent A 声称
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block w-3 h-3 rounded-sm bg-green-500" />
          已验证
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block w-3 h-3 rounded-sm bg-purple-500 opacity-60" />
          隐变量
        </span>
      </div>
      {data.nodes.length === 0 ? (
        <p className="text-center text-gray-500 py-8 text-sm">等待因果图数据…</p>
      ) : (
        <svg
          ref={svgRef}
          width={width}
          height={height}
          className="w-full"
          preserveAspectRatio="xMidYMid meet"
        />
      )}
    </div>
  );
}
