import { useEffect, useRef, useState } from "react";
import * as d3 from "d3";
import dagre from "@dagrejs/dagre";
import type { CausalGraphData, CausalNode, CausalLink, CausalLevel } from "../types";

/* ================================================================
 * 双色模式: 验证状态 (claimed/verified/hidden) vs 因果层级 (L1/L2/L3)
 * ================================================================ */

type ColorMode = "status" | "level";

/** 验证状态三色 */
const STATUS_COLOR: Record<CausalNode["type"], string> = {
  claimed: "#ef4444",   // 红 — Agent A 声称
  verified: "#22c55e",  // 绿 — Agent B/C 验证
  hidden: "#a855f7",    // 紫 — 隐变量
};

/** Pearl 因果层级三色 */
const LEVEL_COLOR: Record<CausalLevel, string> = {
  1: "#3b82f6",  // 蓝 — L1 关联 (Association)
  2: "#f97316",  // 橙 — L2 干预 (Intervention)
  3: "#ef4444",  // 红 — L3 反事实 (Counterfactual)
};

const LEVEL_LABEL: Record<CausalLevel, string> = {
  1: "L1 关联",
  2: "L2 干预",
  3: "L3 反事实",
};

const LINK_DASH: Record<CausalLink["type"], string> = {
  claimed: "6,3",
  verified: "0",
  hidden: "4,4",
};

/** 根据当前色彩模式获取节点颜色 */
function getNodeColor(node: CausalNode, mode: ColorMode, graphLevel?: CausalLevel): string {
  if (mode === "level") {
    const lvl = node.causal_level ?? graphLevel ?? 1;
    return LEVEL_COLOR[lvl as CausalLevel] ?? LEVEL_COLOR[1];
  }
  return STATUS_COLOR[node.type];
}

/** 根据当前色彩模式获取边颜色 */
function getLinkColor(link: CausalLink, mode: ColorMode, graphLevel?: CausalLevel): string {
  if (mode === "level") {
    const lvl = link.causal_level ?? graphLevel ?? 1;
    return LEVEL_COLOR[lvl as CausalLevel] ?? LEVEL_COLOR[1];
  }
  return STATUS_COLOR[link.type];
}

interface Props {
  data: CausalGraphData;
  width?: number;
  height?: number;
}

/**
 * 因果图面板 — 使用 dagre 做 DAG 布局, D3.js 渲染 SVG.
 * 支持双色模式切换: 验证状态三色 / Pearl 因果层级三色.
 */
export default function CausalGraph({
  data,
  width = 420,
  height = 360,
}: Props) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [colorMode, setColorMode] = useState<ColorMode>("status");

  useEffect(() => {
    if (!svgRef.current || data.nodes.length === 0) return;

    const graphLevel = data.causal_level;

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

    // 箭头 marker — 为每种可能的颜色创建
    const defs = svg.append("defs");
    const allColors = new Set<string>();
    data.links.forEach((l) => allColors.add(getLinkColor(l, colorMode, graphLevel)));
    data.nodes.forEach((n) => allColors.add(getNodeColor(n, colorMode, graphLevel)));
    allColors.forEach((color) => {
      const safeId = color.replace("#", "c");
      defs
        .append("marker")
        .attr("id", `arrow-${safeId}`)
        .attr("viewBox", "0 0 10 10")
        .attr("refX", 10)
        .attr("refY", 5)
        .attr("markerWidth", 8)
        .attr("markerHeight", 8)
        .attr("orient", "auto")
        .append("path")
        .attr("d", "M0,0 L10,5 L0,10 Z")
        .attr("fill", color);
    });

    const container = svg.append("g");

    // 边
    const linkMap = new Map(data.links.map((l) => [`${l.source}-${l.target}`, l]));
    g.edges().forEach((e) => {
      const pts = g.edge(e).points as { x: number; y: number }[];
      const key = `${e.v}-${e.w}`;
      const link = linkMap.get(key);
      if (!link) return;
      const color = getLinkColor(link, colorMode, graphLevel);
      const safeId = color.replace("#", "c");
      const line = d3.line<{ x: number; y: number }>()
        .x((d) => d.x)
        .y((d) => d.y)
        .curve(d3.curveBasis);

      container
        .append("path")
        .attr("d", line(pts))
        .attr("fill", "none")
        .attr("stroke", color)
        .attr("stroke-width", 2)
        .attr("stroke-dasharray", LINK_DASH[link.type])
        .attr("marker-end", `url(#arrow-${safeId})`);
    });

    // 节点
    const nodeMap = new Map(data.nodes.map((n) => [n.id, n]));
    g.nodes().forEach((nId) => {
      const pos = g.node(nId);
      const node = nodeMap.get(nId);
      if (!pos || !node) return;

      const color = getNodeColor(node, colorMode, graphLevel);
      const group = container.append("g").attr("transform", `translate(${pos.x},${pos.y})`);

      group
        .append("rect")
        .attr("x", -40)
        .attr("y", -18)
        .attr("width", 80)
        .attr("height", 36)
        .attr("rx", 6)
        .attr("fill", "#1f2937")
        .attr("stroke", color)
        .attr("stroke-width", 2)
        .attr("stroke-dasharray", node.type === "hidden" ? "4,4" : "0");

      group
        .append("text")
        .attr("text-anchor", "middle")
        .attr("dy", "0.35em")
        .attr("fill", color)
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
  }, [data, width, height, colorMode]);

  return (
    <div className="rounded-lg border border-gray-700 bg-gray-800 p-3">
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-sm font-semibold text-gray-300">因果图可视化</h3>
        {/* 色彩模式切换按钮 */}
        <button
          type="button"
          onClick={() => setColorMode((m) => (m === "status" ? "level" : "status"))}
          className="px-2 py-0.5 text-xs rounded border border-gray-600 text-gray-300 hover:bg-gray-700 transition-colors"
          aria-label="切换因果图色彩模式"
        >
          {colorMode === "status" ? "🔬 验证状态" : "📊 因果层级"}
        </button>
      </div>

      {/* 图例 — 根据色彩模式动态切换 */}
      <div className="flex items-center gap-3 mb-2 text-xs text-gray-400">
        {colorMode === "status" ? (
          <>
            <span className="flex items-center gap-1">
              <span className="inline-block w-3 h-3 rounded-sm" style={{ backgroundColor: STATUS_COLOR.claimed }} />
              Agent A 声称
            </span>
            <span className="flex items-center gap-1">
              <span className="inline-block w-3 h-3 rounded-sm" style={{ backgroundColor: STATUS_COLOR.verified }} />
              已验证
            </span>
            <span className="flex items-center gap-1">
              <span className="inline-block w-3 h-3 rounded-sm opacity-60" style={{ backgroundColor: STATUS_COLOR.hidden }} />
              隐变量
            </span>
          </>
        ) : (
          <>
            {([1, 2, 3] as CausalLevel[]).map((lvl) => (
              <span key={lvl} className="flex items-center gap-1">
                <span
                  className="inline-block w-3 h-3 rounded-sm"
                  style={{ backgroundColor: LEVEL_COLOR[lvl] }}
                />
                {LEVEL_LABEL[lvl]}
              </span>
            ))}
          </>
        )}
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
