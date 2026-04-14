/* ---- 因果图 ---- */
/** Pearl 因果层级: 1=关联, 2=干预, 3=反事实 */
export type CausalLevel = 1 | 2 | 3;

export interface CausalNode {
  id: string;
  label: string;
  type: "claimed" | "verified" | "hidden";
  causal_level?: CausalLevel;
}

export interface CausalLink {
  source: string;
  target: string;
  type: "claimed" | "verified" | "hidden";
  causal_level?: CausalLevel;
}

export interface CausalGraphData {
  nodes: CausalNode[];
  links: CausalLink[];
  /** 场景整体因果层级 */
  causal_level?: CausalLevel;
}

/* ---- 辩论事件 ---- */
export type EventKind =
  | "round_start"
  | "claim"
  | "detection"
  | "verdict"
  | "jury"
  | "system"
  | "game_end";

export interface GameEvent {
  event_type: EventKind;
  round_id: number;
  /** data 内含 role / claim / narrative 等动态字段 */
  data: Record<string, any>;
  timestamp: string;
}

/* ---- 陪审团 ---- */
export interface JuryVoteData {
  juror_id: string;
  vote: string;
  confidence: number;
  model: string;
  reasoning?: string;
}

export interface JuryInfo {
  votes: JuryVoteData[];
  consensus: number;
  verdict: string;
}

/* ---- 进化 / 难度 ---- */
export interface EvolutionPoint {
  round_id: number;
  difficulty: number;
  dsr: number;
  strategy_diversity: number;
  arms_race_index: number;
}

/* ---- Dashboard ---- */
export interface MetricEntry {
  name: string;
  value: number;
  category: string;
}

export interface DashboardData {
  experiment_id: string;
  games: number;
  metrics: MetricEntry[];
  scores: Record<string, number>;
}
